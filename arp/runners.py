"""Runners: the things that actually spend compute and return a number.

A runner is anything that can take a `TrialSpec` (parameters + seed + grid cell)
and return a `TrialResult` (metric + aux). Three ship with the platform:

* `SimulatedRunner` — a seeded synthetic response surface. Used by the tests and
  by `arp demo`, so the whole engine can be exercised on a laptop with no GPU.
* `CommandRunner` — runs any shell command, injects parameters as environment
  variables and format placeholders, and parses `key: value` lines from stdout.
  This is how you point the platform at a subject it has never seen.
* `TrainRunner` (in `arp.adapters.autoresearch_train`) — the bridge to this
  repo's existing autoresearch loop: patches constants in `train.py`, runs it,
  reads back `val_bpb`.
"""

import math
import os
import random
import re
import shlex
import subprocess
import time
from typing import Any, Callable, Dict, Optional

from arp.models import Subject, TrialResult, TrialSpec


class Runner:
    name = "runner"

    def __init__(self, subject: Subject, work_dir: str = ".", **kwargs):
        self.subject = subject
        self.work_dir = work_dir
        self.config: Dict[str, Any] = dict(subject.config.get("runner_config", {}))
        self.config.update(kwargs)

    def run(self, spec: TrialSpec) -> TrialResult:  # pragma: no cover - interface
        raise NotImplementedError

    # Handy for subclasses: full parameter set with declared defaults filled in.
    def resolved_params(self, spec: TrialSpec) -> Dict[str, Any]:
        """Declared defaults, then the subject's baseline, then the spec.

        The spec always wins, on both arms. A baseline trial is not "the
        original defaults": it is whatever the subject's baseline currently is,
        at the epoch and grid cell this trial belongs to, and the caller has
        already worked that out.
        """
        params = {
            name: spec_.get("default")
            for name, spec_ in self.subject.param_space.items()
            if "default" in spec_
        }
        params.update(self.subject.baseline_params)
        params.update(spec.params)
        return params


# ---------------------------------------------------------------------------
# Simulated runner
# ---------------------------------------------------------------------------

class SimulatedRunner(Runner):
    """A deterministic, seeded response surface.

    The subject declares the ground truth under `config["simulation"]`:

        {"baseline": 1.0,              # metric at default parameters
         "noise": 0.01,                # relative stdev of measurement noise
         "coefs": {"lr": -0.04},       # metric response per unit of normalised deviation
         "curvature": {"lr": 0.05},    # quadratic penalty, so extremes stop helping
         "cell_shift": {"scale": {"large": 0.01}}}

    Same seed, same parameters, same number — which is what lets the tests assert
    that the decision layer finds real effects and rejects null ones.
    """

    name = "simulated"

    def run(self, spec: TrialSpec) -> TrialResult:
        t0 = time.time()
        sim = self.subject.config.get("simulation", {})
        baseline = float(sim.get("baseline", 1.0))
        noise = float(sim.get("noise", 0.01))
        coefs: Dict[str, float] = sim.get("coefs", {})
        curvature: Dict[str, float] = sim.get("curvature", {})
        params = self.resolved_params(spec)

        shift = 0.0
        for name, coef in coefs.items():
            dev = self._deviation(name, params.get(name))
            shift += coef * dev - float(curvature.get(name, 0.0)) * dev * dev

        for axis, value in (spec.cell or {}).items():
            table = sim.get("cell_shift", {}).get(axis, {})
            shift += float(table.get(str(value), table.get(value, 0.0)) or 0.0)

        # Out-of-range parameters "crash", the way a too-large model OOMs.
        for name, value in params.items():
            pspec = self.subject.param_space.get(name, {})
            low, high = pspec.get("low"), pspec.get("high")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if (low is not None and value < low) or (high is not None and value > high):
                    return TrialResult(
                        ok=False, error=f"{name}={value} outside [{low}, {high}]",
                        duration_s=time.time() - t0,
                    )

        rng = random.Random(spec.seed)
        value = baseline * (1.0 + shift) + rng.gauss(0.0, noise * abs(baseline))
        if float(sim.get("latency_s", 0.0)) > 0:
            time.sleep(float(sim["latency_s"]))
        return TrialResult(
            ok=True,
            metric=value,
            aux={"shift": shift, "seed": spec.seed, "params": params},
            duration_s=time.time() - t0,
        )

    def _deviation(self, name: str, value: Any) -> float:
        """Normalised distance from the parameter's default."""
        pspec = self.subject.param_space.get(name, {})
        default = pspec.get("default")
        if value is None or default is None:
            return 0.0
        kind = pspec.get("type", "float")
        if kind == "float":
            if pspec.get("log", True) and value > 0 and default > 0:
                return math.log(float(value) / float(default))
            scale = float(pspec.get("scale", abs(float(default)) or 1.0))
            return (float(value) - float(default)) / scale
        if kind == "int":
            step = float(pspec.get("step", 1)) or 1.0
            return (float(value) - float(default)) / step
        return 0.0 if value == default else 1.0


# ---------------------------------------------------------------------------
# Generic command runner
# ---------------------------------------------------------------------------

METRIC_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*[:=]\s*([-+0-9.eE]+)\s*$", re.MULTILINE)


class CommandRunner(Runner):
    """Runs a shell command and scrapes `key: value` lines out of its output.

    runner_config keys:
        command       — the command, with `{param}` placeholders substituted
        cwd           — working directory (default: the repo root)
        metric_key    — which parsed key is the subject's metric
        env_prefix    — parameters are also exported as PREFIX_UPPERNAME
        aux_keys      — other parsed keys worth keeping (peak_vram_mb, ...)
        timeout_s     — hard kill after this long
    """

    name = "command"

    def run(self, spec: TrialSpec) -> TrialResult:
        t0 = time.time()
        params = self.resolved_params(spec)
        command = self.config.get("command")
        if not command:
            return TrialResult(ok=False, error="runner_config.command is not set")

        try:
            rendered = command.format(**params, seed=spec.seed, arm=spec.arm, **spec.cell)
        except KeyError as exc:
            return TrialResult(ok=False, error=f"command placeholder {exc} has no value")

        env = os.environ.copy()
        prefix = self.config.get("env_prefix", "ARP_PARAM_")
        for name, value in params.items():
            env[f"{prefix}{name.upper()}"] = str(value)
        env[f"{prefix}SEED"] = str(spec.seed)
        env[f"{prefix}ARM"] = spec.arm
        for axis, value in (spec.cell or {}).items():
            env[f"{prefix}CELL_{axis.upper()}"] = str(value)

        timeout = float(self.config.get("timeout_s", spec.timeout_s))
        cwd = self.config.get("cwd", self.work_dir)
        log_path = os.path.join(self.work_dir, f"trial_{spec.seed}.log")
        try:
            proc = subprocess.run(
                rendered if self.config.get("shell", True) else shlex.split(rendered),
                shell=bool(self.config.get("shell", True)),
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TrialResult(ok=False, error=f"timeout after {timeout:.0f}s", duration_s=time.time() - t0)
        except OSError as exc:
            return TrialResult(ok=False, error=f"failed to launch: {exc}", duration_s=time.time() - t0)

        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        try:
            os.makedirs(self.work_dir, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(output)
        except OSError:
            log_path = ""

        parsed = parse_metrics(output)
        metric_key = self.config.get("metric_key", self.subject.metric)
        if proc.returncode != 0 and metric_key not in parsed:
            tail = "\n".join(output.strip().splitlines()[-15:])
            return TrialResult(
                ok=False, error=f"exit {proc.returncode}: {tail}",
                duration_s=time.time() - t0, log_path=log_path,
            )
        if metric_key not in parsed:
            return TrialResult(
                ok=False, error=f"metric {metric_key!r} not found in output",
                duration_s=time.time() - t0, log_path=log_path,
            )

        aux_keys = self.config.get("aux_keys") or [k for k in parsed if k != metric_key]
        return TrialResult(
            ok=True,
            metric=parsed[metric_key],
            aux={k: parsed[k] for k in aux_keys if k in parsed},
            duration_s=time.time() - t0,
            log_path=log_path,
        )


def parse_metrics(text: str) -> Dict[str, float]:
    """Pull `key: value` / `key = value` numeric pairs out of arbitrary output."""
    out: Dict[str, float] = {}
    for key, value in METRIC_LINE.findall(text or ""):
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _train_runner(subject: Subject, work_dir: str, **kwargs) -> Runner:
    from arp.adapters.autoresearch_train import TrainRunner

    return TrainRunner(subject, work_dir, **kwargs)


REGISTRY: Dict[str, Callable[..., Runner]] = {
    "simulated": SimulatedRunner,
    "command": CommandRunner,
    "train": _train_runner,
}


def build_runner(subject: Subject, work_dir: str = ".", override: Optional[str] = None, **kwargs) -> Runner:
    key = override or subject.runner or "simulated"
    if key not in REGISTRY:
        raise KeyError(f"unknown runner {key!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[key](subject, work_dir, **kwargs)
