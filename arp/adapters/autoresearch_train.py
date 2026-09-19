"""Bridge to this repo's existing autoresearch loop.

`train.py` is a single file whose top-level constants are the experiment's knobs
and whose stdout ends in a `key: value` summary block. That is already a runner
interface — it just needs someone to set the constants, seed it, run it and read
the number back. This module does that, without touching `train.py` itself: each
trial gets a patched copy written next to it (so `from prepare import ...` still
resolves), which is deleted afterwards.

The subject built by `autoresearch_subject()` mirrors the constants that the
agent in `program.md` is told to play with.
"""

import os
import re
from typing import Any, Dict, Optional

from arp.config import REPO_ROOT
from arp.models import Subject, TrialResult, TrialSpec
from arp.runners import CommandRunner, parse_metrics


def patch_constants(source: str, params: Dict[str, Any]) -> str:
    """Rewrite top-level `NAME = value` assignments, preserving trailing comments.

    Only assignments that already exist are rewritten; an unknown parameter name
    is an error rather than a silently ignored typo, because a silently ignored
    typo turns into an experiment that measured nothing.
    """
    missing = []
    out = source
    for name, value in params.items():
        pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]+)(#.*)?$", re.MULTILINE)
        if not pattern.search(out):
            missing.append(name)
            continue
        literal = repr(value) if isinstance(value, str) else str(value)

        def replace(match: "re.Match") -> str:
            comment = match.group(3)
            if not comment:
                return f"{match.group(1)}{literal}"
            # Keep the comment in its original column: these files are read by
            # humans between runs, and a patched copy should still look like the
            # file it came from.
            width = max(len(match.group(2)), len(literal) + 1)
            return f"{match.group(1)}{literal.ljust(width)}{comment}"

        out = pattern.sub(replace, out, count=1)
    if missing:
        raise KeyError(f"no top-level constant(s) to patch: {', '.join(sorted(missing))}")
    return out


def patch_seed(source: str, seed: int) -> str:
    """Point every manual_seed call at this trial's seed.

    Without this, replicates are not replicates: `train.py` pins seed 42, so the
    only variation between runs would be nondeterministic kernels.
    """
    return re.sub(r"manual_seed\(\s*\d+\s*\)", f"manual_seed({seed})", source)


class TrainRunner(CommandRunner):
    """Runs a patched copy of `train.py` and reads back `val_bpb`."""

    name = "train"

    def run(self, spec: TrialSpec) -> TrialResult:
        repo = self.config.get("repo_root", REPO_ROOT)
        script = self.config.get("script", "train.py")
        source_path = os.path.join(repo, script)
        try:
            with open(source_path) as f:
                source = f.read()
        except OSError as exc:
            return TrialResult(ok=False, error=f"cannot read {source_path}: {exc}")

        params = self.resolved_params(spec)
        params.update(_cell_overrides(self.subject, spec.cell))
        try:
            patched = patch_seed(patch_constants(source, params), spec.seed)
        except KeyError as exc:
            return TrialResult(ok=False, error=str(exc))

        temp_name = f".arp_train_{spec.seed}.py"
        temp_path = os.path.join(repo, temp_name)
        try:
            with open(temp_path, "w") as f:
                f.write(patched)
            # Reuse CommandRunner's process handling, parsing and logging.
            self.config = dict(self.config)
            self.config.setdefault("metric_key", self.subject.metric)
            self.config.setdefault("aux_keys", ["peak_vram_mb", "mfu_percent", "num_steps", "num_params_M"])
            self.config["cwd"] = repo
            self.config["command"] = f"{self.config.get('launcher', 'uv run')} {temp_name}"
            result = super().run(spec)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return result


def _cell_overrides(subject: Subject, cell: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate exhaustive-grid coordinates into parameter overrides."""
    overrides: Dict[str, Any] = {}
    for axis, label in (cell or {}).items():
        spec = subject.stress_axes.get(axis)
        if isinstance(spec, dict) and isinstance(spec.get(label), dict):
            overrides.update(spec[label])
    return overrides


# ---------------------------------------------------------------------------
# The ready-made subject
# ---------------------------------------------------------------------------

def autoresearch_subject(slug: str = "nanochat-pretrain") -> Subject:
    """The default subject: minimise val_bpb inside the fixed 5-minute budget."""
    return Subject(
        slug=slug,
        title="Minimise val_bpb on the autoresearch 5-minute pretraining budget",
        metric="val_bpb",
        direction="minimize",
        runner="train",
        description=(
            "Single-GPU nanochat-style pretraining with a fixed wall-clock budget. "
            "Architecture, optimiser and schedule are all fair game; peak VRAM is a "
            "soft constraint and simpler code wins ties."
        ),
        config={
            "param_space": {
                "MATRIX_LR": {"type": "float", "default": 0.04, "low": 0.005, "high": 0.2, "log": True, "group": "lr", "step": 0.25},
                "EMBEDDING_LR": {"type": "float", "default": 0.6, "low": 0.05, "high": 3.0, "log": True, "group": "lr", "step": 0.25},
                "UNEMBEDDING_LR": {"type": "float", "default": 0.004, "low": 0.0005, "high": 0.05, "log": True, "group": "lr", "step": 0.3},
                "SCALAR_LR": {"type": "float", "default": 0.5, "low": 0.05, "high": 3.0, "log": True, "group": "lr", "step": 0.3},
                "WEIGHT_DECAY": {"type": "float", "default": 0.2, "low": 0.0, "high": 1.0, "log": False, "scale": 0.2, "group": "regularisation", "step": 0.3},
                "WARMUP_RATIO": {"type": "float", "default": 0.0, "low": 0.0, "high": 0.3, "log": False, "scale": 0.05, "group": "schedule", "step": 0.5},
                "WARMDOWN_RATIO": {"type": "float", "default": 0.5, "low": 0.1, "high": 0.9, "log": False, "scale": 0.1, "group": "schedule", "step": 0.3},
                "FINAL_LR_FRAC": {"type": "float", "default": 0.0, "low": 0.0, "high": 0.5, "log": False, "scale": 0.05, "group": "schedule", "step": 0.5},
                "DEPTH": {"type": "int", "default": 8, "low": 4, "high": 16, "step": 1, "group": "architecture"},
                "ASPECT_RATIO": {"type": "int", "default": 64, "low": 32, "high": 128, "step": 8, "group": "architecture"},
                "HEAD_DIM": {"type": "int", "default": 128, "low": 64, "high": 256, "step": 64, "group": "architecture"},
                "DEVICE_BATCH_SIZE": {"type": "int", "default": 128, "low": 16, "high": 256, "step": 32, "group": "batching"},
                "WINDOW_PATTERN": {"type": "choice", "default": "SSSL", "values": ["SSSL", "SSLL", "L", "SL"], "group": "attention"},
            },
            "stress_axes": {
                # The exhaustive stage re-tests a win at a smaller and a larger
                # model, because "wins at depth 8 only" is a tuning artefact.
                "scale": {
                    "nominal": {},
                    "smaller": {"DEPTH": 6},
                    "larger": {"DEPTH": 10},
                }
            },
            "runner_config": {
                "repo_root": REPO_ROOT,
                "script": "train.py",
                "launcher": "uv run",
                "timeout_s": 900,
                "metric_key": "val_bpb",
            },
            "guardrails": {
                "peak_vram_mb": {"max": 70000, "note": "VRAM is a soft constraint"},
            },
        },
    )


def read_results_tsv(path: str) -> list:
    """Import an existing `results.tsv` from the manual autoresearch loop.

    Lets a subject start with the history a human (or an earlier agent run)
    already produced instead of from nothing.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                row["val_bpb"] = float(row.get("val_bpb", "0") or 0)
            except ValueError:
                row["val_bpb"] = 0.0
            rows.append(row)
    return rows


def parse_run_log(path: str) -> Dict[str, float]:
    """Read the summary block out of a `run.log` produced by the manual loop."""
    try:
        with open(path) as f:
            return parse_metrics(f.read())
    except OSError:
        return {}
