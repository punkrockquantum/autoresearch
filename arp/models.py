"""Dataclasses for everything the platform stores or passes around.

Kept deliberately plain: dicts go to SQLite as JSON, and every object round-trips
through `from_row`/`to_row` so the store stays a thin layer.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _loads(blob: Optional[str]) -> Dict[str, Any]:
    if not blob:
        return {}
    try:
        value = json.loads(blob)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class Verdict:
    """Lifecycle of a thesis. Only PROVEN and REFUTED are terminal."""

    PROPOSED = "proposed"        # queued, no evidence yet
    TESTING = "testing"          # accumulating trials
    SUPPORTED = "supported"      # screening stage passed, exhaustive stage pending
    PROVEN = "proven"            # exhaustive grid all-pass, no open challenges
    REFUTED = "refuted"          # evidence points the wrong way, conclusively
    INCONCLUSIVE = "inconclusive"  # budget spent inside the region of practical equivalence
    CONTESTED = "contested"      # a human second-guessed it; re-testing under a stricter bar

    TERMINAL = (PROVEN, REFUTED, INCONCLUSIVE)
    OPEN = (PROPOSED, TESTING, SUPPORTED, CONTESTED)


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------

@dataclass
class Subject:
    """A research subject: what we are trying to learn, and how it is measured."""

    slug: str
    title: str
    metric: str = "val_bpb"
    direction: str = "minimize"           # "minimize" | "maximize"
    description: str = ""
    runner: str = "simulated"             # key into arp.runners.REGISTRY
    id: str = field(default_factory=lambda: new_id("subj"))
    status: str = "active"
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    # config keys the rest of the platform reads
    #   param_space: {name: {"type": "float"|"int"|"choice", ...}}
    #   operators:   [operator names the proposer may use]
    #   stress_axes: {axis: [values]}  -> the exhaustive confirmation grid
    #   runner_config: {...}           -> passed through to the runner
    #   baseline_params: {...}

    @property
    def param_space(self) -> Dict[str, Any]:
        return self.config.get("param_space", {})

    @property
    def stress_axes(self) -> Dict[str, List[Any]]:
        return self.config.get("stress_axes", {})

    @property
    def baseline_params(self) -> Dict[str, Any]:
        return self.config.get("baseline_params", {})

    def better(self, a: float, b: float) -> bool:
        """True if metric value `a` is better than `b` under this subject's direction."""
        return a < b if self.direction == "minimize" else a > b

    def gain(self, baseline: float, treatment: float) -> float:
        """Signed improvement of treatment over baseline; positive is better."""
        raw = baseline - treatment if self.direction == "minimize" else treatment - baseline
        return raw

    def relative_gain(self, baseline: float, treatment: float) -> float:
        denom = abs(baseline) if abs(baseline) > 1e-12 else 1.0
        return self.gain(baseline, treatment) / denom

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["config"] = _dumps(self.config)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Subject":
        data = dict(row)
        data["config"] = _loads(data.get("config"))
        return Subject(**data)


@dataclass
class Prompt:
    """A turn in the human/platform conversation about a subject.

    Prompts are first-class state, not chat scrollback: the suggestion engine
    mines them, and interventions are stored as prompts so the audit trail of
    "who asked for what, when" survives restarts.
    """

    subject_id: str
    text: str
    role: str = "human"          # "human" | "platform"
    kind: str = "steer"          # steer | hypothesis | constraint | question | challenge | note
    id: str = field(default_factory=lambda: new_id("pr"))
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["meta"] = _dumps(self.meta)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Prompt":
        data = dict(row)
        data["meta"] = _loads(data.get("meta"))
        return Prompt(**data)


@dataclass
class Hypothesis:
    """One testable change: "do X and the metric moves this way"."""

    subject_id: str
    title: str
    operator: str                       # family of change, e.g. "lr_scale"
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    origin: str = "platform"            # platform | human | web | evolution
    id: str = field(default_factory=lambda: new_id("hyp"))
    parent_id: Optional[str] = None
    status: str = Verdict.PROPOSED
    created_at: str = field(default_factory=now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["params"] = _dumps(self.params)
        row["meta"] = _dumps(self.meta)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Hypothesis":
        data = dict(row)
        data["params"] = _loads(data.get("params"))
        data["meta"] = _loads(data.get("meta"))
        return Hypothesis(**data)


@dataclass
class TrialSpec:
    """What a runner is asked to do. Fully determined, including the seed."""

    subject: Subject
    params: Dict[str, Any]
    seed: int
    arm: str = "treatment"              # "baseline" | "treatment"
    cell: Dict[str, Any] = field(default_factory=dict)  # exhaustive-grid coordinates
    timeout_s: float = 1800.0
    hypothesis_id: Optional[str] = None


@dataclass
class TrialResult:
    """What a runner gives back."""

    ok: bool
    metric: Optional[float] = None
    aux: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0
    log_path: str = ""


@dataclass
class Trial:
    """A stored measurement."""

    subject_id: str
    hypothesis_id: Optional[str]
    arm: str
    seed: int
    metric_value: Optional[float]
    ok: bool = True
    stage: str = "screen"               # screen | confirm | challenge
    params: Dict[str, Any] = field(default_factory=dict)
    cell: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0
    id: str = field(default_factory=lambda: new_id("tr"))
    created_at: str = field(default_factory=now_iso)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["params"] = _dumps(self.params)
        row["cell"] = _dumps(self.cell)
        row["aux"] = _dumps(self.aux)
        row["ok"] = 1 if self.ok else 0
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Trial":
        data = dict(row)
        data["params"] = _loads(data.get("params"))
        data["cell"] = _loads(data.get("cell"))
        data["aux"] = _loads(data.get("aux"))
        data["ok"] = bool(data.get("ok"))
        return Trial(**data)


@dataclass
class Finding:
    """The platform's current belief about one hypothesis, with its evidence."""

    subject_id: str
    hypothesis_id: str
    verdict: str = Verdict.TESTING
    effect: float = 0.0                 # relative gain, positive = better
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float = 1.0
    n_trials: int = 0
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    impact: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("fnd"))
    proven_at: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["evidence"] = _dumps(self.evidence)
        row["impact"] = _dumps(self.impact)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Finding":
        data = dict(row)
        data["evidence"] = _loads(data.get("evidence"))
        data["impact"] = _loads(data.get("impact"))
        return Finding(**data)


@dataclass
class Challenge:
    """A human saying "I don't believe that result".

    A challenge is not a comment: it tightens the decision policy, forces the
    finding back into testing, and adds its own stress axis to the exhaustive
    grid. It closes only when the re-test clears the stricter bar.
    """

    subject_id: str
    finding_id: str
    reason: str
    axis: Dict[str, List[Any]] = field(default_factory=dict)
    status: str = "open"                # open | resolved
    resolution: str = ""
    id: str = field(default_factory=lambda: new_id("chl"))
    created_at: str = field(default_factory=now_iso)
    resolved_at: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["axis"] = _dumps(self.axis)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Challenge":
        data = dict(row)
        data["axis"] = _loads(data.get("axis"))
        return Challenge(**data)


@dataclass
class Capability:
    """What the platform has learned about an operator, per subject.

    Beta(alpha, beta) over "this operator produces a real win", plus running
    reward moments. This is the memory that makes the platform better at a
    subject the more it works on it — and, via subject similarity, better at
    new subjects that resemble old ones.
    """

    operator: str
    subject_id: str = ""                # "" means the global/cross-subject prior
    alpha: float = 1.0
    beta: float = 1.0
    reward_sum: float = 0.0
    reward_sq: float = 0.0
    n: int = 0
    id: str = field(default_factory=lambda: new_id("cap"))
    updated_at: str = field(default_factory=now_iso)

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Capability":
        return Capability(**dict(row))


@dataclass
class Lead:
    """A web result an agent judged relevant to a subject or finding."""

    subject_id: str
    title: str
    url: str
    snippet: str = ""
    source: str = "web"
    query: str = ""
    finding_id: Optional[str] = None
    scores: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("lead"))
    created_at: str = field(default_factory=now_iso)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["scores"] = _dumps(self.scores)
        return row

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Lead":
        data = dict(row)
        data["scores"] = _loads(data.get("scores"))
        return Lead(**data)


@dataclass
class Run:
    """One invocation of the orchestrator loop."""

    subject_id: str
    salt: str
    steps_requested: int
    steps_done: int = 0
    status: str = "running"
    notes: str = ""
    id: str = field(default_factory=lambda: new_id("run"))
    started_at: str = field(default_factory=now_iso)
    ended_at: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_row(row: Dict[str, Any]) -> "Run":
        return Run(**dict(row))


def stopwatch():
    """Tiny helper: returns a callable giving elapsed seconds since creation."""
    t0 = time.time()
    return lambda: time.time() - t0
