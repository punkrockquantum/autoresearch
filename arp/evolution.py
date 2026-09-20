"""Capability memory: how the platform gets better at the subjects it is given.

Two mechanisms, both boring on purpose:

* **Per-operator Beta posteriors, per subject.** Every time an operator family
  ("tune the learning-rate group", "change depth", "combine two winners")
  produces a verdict, its posterior moves. The allocator reads those posteriors
  as priors, so the platform stops wasting compute on knobs that never pay on
  *this* subject.
* **Transfer by subject similarity.** A new subject starts from a blend of the
  capability tables of subjects that look like it (token overlap of title,
  description and metric), rather than from a flat prior. That is what makes the
  tenth subject cheaper to research than the first.

The operator catalogue itself is derived from the subject's declared parameter
space, so a subject about training hyperparameters and a subject about, say,
pricing copy get the same machinery with different knobs.
"""

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from arp.models import Capability, Subject, Verdict
from arp.store import Store

STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "or", "on", "in", "with", "by",
    "is", "are", "be", "we", "our", "how", "what", "this", "that", "it", "its",
    "can", "do", "does", "using", "use", "when", "at", "as", "from", "into",
    # Filler that would otherwise read as a research theme in its own right:
    # "you have mentioned 'should' in 12 prompts" is not a suggestion.
    "should", "would", "could", "will", "shall", "must", "need", "want", "let",
    "also", "make", "get", "got", "try", "look", "see", "think", "maybe", "about",
    "more", "less", "some", "any", "all", "very", "just", "like", "one", "two",
    "new", "now", "then", "than", "them", "they", "there", "here", "have", "has",
    "had", "been", "being", "per", "out", "up", "down", "off", "over", "but",
    "not", "you", "your", "was", "were", "why", "who", "each", "may", "might",
}

# Operators that exist for every subject, whatever its parameter space.
UNIVERSAL_OPERATORS = ("combine", "revert", "simplify")


# ---------------------------------------------------------------------------
# Parameter space / operator catalogue
# ---------------------------------------------------------------------------

def param_groups(subject: Subject) -> Dict[str, List[str]]:
    """Map group name -> parameter names, from the subject's param_space."""
    groups: Dict[str, List[str]] = {}
    for name, spec in subject.param_space.items():
        group = str(spec.get("group", "misc"))
        groups.setdefault(group, []).append(name)
    return groups


def operator_catalog(subject: Subject) -> List[str]:
    """All operator families available on this subject."""
    declared = subject.config.get("operators")
    if declared:
        return list(declared)
    ops = [f"tune:{g}" for g in sorted(param_groups(subject))]
    ops.extend(UNIVERSAL_OPERATORS)
    return ops


def _clamp(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


def mutate(
    subject: Subject,
    operator: str,
    base_params: Dict[str, Any],
    rng: random.Random,
    boldness: float = 1.0,
    direction_hints: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Apply one operator to the base parameters.

    Returns (new_params, human-readable change description). `boldness` scales
    how far the mutation moves: the orchestrator raises it when the subject has
    gone a long stretch without a win.

    `direction_hints` biases *which way* a numeric knob moves, from what earlier
    trials showed about that same knob. Without it the direction is a coin flip,
    and a run can spend its whole budget turning the learning rate down when
    every result so far said to turn it up.
    """
    params = dict(base_params)
    space = subject.param_space

    if operator.startswith("tune:"):
        group = operator.split(":", 1)[1]
        names = param_groups(subject).get(group, [])
        if not names:
            return params, "no parameters in group"
        name = rng.choice(names)
        spec = space.get(name, {})
        kind = spec.get("type", "float")
        current = params.get(name, spec.get("default"))

        sign = _pick_direction(rng, (direction_hints or {}).get(name, 0.0))

        if kind == "float":
            step = float(spec.get("step", 0.25)) * boldness
            factor = math.exp(sign * step)
            if current is None:
                current = float(spec.get("default", 1.0))
            new = _clamp(float(current) * factor, spec.get("low"), spec.get("high"))
            params[name] = round(new, 8)
            return params, f"{name}: {current:g} -> {params[name]:g} (x{factor:.3f})"

        if kind == "int":
            step = max(1, int(round(float(spec.get("step", 1)) * boldness)))
            if current is None:
                current = int(spec.get("default", 1))
            new = int(_clamp(int(current) + int(sign) * step, spec.get("low"), spec.get("high")))
            params[name] = new
            return params, f"{name}: {current} -> {new}"

        values = list(spec.get("values", []))
        if not values:
            return params, "choice parameter with no values"
        alternatives = [v for v in values if v != current] or values
        new = rng.choice(alternatives)
        params[name] = new
        return params, f"{name}: {current!r} -> {new!r}"

    if operator == "combine":
        # Handled by the proposer, which knows which findings are proven; on its
        # own this is a no-op that keeps the catalogue uniform.
        return params, "combination of previously proven changes"

    if operator == "revert":
        for name, spec in space.items():
            if "default" in spec:
                params[name] = spec["default"]
        return params, "revert every parameter to its declared default"

    if operator == "simplify":
        # Drop the parameter furthest from its default: equal results with fewer
        # deviations is a win under the simplicity criterion.
        deviations = [
            (name, abs(_numeric(params.get(name)) - _numeric(spec.get("default"))))
            for name, spec in space.items()
            if "default" in spec and _is_numeric(params.get(name)) and _is_numeric(spec.get("default"))
        ]
        if deviations:
            name = max(deviations, key=lambda kv: kv[1])[0]
            params[name] = space[name]["default"]
            return params, f"simplify: reset {name} to its default"
        return params, "nothing to simplify"

    return params, f"unknown operator {operator}"


def _pick_direction(rng: random.Random, hint: float) -> float:
    """Up or down, biased by evidence but never fully committed.

    The floor of 10% either way matters: a knob that helped for three trials can
    still be past its optimum, and a proposer that stops checking the other
    direction will happily walk off a cliff.
    """
    p_up = min(0.9, max(0.1, 0.5 + 0.4 * math.tanh(hint)))
    return 1.0 if rng.random() < p_up else -1.0


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric(value: Any) -> float:
    return float(value) if _is_numeric(value) else 0.0


# ---------------------------------------------------------------------------
# Subject similarity
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if t not in STOPWORDS and len(t) > 2]


def subject_tokens(subject: Subject) -> set:
    parts = [subject.title, subject.description, subject.metric, subject.slug]
    parts.extend(subject.param_space.keys())
    return set(tokenize(" ".join(str(p) for p in parts)))


def similarity(a: Subject, b: Subject) -> float:
    """Vocabulary overlap between two subjects. Cheap, explainable, good enough.

    Half Jaccard, half overlap coefficient. Jaccard alone punishes a subject for
    having a long description — a one-line question about serving cost would
    score as unrelated to a well-documented subject on exactly that, purely
    because the documented one has more words. The overlap coefficient asks the
    question that actually matters ("is this about that?"); Jaccard keeps it from
    calling every short subject a match for everything.
    """
    ta, tb = subject_tokens(a), subject_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    jaccard = inter / union if union else 0.0
    containment = inter / min(len(ta), len(tb))
    base = 0.5 * jaccard + 0.5 * containment
    # Same metric and direction means the transfer is much more likely to hold.
    if a.metric == b.metric and a.direction == b.direction:
        base = 0.5 * base + 0.5
    return base


# ---------------------------------------------------------------------------
# Capability memory
# ---------------------------------------------------------------------------

@dataclass
class SkillProfile:
    subject_id: str
    level: float
    trials: int
    proven: int
    operators: List[Dict[str, Any]]

    def top_operators(self, k: int = 3) -> List[str]:
        return [o["operator"] for o in self.operators[:k]]


class CapabilityMemory:
    """Reads and writes the `capabilities` table; the platform's evolving ability."""

    def __init__(self, store: Store):
        self.store = store

    # -- reading -----------------------------------------------------------

    def prior_for(self, subject: Subject, operator: str) -> float:
        """Expected per-trial gain for this operator on this subject.

        Blend of: what this operator did here, what it did on similar subjects,
        and the global average. Falls back to a small positive number so an
        untried operator is still worth one shot.
        """
        own = self.store.get_capability(operator, subject.id)
        transfer = self._transfer_estimate(subject, operator)
        glob = self.store.get_capability(operator, "")

        parts: List[Tuple[float, float]] = []  # (weight, value)
        if own and own.n:
            parts.append((float(own.n), own.mean_reward))
        if transfer is not None:
            parts.append((2.0, transfer))
        if glob and glob.n:
            parts.append((1.0, glob.mean_reward))
        if not parts:
            return 0.0
        wsum = sum(w for w, _ in parts)
        return sum(w * v for w, v in parts) / wsum

    def success_rate(self, subject: Subject, operator: str) -> float:
        cap = self.store.get_capability(operator, subject.id)
        if cap:
            return cap.success_rate
        glob = self.store.get_capability(operator, "")
        return glob.success_rate if glob else 0.5

    def _transfer_estimate(self, subject: Subject, operator: str) -> Optional[float]:
        """Similarity-weighted mean reward for this operator on other subjects."""
        others = [s for s in self.store.list_subjects() if s.id != subject.id]
        num = den = 0.0
        for other in others:
            cap = self.store.get_capability(operator, other.id)
            if not cap or not cap.n:
                continue
            w = similarity(subject, other) * math.log1p(cap.n)
            if w <= 0:
                continue
            num += w * cap.mean_reward
            den += w
        return num / den if den else None

    def priors(self, subject: Subject, operators: Iterable[str]) -> Dict[str, float]:
        return {op: self.prior_for(subject, op) for op in operators}

    # -- writing -----------------------------------------------------------

    def update(self, subject: Subject, operator: str, reward: float, success: bool) -> None:
        """Record one outcome, for this subject and for the global prior."""
        for subject_id in (subject.id, ""):
            cap = self.store.get_capability(operator, subject_id) or Capability(
                operator=operator, subject_id=subject_id
            )
            cap.alpha += 1.0 if success else 0.0
            cap.beta += 0.0 if success else 1.0
            cap.reward_sum += reward
            cap.reward_sq += reward * reward
            cap.n += 1
            self.store.save_capability(cap)

    def record_verdict(self, subject: Subject, operator: str, verdict: str, effect: float) -> None:
        success = verdict in (Verdict.SUPPORTED, Verdict.PROVEN)
        self.update(subject, operator, reward=effect, success=success)

    def seed_from_similar(self, subject: Subject, min_similarity: float = 0.25) -> int:
        """Warm-start a new subject's capability table from subjects like it.

        Returns how many operators were seeded. This is the concrete sense in
        which the platform "already knows something" about a brand-new subject.
        """
        others = [s for s in self.store.list_subjects() if s.id != subject.id]
        scored = [(similarity(subject, o), o) for o in others]
        scored = [(s, o) for s, o in scored if s >= min_similarity]
        if not scored:
            return 0
        seeded = 0
        for operator in operator_catalog(subject):
            num = den = 0.0
            alpha = beta = 0.0
            for sim, other in scored:
                cap = self.store.get_capability(operator, other.id)
                if not cap or not cap.n:
                    continue
                num += sim * cap.reward_sum
                den += sim * cap.n
                alpha += sim * (cap.alpha - 1.0)
                beta += sim * (cap.beta - 1.0)
            if den <= 0:
                continue
            # Transferred evidence counts for less than first-hand evidence.
            discount = 0.5
            self.store.save_capability(
                Capability(
                    operator=operator,
                    subject_id=subject.id,
                    alpha=1.0 + discount * max(alpha, 0.0),
                    beta=1.0 + discount * max(beta, 0.0),
                    reward_sum=discount * num,
                    reward_sq=0.0,
                    n=int(max(1, round(discount * den))),
                )
            )
            seeded += 1
        return seeded

    # -- reporting ---------------------------------------------------------

    def skill_profile(self, subject: Subject) -> SkillProfile:
        caps = self.store.list_capabilities(subject.id)
        rows = sorted(
            (
                {
                    "operator": c.operator,
                    "n": c.n,
                    "success_rate": round(c.success_rate, 3),
                    "mean_reward": round(c.mean_reward, 6),
                }
                for c in caps
            ),
            key=lambda r: (r["mean_reward"], r["success_rate"]),
            reverse=True,
        )
        trials = self.store.count_trials(subject.id)
        proven = len(self.store.list_findings(subject.id, [Verdict.PROVEN]))
        # A blunt but honest competence number: evidence gathered x hit rate.
        hit_rate = (sum(r["success_rate"] for r in rows) / len(rows)) if rows else 0.0
        level = math.log1p(trials) * (0.5 + hit_rate) + proven
        return SkillProfile(
            subject_id=subject.id, level=round(level, 3), trials=trials, proven=proven, operators=rows
        )


def choose_operator(
    memory: CapabilityMemory,
    subject: Subject,
    rng: random.Random,
    exclude: Sequence[str] = (),
) -> str:
    """Thompson-sample an operator family from the evolved capability table."""
    catalog = [op for op in operator_catalog(subject) if op not in exclude]
    if not catalog:
        catalog = list(operator_catalog(subject))
    best, best_draw = catalog[0], -float("inf")
    for op in catalog:
        cap = memory.store.get_capability(op, subject.id)
        alpha = cap.alpha if cap else 1.0
        beta = cap.beta if cap else 1.0
        draw = rng.betavariate(max(alpha, 1e-6), max(beta, 1e-6))
        # Tie-break towards operators with a better observed reward, not just hit rate.
        draw += 0.1 * math.tanh(memory.prior_for(subject, op) * 50)
        if draw > best_draw:
            best, best_draw = op, draw
    return best
