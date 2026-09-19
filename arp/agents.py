"""The agent layer: proposing, criticising, and writing up.

Each agent has two paths. The **LLM path** (Claude, via the Messages API) is used
when an API key is present and gives better hypotheses and sharper critiques. The
**heuristic path** is deterministic, uses the evolved capability memory, and
needs no network — it is not a stub, it is the fallback the platform runs on when
offline, and every test exercises it.

Agents never decide anything. They propose changes, raise doubts and write
English; the verdict comes from `arp.stats` and `arp.exhaustive` alone. That
separation is the whole point: an LLM that gets excited about a result cannot
promote it.
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from arp.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
from arp.evolution import CapabilityMemory, choose_operator, mutate, operator_catalog
from arp.models import Finding, Hypothesis, Prompt, Subject, Verdict
from arp.netutil import HttpError, post_json
from arp.store import Store


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal Anthropic Messages API client over urllib."""

    def __init__(self, api_key: str = "", model: str = "", base_url: str = ""):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or ANTHROPIC_MODEL
        self.base_url = (base_url or ANTHROPIC_BASE_URL).rstrip("/")
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self, system: str, user: str, max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        if not self.available:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            payload["tools"] = tools
        try:
            data = post_json(
                f"{self.base_url}/v1/messages",
                payload,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            )
        except HttpError as exc:
            self.last_error = str(exc)
            return None
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip() or None


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON array/object out of a model response."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Hypothesis proposer
# ---------------------------------------------------------------------------

@dataclass
class ProposalContext:
    subject: Subject
    best_params: Dict[str, Any] = field(default_factory=dict)
    recent_findings: List[Finding] = field(default_factory=list)
    recent_prompts: List[Prompt] = field(default_factory=list)
    boldness: float = 1.0
    epoch: int = 0


class HypothesisProposer:
    def __init__(self, store: Store, memory: CapabilityMemory, llm: Optional[LLMClient] = None,
                 rng: Optional[random.Random] = None):
        self.store = store
        self.memory = memory
        self.llm = llm or LLMClient()
        self.rng = rng or random.Random(0)

    def propose(self, ctx: ProposalContext, k: int = 3) -> List[Hypothesis]:
        hypotheses = self._propose_llm(ctx, k) if self.llm.available else []
        if not hypotheses:
            hypotheses = self._propose_heuristic(ctx, k)
        return hypotheses

    # -- heuristic ---------------------------------------------------------

    def _propose_heuristic(self, ctx: ProposalContext, k: int) -> List[Hypothesis]:
        subject = ctx.subject
        out: List[Hypothesis] = []
        # Don't re-propose a change already tried against this baseline: duplicate
        # arms split the evidence for one question across two posteriors, and
        # re-running a settled one buys nothing. Scoped to the epoch, because
        # once the baseline moves the same delta is a different experiment.
        seen: set = {
            json.dumps(h.params, sort_keys=True, default=str)
            for h in self.store.list_hypotheses(subject.id)
            if int((h.meta or {}).get("epoch", 0)) == ctx.epoch
        }
        recent_ops = [h.operator for h in self.store.list_hypotheses(subject.id)[-6:]]
        catalog = set(operator_catalog(subject))
        # Operators that had nothing to offer this round: `revert` and `simplify`
        # are no-ops when the baseline already is the default, and `combine` needs
        # two independent wins. Without retiring them for the round, a subject
        # with few knobs can starve — the loop then spends its whole budget
        # re-measuring the baseline because nothing was ever proposed.
        exhausted: Dict[str, int] = {}
        hints = self.direction_hints(subject)
        for _ in range(k * 8):
            if len(out) >= k:
                break
            dead = {op for op, fails in exhausted.items() if fails >= 3}
            exclude = set(dead)
            if len(catalog - dead) > 3:
                exclude |= set(recent_ops[-2:])
            operator = choose_operator(self.memory, subject, self.rng, exclude=sorted(exclude))
            base = dict(ctx.best_params or subject.baseline_params)
            if operator == "combine":
                hyp = self._combine(ctx)
                if hyp is None:
                    exhausted[operator] = exhausted.get(operator, 0) + 3
                    continue
                fingerprint = json.dumps(hyp.params, sort_keys=True, default=str)
                if fingerprint in seen:
                    exhausted[operator] = exhausted.get(operator, 0) + 1
                    continue
                seen.add(fingerprint)
                out.append(hyp)
                continue
            params, change = mutate(subject, operator, base, self.rng, ctx.boldness, hints)
            delta = {n: v for n, v in params.items() if base.get(n) != v}
            if not delta:
                exhausted[operator] = exhausted.get(operator, 0) + 1
                continue
            fingerprint = json.dumps(delta, sort_keys=True, default=str)
            if fingerprint in seen:
                exhausted[operator] = exhausted.get(operator, 0) + 1
                continue
            seen.add(fingerprint)
            prior = self.memory.prior_for(subject, operator)
            out.append(
                Hypothesis(
                    subject_id=subject.id,
                    title=change,
                    operator=operator,
                    # Store the *delta*, not the whole configuration. A hypothesis
                    # is "change this"; if it carried a full parameter set it would
                    # silently revert every win adopted after it was proposed.
                    params=delta,
                    origin="platform",
                    # `base` is what the parameter was when this was proposed;
                    # direction_hints needs it to tell "up" from "down" later,
                    # after the baseline has moved on.
                    meta={"epoch": ctx.epoch, "base": {n: base.get(n) for n in delta}},
                    rationale=(
                        f"Operator {operator} has a prior expected gain of {prior:+.4%} on this "
                        f"subject (evolved from {self._evidence_count(subject, operator)} prior outcomes). "
                        f"Change under test: {change}."
                    ),
                )
            )
        return out

    def direction_hints(self, subject: Subject) -> Dict[str, float]:
        """Which way each numeric knob has paid off so far.

        Positive means "turning this up has helped"; negative means the opposite.
        Built only from single-parameter hypotheses with a settled verdict, where
        the direction is unambiguous, and weighted by how much each one moved the
        metric. This is the platform noticing what a researcher would notice
        after two experiments, instead of flipping a coin on the third.
        """
        hints: Dict[str, float] = {}
        for finding in self.store.list_findings(subject.id):
            if finding.verdict not in (Verdict.SUPPORTED, Verdict.PROVEN, Verdict.REFUTED):
                continue
            hypothesis = self.store.get_hypothesis(finding.hypothesis_id)
            if hypothesis is None or len(hypothesis.params or {}) != 1:
                continue
            (name, value), = hypothesis.params.items()
            base = (hypothesis.meta or {}).get("base", {}).get(name)
            if not (_is_number(value) and _is_number(base)) or value == base:
                continue
            moved_up = 1.0 if value > base else -1.0
            # effect is signed so that positive always means "better"
            hints[name] = hints.get(name, 0.0) + moved_up * (finding.effect / 0.01)
        return {name: max(-2.0, min(2.0, value)) for name, value in hints.items()}

    def _evidence_count(self, subject: Subject, operator: str) -> int:
        cap = self.store.get_capability(operator, subject.id)
        return cap.n if cap else 0

    def _combine(self, ctx: ProposalContext) -> Optional[Hypothesis]:
        """Stack the parameter changes from two proven findings.

        Wins found separately often interact; this is the cheapest way to find
        out whether they add up or cancel.
        """
        baseline = ctx.best_params or ctx.subject.baseline_params
        candidates = []
        for finding in self.store.list_findings(ctx.subject.id, [Verdict.PROVEN, Verdict.SUPPORTED]):
            hyp = self.store.get_hypothesis(finding.hypothesis_id)
            if hyp is None or not hyp.params:
                continue
            # Skip anything already folded into the baseline by adoption — there
            # is nothing to learn from re-testing the setup against itself.
            if all(baseline.get(k) == v for k, v in hyp.params.items()):
                continue
            candidates.append(hyp)
        if len(candidates) < 2:
            return None

        self.rng.shuffle(candidates)
        pair = None
        for i, first in enumerate(candidates):
            for second in candidates[i + 1:]:
                # Only combine changes that touch different knobs; stacking two
                # settings of the same parameter just means the second one.
                if not set(first.params) & set(second.params):
                    pair = (first, second)
                    break
            if pair:
                break
        if pair is None:
            return None

        params: Dict[str, Any] = {}
        titles = []
        for hyp in pair:
            params.update(hyp.params)
            titles.append(hyp.title)
        return Hypothesis(
            subject_id=ctx.subject.id,
            title=f"combine: {titles[0]} + {titles[1]}",
            operator="combine",
            params=params,
            origin="evolution",
            meta={"epoch": ctx.epoch},
            rationale="Two independently proven changes, applied together, to test for interaction.",
        )

    # -- LLM ---------------------------------------------------------------

    def _propose_llm(self, ctx: ProposalContext, k: int) -> List[Hypothesis]:
        subject = ctx.subject
        space = json.dumps(subject.param_space, indent=2, default=str)[:4000]
        history = "\n".join(
            f"- {f.verdict.upper()} {f.effect:+.3%}: "
            f"{(self.store.get_hypothesis(f.hypothesis_id).title if self.store.get_hypothesis(f.hypothesis_id) else f.hypothesis_id)}"
            for f in ctx.recent_findings[:12]
        ) or "- (no findings yet)"
        prompts = "\n".join(f"- [{p.kind}] {p.text}" for p in ctx.recent_prompts[:8]) or "- (none)"
        skill = self.memory.skill_profile(subject)
        capabilities = "\n".join(
            f"- {row['operator']}: n={row['n']}, hit rate {row['success_rate']:.2f}, "
            f"mean gain {row['mean_reward']:+.4%}"
            for row in skill.operators[:10]
        ) or "- (no capability history yet)"

        system = (
            "You are the hypothesis proposer inside an automated research platform. "
            "You propose testable parameter changes; you never judge whether a result is real "
            "— a statistical engine does that. Favour changes that are cheap to test, "
            "mechanistically plausible, and different from what has already been tried."
        )
        user = f"""Subject: {subject.title}
Metric: {subject.metric} ({subject.direction})
Description: {subject.description}

Parameter space (JSON):
{space}

Current best parameters: {json.dumps(ctx.best_params, default=str)}

Operator families available: {', '.join(operator_catalog(subject))}

What the platform has learned about operators here:
{capabilities}

Recent verdicts:
{history}

Recent human guidance:
{prompts}

Propose exactly {k} new hypotheses as a JSON array. Each element:
{{"operator": "<one of the operator families>",
  "params": {{"<param>": <value>, ...}},   // only parameters from the space, within bounds
  "title": "<short description of the change>",
  "rationale": "<one or two sentences of mechanism, not hype>"}}
Return only the JSON array."""

        raw = self.llm.complete(system, user, max_tokens=2000)
        parsed = extract_json(raw or "")
        if not isinstance(parsed, list):
            return []

        out: List[Hypothesis] = []
        catalog = set(operator_catalog(subject))
        for item in parsed[:k]:
            if not isinstance(item, dict):
                continue
            params = item.get("params")
            if not isinstance(params, dict) or not params:
                continue
            clean = self._validate_params(subject, params)
            if not clean:
                continue
            operator = item.get("operator")
            if operator not in catalog:
                operator = _infer_operator(subject, clean)
            out.append(
                Hypothesis(
                    subject_id=subject.id,
                    title=str(item.get("title") or "llm proposal")[:200],
                    operator=operator,
                    params=clean,  # a delta against the current baseline, as above
                    rationale=str(item.get("rationale") or "")[:1000],
                    origin="platform",
                    meta={"proposer": "llm", "model": self.llm.model, "epoch": ctx.epoch},
                )
            )
        return out

    @staticmethod
    def _validate_params(subject: Subject, params: Dict[str, Any]) -> Dict[str, Any]:
        """Drop anything outside the declared space or its bounds.

        A model that invents a parameter name would otherwise produce a trial
        that silently measures the baseline.
        """
        clean: Dict[str, Any] = {}
        for name, value in params.items():
            spec = subject.param_space.get(name)
            if not spec:
                continue
            kind = spec.get("type", "float")
            try:
                if kind == "float":
                    value = float(value)
                elif kind == "int":
                    value = int(value)
                elif kind == "choice" and value not in spec.get("values", []):
                    continue
            except (TypeError, ValueError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                low, high = spec.get("low"), spec.get("high")
                if (low is not None and value < low) or (high is not None and value > high):
                    continue
            clean[name] = value
        return clean


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _infer_operator(subject: Subject, params: Dict[str, Any]) -> str:
    from arp.evolution import param_groups

    groups = param_groups(subject)
    for group, names in groups.items():
        if any(n in params for n in names):
            return f"tune:{group}"
    return "combine"


# ---------------------------------------------------------------------------
# Critic — the platform second-guessing itself
# ---------------------------------------------------------------------------

@dataclass
class Critique:
    concern: str
    axis: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"


class Critic:
    """Looks for reasons a finding might be wrong, and turns them into stress axes.

    The human can do this by hand (`arp challenge`); this is the automatic
    version, run before anything is allowed to be called proven.
    """

    def __init__(self, store: Store, llm: Optional[LLMClient] = None):
        self.store = store
        self.llm = llm or LLMClient()

    def critique(self, subject: Subject, finding: Finding, hypothesis: Hypothesis) -> List[Critique]:
        out = self._heuristic(subject, finding, hypothesis)
        llm_notes = self._llm(subject, finding, hypothesis)
        return out + llm_notes

    def _heuristic(self, subject: Subject, finding: Finding, hypothesis: Hypothesis) -> List[Critique]:
        notes: List[Critique] = []
        est = finding.evidence.get("estimate", {})
        n_treatment = int(est.get("n_treatment", finding.n_trials) or 0)
        sigma = float(est.get("sigma", 0.0) or 0.0)

        if n_treatment and n_treatment < 5:
            # Medium, not high: the exhaustive stage is about to add replicates
            # anyway, and a challenge here would only duplicate that work.
            notes.append(Critique(
                f"only {n_treatment} treatment replicates: the interval is carried by very few runs",
                severity="medium",
            ))
        if sigma and abs(finding.effect) < sigma:
            notes.append(Critique(
                f"effect {finding.effect:+.3%} is smaller than run-to-run noise ({sigma:.3%})",
                severity="high",
            ))
        elif sigma and abs(finding.effect) < 2 * sigma:
            notes.append(Critique(
                f"effect {finding.effect:+.3%} is of the same order as run-to-run noise ({sigma:.3%})",
                severity="medium",
            ))
        if not subject.stress_axes:
            notes.append(Critique(
                "subject declares no stress axes, so 'holds everywhere' means 'holds in one configuration'",
                severity="medium",
            ))
        # Parameters pushed against a declared bound are usually hitting a
        # constraint of the setup, not a real optimum.
        for name, value in (hypothesis.params or {}).items():
            spec = subject.param_space.get(name, {})
            for bound in ("low", "high"):
                edge = spec.get(bound)
                if edge is not None and isinstance(value, (int, float)) and abs(float(value) - float(edge)) < 1e-9:
                    notes.append(Critique(
                        f"{name} sits exactly on its {bound} bound: the win may be an artefact of the bound",
                        severity="medium",
                    ))
        return notes

    def _llm(self, subject: Subject, finding: Finding, hypothesis: Hypothesis) -> List[Critique]:
        if not self.llm.available:
            return []
        system = (
            "You are the adversarial reviewer in an automated research platform. "
            "Given a claimed experimental result, list the most plausible ways it could be "
            "an artefact rather than a real effect. Be concrete and testable. No praise."
        )
        user = f"""Subject: {subject.title} (metric {subject.metric}, {subject.direction})
Change: {hypothesis.title}
Parameters: {json.dumps(hypothesis.params, default=str)}
Claimed effect: {finding.effect:+.4%} (interval [{finding.ci_low:+.4%}, {finding.ci_high:+.4%}], p={finding.p_value:.4g})
Evidence: {json.dumps(finding.evidence, default=str)[:2000]}

Return a JSON array of at most 3 objects:
{{"concern": "<what could make this wrong>",
  "axis": {{"<stress axis name>": ["<label>", "<label>"]}},   // how to test it, or {{}}
  "severity": "low"|"medium"|"high"}}"""
        parsed = extract_json(self.llm.complete(system, user, max_tokens=1200) or "")
        if not isinstance(parsed, list):
            return []
        out = []
        for item in parsed[:3]:
            if isinstance(item, dict) and item.get("concern"):
                axis = item.get("axis") if isinstance(item.get("axis"), dict) else {}
                out.append(Critique(str(item["concern"])[:500], axis, str(item.get("severity", "medium"))))
        return out


# ---------------------------------------------------------------------------
# Prompt interpretation
# ---------------------------------------------------------------------------

CHALLENGE_WORDS = ("challenge", "don't believe", "doubt", "suspicious", "second guess",
                   "second-guess", "prove it", "not convinced", "verify", "re-test", "retest")
CONSTRAINT_WORDS = ("must", "never", "don't ", "do not ", "avoid", "keep", "budget", "limit", "cap ")
QUESTION_WORDS = ("?", "what ", "why ", "how ", "which ")


def classify_prompt(text: str) -> str:
    """Route a human prompt to an intent, without needing the LLM to be up."""
    lowered = (text or "").lower()
    if any(w in lowered for w in CHALLENGE_WORDS):
        return "challenge"
    if any(w in lowered for w in ("try ", "test ", "experiment", "what if", "increase", "decrease", "swap")):
        return "hypothesis"
    if any(w in lowered for w in CONSTRAINT_WORDS):
        return "constraint"
    if any(w in lowered for w in QUESTION_WORDS):
        return "question"
    return "steer"


def parse_param_hints(subject: Subject, text: str) -> Dict[str, Any]:
    """Pull explicit `NAME=value` or `NAME to value` instructions out of a prompt.

    Lets a human say "try DEPTH=12" and have it become a real hypothesis rather
    than a note nobody reads.
    """
    hints: Dict[str, Any] = {}
    for name, spec in subject.param_space.items():
        pattern = re.compile(
            rf"\b{re.escape(name)}\b\s*(?:=|to|:|->)\s*([A-Za-z0-9_.\-+]+)", re.IGNORECASE
        )
        match = pattern.search(text or "")
        if not match:
            continue
        raw = match.group(1)
        kind = spec.get("type", "float")
        try:
            if kind == "float":
                hints[name] = float(raw)
            elif kind == "int":
                hints[name] = int(float(raw))
            else:
                candidates = [v for v in spec.get("values", []) if str(v).lower() == raw.lower()]
                if candidates:
                    hints[name] = candidates[0]
        except ValueError:
            continue
    return hints
