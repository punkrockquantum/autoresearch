"""Turning a sentence into a research subject.

`arp suggest` will tell you "you keep mentioning quantisation and nothing is
researching it" — and then, before this module existed, leave you to hand-write a
parameter space in JSON. This closes that loop: describe the subject in your own
words and the platform drafts it, using what it already knows.

Three sources, in order of preference, and the drafting says which one it used:

1. **An explicit template** (`--from <slug>`): clone that subject's parameter
   space and stress axes. The cheapest way to start a variation of work you are
   already doing.
2. **The most similar existing subject**, found by vocabulary overlap with the
   description *and* with your prior prompts about it. Research questions come in
   families; the second one about serving cost should not start from nothing.
3. **An LLM draft**, when a key is configured, validated field by field against
   what the platform can actually run.

If none of those produce a parameter space, the subject is still created — with
an empty space and an explicit note saying what to add. A subject with no knobs
is a question nobody has made testable yet, which is a fair thing to have
written down.

Prior prompts that mention the same things are copied into the new subject's log
as context, so the proposer starts with what you have already said about it
rather than with a blank page.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arp.agents import LLMClient, extract_json
from arp.evolution import CapabilityMemory, similarity, subject_tokens, tokenize
from arp.models import Prompt, Subject
from arp.store import Store

# Words that say which way the metric should go. Checked longest-first so
# "minimise" wins over a stray "increase" later in the sentence.
MINIMISE_WORDS = ("minimise", "minimize", "reduce", "lower", "cut", "decrease", "shrink", "cheaper")
MAXIMISE_WORDS = ("maximise", "maximize", "increase", "improve", "raise", "grow", "boost", "faster")

# Metric names worth recognising without an LLM, longest first so
# "cost per token" beats "cost".
KNOWN_METRICS = (
    "val_bpb", "validation loss", "val_loss", "perplexity", "accuracy", "f1",
    "cost per million tokens", "cost per token", "tokens per second", "throughput",
    "latency", "win rate", "conversion rate", "revenue", "cost", "loss", "error rate",
)


@dataclass
class SubjectDraft:
    subject: Subject
    source: str                       # template | similar | llm | empty
    notes: List[str] = field(default_factory=list)
    context_prompts: int = 0

    @property
    def ready(self) -> bool:
        """Can the loop actually run this subject as drafted?"""
        return bool(self.subject.param_space)


def slugify(text: str, taken: Optional[set] = None) -> str:
    words = [w for w in tokenize(text)][:4]
    base = "-".join(words) or "subject"
    base = re.sub(r"[^a-z0-9-]", "", base)[:48].strip("-") or "subject"
    taken = taken or set()
    if base not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base}-{n}"
        if candidate not in taken:
            return candidate
    return f"{base}-{len(taken)}"


def infer_direction(text: str) -> str:
    lowered = (text or "").lower()
    first_min = min((lowered.find(w) for w in MINIMISE_WORDS if w in lowered), default=-1)
    first_max = min((lowered.find(w) for w in MAXIMISE_WORDS if w in lowered), default=-1)
    if first_min == -1 and first_max == -1:
        return "minimize"
    if first_max == -1:
        return "minimize"
    if first_min == -1:
        return "maximize"
    return "minimize" if first_min <= first_max else "maximize"


def infer_metric(text: str) -> str:
    """Pick a metric name out of the description, or fall back to a placeholder."""
    lowered = (text or "").lower()
    for name in KNOWN_METRICS:
        if name in lowered:
            return name.replace(" ", "_")
    # "minimise <word>" / "maximise the <word>"
    match = re.search(
        r"\b(?:" + "|".join(MINIMISE_WORDS + MAXIMISE_WORDS) + r")\s+(?:the\s+)?([a-z][a-z0-9_ ]{2,30})",
        lowered,
    )
    if match:
        words = [w for w in tokenize(match.group(1))][:3]
        if words:
            return "_".join(words)
    return "score"


def related_prompts(store: Store, text: str, limit: int = 12) -> List[Prompt]:
    """Earlier things the human said that are about this, wherever they said them."""
    terms = set(tokenize(text))
    if not terms:
        return []
    scored: List[Tuple[float, Prompt]] = []
    for prompt in store.list_prompts(limit=400):
        if prompt.role != "human":
            continue
        prompt_terms = set(tokenize(prompt.text))
        if not prompt_terms:
            continue
        overlap = len(terms & prompt_terms) / len(terms)
        if overlap > 0:
            scored.append((overlap, prompt))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [prompt for _, prompt in scored[:limit]]


def nearest_subject(store: Store, probe: Subject) -> Tuple[Optional[Subject], float]:
    best, best_score = None, 0.0
    for other in store.list_subjects():
        if other.id == probe.id:
            continue
        score = similarity(probe, other)
        if score > best_score:
            best, best_score = other, score
    return best, best_score


def draft_subject(
    store: Store,
    text: str,
    slug: Optional[str] = None,
    metric: Optional[str] = None,
    direction: Optional[str] = None,
    runner: Optional[str] = None,
    template_slug: Optional[str] = None,
    llm: Optional[LLMClient] = None,
    min_similarity: float = 0.3,
) -> SubjectDraft:
    """Draft (but do not save) a subject from a description."""
    taken = {s.slug for s in store.list_subjects()}
    context = related_prompts(store, text)
    # The description plus what you have already said about it: a prompt history
    # mentioning "serving cost" should pull this subject towards the serving-cost
    # family even when the one-line description doesn't say the words.
    enriched = " ".join([text] + [p.text for p in context])

    subject = Subject(
        slug=slug or slugify(text, taken),
        title=text.strip()[:200] or "Untitled subject",
        metric=metric or infer_metric(text),
        direction=direction or infer_direction(text),
        runner=runner or "command",
        description=text.strip(),
    )
    notes: List[str] = []

    template: Optional[Subject] = None
    source = "empty"
    if template_slug:
        template = store.get_subject(template_slug)
        if template is None:
            raise KeyError(f"no such subject to copy from: {template_slug}")
        source = "template"
    else:
        # Match on the description alone *and* on the description plus prior
        # chat, taking whichever scores higher. Prior chat usually helps, but it
        # can also drown out a short, precise description with older vocabulary,
        # and context should never make the platform recognise less.
        candidate, score = None, 0.0
        for probe_text in (text, enriched):
            probe = Subject(slug=subject.slug, title=probe_text, metric=subject.metric,
                            direction=subject.direction, description=probe_text)
            found, found_score = nearest_subject(store, probe)
            if found is not None and found_score > score:
                candidate, score = found, found_score
        if candidate is not None and score >= min_similarity and candidate.param_space:
            template, source = candidate, "similar"
            notes.append(
                f"parameter space copied from '{candidate.slug}' ({score:.0%} vocabulary overlap) "
                "— edit it if this subject has different knobs"
            )

    if template is not None:
        subject.config["param_space"] = json.loads(json.dumps(template.param_space))
        subject.config["baseline_params"] = {
            name: spec["default"] for name, spec in subject.param_space.items() if "default" in spec
        }
        if template.stress_axes:
            subject.config["stress_axes"] = json.loads(json.dumps(template.stress_axes))
        if runner is None:
            # Copy the runner *and everything it needs to run*. Taking the
            # parameter space but leaving the runner behind produces a subject
            # that looks ready and fails on every trial.
            subject.runner = template.runner
            for key in ("runner_config", "simulation"):
                if template.config.get(key):
                    subject.config[key] = json.loads(json.dumps(template.config[key]))
            notes.append(
                f"runner '{template.runner}' copied from '{template.slug}' — check that it "
                "measures this subject and not the one it came from"
            )
        if metric is None and runner is None and subject.metric != template.metric:
            # A copied runner emits the template's metric name. Keeping a metric
            # guessed from the description would leave the parser looking for a
            # key that never appears, and every trial would fail for no visible
            # reason.
            notes.append(
                f"metric set to '{template.metric}' to match the runner copied from "
                f"'{template.slug}' (guessed '{subject.metric}' from your description)"
            )
            subject.metric = template.metric
            if direction is None:
                subject.direction = template.direction
        if source == "template":
            notes.append(f"parameter space copied from '{template.slug}' as requested")

    llm = llm or LLMClient()
    if not subject.param_space and llm.available:
        drafted = _llm_param_space(llm, subject, enriched)
        if drafted:
            subject.config["param_space"] = drafted
            subject.config["baseline_params"] = {
                name: spec["default"] for name, spec in drafted.items() if "default" in spec
            }
            source = "llm"
            notes.append(f"parameter space drafted by {llm.model} — review the bounds before running")

    if not subject.param_space:
        notes.append(
            "no parameter space yet: nothing similar to copy and no model configured. "
            "Add one with `arp subject update <slug> --config space.json` before running."
        )

    if not subject.stress_axes:
        notes.append(
            "no stress axes declared, so 'proven' will mean 'held up in one configuration'"
        )

    return SubjectDraft(subject=subject, source=source, notes=notes, context_prompts=len(context))


def create_subject(store: Store, draft: SubjectDraft, carry_context: bool = True) -> SubjectDraft:
    """Persist a draft: save it, warm-start its memory, carry the conversation over."""
    store.add_subject(draft.subject)
    seeded = CapabilityMemory(store).seed_from_similar(draft.subject)
    if seeded:
        draft.notes.append(f"warm-started {seeded} operator prior(s) from similar subjects")

    if carry_context:
        for prompt in related_prompts(store, draft.subject.description or draft.subject.title):
            if prompt.subject_id == draft.subject.id:
                continue
            store.add_prompt(Prompt(
                subject_id=draft.subject.id,
                text=prompt.text,
                role="human",
                kind="context",
                meta={"carried_from_prompt": prompt.id, "carried_from_subject": prompt.subject_id},
            ))
    store.add_prompt(Prompt(
        subject_id=draft.subject.id,
        text=(
            f"Subject created from a prompt: \"{draft.subject.title}\". "
            f"Parameter space source: {draft.source}."
        ),
        role="platform", kind="note",
    ))
    return draft


def starter_config(subject: Subject) -> Dict[str, Any]:
    """A template the human can fill in when nothing could be copied or drafted."""
    return {
        "param_space": {
            "example_knob": {
                "type": "float", "default": 1.0, "low": 0.1, "high": 10.0,
                "log": True, "group": "example", "step": 0.25,
            }
        },
        "baseline_params": {"example_knob": 1.0},
        "stress_axes": {"scale": {"nominal": {}, "larger": {"example_knob": 2.0}}},
        "runner_config": {
            "command": f"echo {subject.metric}: 1.0",
            "metric_key": subject.metric,
            "timeout_s": 600,
        },
        "value_model": {"reference_effect": 0.01, "effort_days": 5},
    }


# ---------------------------------------------------------------------------
# LLM drafting
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"float", "int", "choice"}


def _llm_param_space(llm: LLMClient, subject: Subject, context: str) -> Dict[str, Any]:
    system = (
        "You design experiment parameter spaces for an automated research platform. "
        "Only propose knobs that could plausibly be set programmatically before a run. "
        "Give realistic bounds. Return JSON only."
    )
    user = f"""Subject: {subject.title}
Metric: {subject.metric} ({subject.direction})
Context from earlier conversation:
{context[:2000]}

Return a JSON object mapping parameter name to spec. Each spec is one of:
{{"type": "float", "default": <n>, "low": <n>, "high": <n>, "log": true|false, "group": "<family>", "step": <n>}}
{{"type": "int", "default": <n>, "low": <n>, "high": <n>, "step": <n>, "group": "<family>"}}
{{"type": "choice", "default": "<v>", "values": ["<v>", ...], "group": "<family>"}}
Use between 2 and 8 parameters. `group` names the family of knob (e.g. "lr",
"architecture", "batching") and is what the platform reasons about."""

    parsed = extract_json(llm.complete(system, user, max_tokens=1500) or "")
    if not isinstance(parsed, dict):
        return {}
    return validate_param_space(parsed)


def validate_param_space(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only specs the runners and mutation operators can actually use.

    A drafted space that looks plausible but is malformed would produce trials
    that measure nothing, so anything that does not typecheck is dropped rather
    than repaired.
    """
    clean: Dict[str, Any] = {}
    for name, spec in (raw or {}).items():
        if not isinstance(name, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            continue
        if not isinstance(spec, dict) or spec.get("type") not in ALLOWED_TYPES:
            continue
        kind = spec["type"]
        out: Dict[str, Any] = {"type": kind, "group": str(spec.get("group", "misc"))}
        try:
            if kind == "choice":
                values = [v for v in spec.get("values", []) if isinstance(v, (str, int, float, bool))]
                if len(values) < 2:
                    continue
                default = spec.get("default", values[0])
                out["values"] = values
                out["default"] = default if default in values else values[0]
            else:
                cast = float if kind == "float" else int
                out["default"] = cast(spec["default"])
                if "low" in spec:
                    out["low"] = cast(spec["low"])
                if "high" in spec:
                    out["high"] = cast(spec["high"])
                if out.get("low") is not None and out.get("high") is not None:
                    if out["low"] > out["high"]:
                        continue
                    if not (out["low"] <= out["default"] <= out["high"]):
                        continue
                if "step" in spec:
                    out["step"] = float(spec["step"]) if kind == "float" else int(spec["step"])
                if kind == "float":
                    out["log"] = bool(spec.get("log", True))
                    if out["log"] and out["default"] <= 0:
                        out["log"] = False
        except (KeyError, TypeError, ValueError):
            continue
        clean[name] = out
    return clean
