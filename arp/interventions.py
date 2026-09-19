"""Human intervention: prompts that actually change what the engine does.

Three things a person can do between (or during) runs:

* **Steer** — "focus on the schedule, the architecture is a dead end". Stored,
  and read by the proposer on the next planning step.
* **Suggest a hypothesis** — "try DEPTH=12". Parsed into a real hypothesis with
  its own arm in the bandit, not a note in a log.
* **Challenge a result** — "I don't believe that win". This is the important
  one. A challenge tightens the decision policy, drags the finding back out of
  SUPPORTED/PROVEN into CONTESTED, and bolts a fresh axis onto the exhaustive
  grid so the entire stress grid is re-run from scratch. The finding cannot
  return to PROVEN until it clears the harder bar. Second-guessing is therefore
  cheap for the human and expensive for the claim, which is the right way round.
"""

from typing import Any, Dict, List, Optional

from arp.agents import Critic, classify_prompt, parse_param_hints
from arp.models import (
    Challenge,
    Finding,
    Hypothesis,
    Prompt,
    Subject,
    Verdict,
    now_iso,
)
from arp.store import Store


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def record_prompt(
    store: Store, subject: Subject, text: str, kind: Optional[str] = None,
    role: str = "human", meta: Optional[Dict[str, Any]] = None,
) -> Prompt:
    prompt = Prompt(
        subject_id=subject.id, text=text, role=role,
        kind=kind or classify_prompt(text), meta=meta or {},
    )
    return store.add_prompt(prompt)


def apply_prompt(store: Store, subject: Subject, text: str) -> Dict[str, Any]:
    """Store a human prompt and carry out whatever it asks for.

    Returns a small dict describing what happened, which the CLI prints — a
    person should never have to guess whether their prompt did anything.
    """
    kind = classify_prompt(text)
    prompt = record_prompt(store, subject, text, kind)
    outcome: Dict[str, Any] = {"prompt_id": prompt.id, "kind": kind, "actions": []}

    hints = parse_param_hints(subject, text)
    if hints:
        hyp = Hypothesis(
            subject_id=subject.id,
            title=", ".join(f"{k}={v}" for k, v in sorted(hints.items())),
            operator=_operator_for(subject, hints),
            params=hints,  # a delta against whatever the baseline is when it runs
            rationale=f"Requested by a human: {text.strip()[:400]}",
            origin="human",
            meta={"prompt_id": prompt.id},
        )
        store.add_hypothesis(hyp)
        outcome["actions"].append({"created_hypothesis": hyp.id, "params": hints})
        outcome["hypothesis_id"] = hyp.id

    if kind == "challenge":
        target = _resolve_challenge_target(store, subject, text)
        if target is not None:
            challenge = open_challenge(store, subject, target, reason=text.strip())
            outcome["actions"].append({"opened_challenge": challenge.id, "finding": target.id})
            outcome["challenge_id"] = challenge.id
        else:
            outcome["actions"].append({"note": "no finding to challenge yet"})

    if kind == "constraint":
        constraints = list(subject.config.get("constraints", []))
        constraints.append(text.strip())
        subject.config["constraints"] = constraints
        store.save_subject(subject)
        outcome["actions"].append({"recorded_constraint": text.strip()[:200]})

    if kind == "steer":
        focus = list(subject.config.get("focus", []))
        focus.append(text.strip())
        subject.config["focus"] = focus[-10:]
        store.save_subject(subject)
        outcome["actions"].append({"recorded_focus": text.strip()[:200]})

    return outcome


def _operator_for(subject: Subject, params: Dict[str, Any]) -> str:
    from arp.evolution import param_groups

    for group, names in param_groups(subject).items():
        if any(n in params for n in names):
            return f"tune:{group}"
    return "combine"


def _resolve_challenge_target(store: Store, subject: Subject, text: str) -> Optional[Finding]:
    """A challenge names a finding, or means "the most recent claim you made"."""
    for token in (text or "").split():
        token = token.strip(".,;:!?()[]")
        if token.startswith("fnd_"):
            finding = store.get_finding(token)
            if finding:
                return finding
    candidates = store.list_findings(subject.id, [Verdict.PROVEN, Verdict.SUPPORTED, Verdict.CONTESTED])
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------

def open_challenge(
    store: Store, subject: Subject, finding: Finding, reason: str,
    axis: Optional[Dict[str, List[Any]]] = None,
) -> Challenge:
    """Re-open a finding under a stricter bar and a wider grid."""
    challenge = Challenge(
        subject_id=subject.id, finding_id=finding.id, reason=reason,
        axis=axis or {},
    )
    store.add_challenge(challenge)
    if not challenge.axis:
        # Default axis: a fresh replication label. Because every grid cell key
        # now carries it, none of the existing confirmation trials count and the
        # whole grid is measured again from scratch.
        challenge.axis = {f"recheck_{challenge.id[-6:]}": ["fresh"]}
        store.save_challenge(challenge)

    finding.verdict = Verdict.CONTESTED
    finding.reason = f"challenged: {reason[:200]}"
    finding.proven_at = None
    note = unadopt(store, subject, finding)
    if note:
        finding.evidence = {**finding.evidence,
                            "challenge_notes": [*(finding.evidence.get("challenge_notes") or []), note]}
    store.save_finding(finding)

    hyp = store.get_hypothesis(finding.hypothesis_id)
    if hyp:
        hyp.status = Verdict.CONTESTED
        store.save_hypothesis(hyp)

    record_prompt(store, subject, reason, kind="challenge", meta={"challenge_id": challenge.id,
                                                                 "finding_id": finding.id})
    return challenge


def unadopt(store: Store, subject: Subject, finding: Finding) -> str:
    """Roll a challenged change back out of the baseline, if it was adopted.

    Re-testing an adopted change against a baseline that already contains it
    measures nothing — worse, it measures the reverse. So a challenge to an
    adopted finding reverts it first, exactly like reverting the commit before
    re-running the experiment. The epoch bumps, which invalidates the old
    evidence, and the standard machinery then re-runs the comparison properly.

    Returns a human-readable note, or "" if there was nothing to do.
    """
    history = list(subject.config.get("adopted") or [])
    index = next((i for i, e in enumerate(history) if e.get("finding_id") == finding.id), None)
    if index is None:
        return ""

    entry = history[index]
    previous = entry.get("previous") or {}
    if not previous:
        return "adopted before roll-back was recorded, so the baseline was left as it is"

    # If a later adoption touched the same parameters, this change is no longer
    # what the baseline holds: rolling it back would silently undo the newer,
    # better-evidenced result. Say so instead of doing damage.
    superseding = [
        later.get("title", "a later change")
        for later in history[index + 1:]
        if set(later.get("previous") or {}) & set(previous)
    ]
    if superseding:
        return (
            f"not rolled back: superseded by {', '.join(superseding)}, which now holds those "
            "parameters — challenge that finding instead"
        )

    params = dict(subject.baseline_params)
    params.update(previous)
    subject.config["baseline_params"] = params
    subject.config["epoch"] = int(subject.config.get("epoch", 0)) + 1
    entry["reverted_at"] = now_iso()
    history[index] = entry
    subject.config["adopted"] = history
    store.save_subject(subject)
    restored = ", ".join(f"{k}={v}" for k, v in sorted(previous.items()))
    return f"rolled the change back out of the baseline ({restored}) and re-opened the comparison"


def challenge_axes(store: Store, finding: Finding) -> Dict[str, Any]:
    """Every axis contributed by challenges on this finding, open or resolved.

    Resolved challenges keep their axis: once a question has been asked of a
    result, the answer stays part of the standard of proof for it.
    """
    axes: Dict[str, Any] = {}
    for challenge in store.list_challenges(finding_id=finding.id):
        for name, labels in (challenge.axis or {}).items():
            axes[name] = labels
    return axes


def resolve_challenges(store: Store, finding: Finding, verdict: str, note: str = "") -> int:
    """Close the open challenges on a finding once the re-test has landed.

    A challenge closes on any terminal verdict — proven *or* refuted. The human
    asked a question; both answers count as an answer.
    """
    if verdict not in Verdict.TERMINAL:
        return 0
    closed = 0
    for challenge in store.open_challenges(finding.id):
        challenge.status = "resolved"
        challenge.resolved_at = now_iso()
        challenge.resolution = f"{verdict}: {note or finding.reason}"[:500]
        store.save_challenge(challenge)
        closed += 1
    return closed


def auto_challenge(
    store: Store, subject: Subject, finding: Finding, critic: Optional[Critic] = None,
    max_new: int = 1,
) -> List[Challenge]:
    """The platform second-guessing itself before it claims anything is proven.

    Only critiques that come with a testable axis become challenges; the rest are
    recorded as notes on the finding so a human can see what was considered.
    """
    critic = critic or Critic(store)
    hyp = store.get_hypothesis(finding.hypothesis_id)
    if hyp is None:
        return []
    critiques = critic.critique(subject, finding, hyp)
    if not critiques:
        return []

    notes = list((finding.evidence.get("critiques") or []))
    opened: List[Challenge] = []
    existing_reasons = {c.reason for c in store.list_challenges(finding_id=finding.id)}
    for critique in critiques:
        notes.append({"concern": critique.concern, "severity": critique.severity})
        if len(opened) >= max_new or critique.severity != "high":
            continue
        if critique.concern in existing_reasons:
            continue
        opened.append(open_challenge(store, subject, finding, critique.concern, critique.axis or None))
    finding.evidence["critiques"] = notes[-10:]
    store.save_finding(finding)
    return opened
