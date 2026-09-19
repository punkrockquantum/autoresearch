"""Subject suggestion: what should we research next?

The human asks for a subject; the platform should be able to answer "based on
everything you've told me and everything I've found, here is what I'd look at
next, and why". Five sources feed it, all of them things the platform already
has on disk:

    1. recurring themes in the human's own prompts that no subject covers yet
    2. operators with a good track record but thin evidence on a subject
    3. near-misses: findings that were refuted or inconclusive but close
    4. proven findings worth transferring to a similar subject
    5. high-demand web leads with no subject attached

Scores are deterministic. An LLM, when available, only rewrites the phrasing —
it never changes the ranking.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from arp.agents import LLMClient, extract_json
from arp.evolution import CapabilityMemory, operator_catalog, similarity, subject_tokens, tokenize
from arp.models import Subject, Verdict
from arp.store import Store


@dataclass
class Suggestion:
    title: str
    rationale: str
    score: float
    kind: str                       # frontier | near_miss | transfer | theme | market
    subject_slug: Optional[str] = None
    seed_prompt: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        where = f" [{self.subject_slug}]" if self.subject_slug else ""
        return f"{self.score:5.1f}  {self.kind:10s}{where} {self.title}\n         {self.rationale}"


def suggest(store: Store, subject: Optional[Subject] = None, limit: int = 8) -> List[Suggestion]:
    """Rank what to look at next, optionally focused on one subject."""
    subjects = [subject] if subject else store.list_subjects()
    out: List[Suggestion] = []
    for subj in subjects:
        out.extend(_frontier(store, subj))
        out.extend(_near_misses(store, subj))
        out.extend(_transfers(store, subj))
        out.extend(_market(store, subj))
    out.extend(_themes(store, subjects))
    out.sort(key=lambda s: s.score, reverse=True)
    return out[:limit]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _frontier(store: Store, subject: Subject) -> List[Suggestion]:
    """Operators that look promising here but have barely been tried."""
    memory = CapabilityMemory(store)
    out: List[Suggestion] = []
    for operator in operator_catalog(subject):
        cap = store.get_capability(operator, subject.id)
        n = cap.n if cap else 0
        prior = memory.prior_for(subject, operator)
        if n >= 6 or prior <= 0:
            continue
        # High expected gain, low evidence: exactly where more compute pays.
        score = 100.0 * min(1.0, prior / 0.01) / (1.0 + n)
        if score < 1.0:
            continue
        out.append(Suggestion(
            title=f"Go deeper on {operator} for {subject.slug}",
            rationale=(
                f"Expected gain {prior:+.3%} per trial from only {n} observation(s) — "
                "the cheapest unexplored direction on this subject."
            ),
            score=score, kind="frontier", subject_slug=subject.slug,
            seed_prompt=f"Focus the next runs on {operator}; I want that family tested properly.",
            evidence={"operator": operator, "n": n, "prior": prior},
        ))
    return out


def _near_misses(store: Store, subject: Subject) -> List[Suggestion]:
    """Things that nearly worked. Often a better bet than anything brand new."""
    out: List[Suggestion] = []
    for finding in store.list_findings(subject.id, [Verdict.INCONCLUSIVE, Verdict.REFUTED]):
        if finding.ci_high <= 0:
            continue
        hyp = store.get_hypothesis(finding.hypothesis_id)
        if hyp is None:
            continue
        headroom = finding.ci_high
        score = 60.0 * min(1.0, headroom / 0.01)
        out.append(Suggestion(
            title=f"Re-test '{hyp.title}' with more replicates",
            rationale=(
                f"Came back {finding.verdict} at {finding.effect:+.3%}, but the interval still "
                f"reaches {finding.ci_high:+.3%}: the budget ran out before the question did."
            ),
            score=score, kind="near_miss", subject_slug=subject.slug,
            seed_prompt=f"Re-open {finding.id} and give it a bigger replicate budget.",
            evidence={"finding_id": finding.id, "ci_high": finding.ci_high},
        ))
    return out


def _transfers(store: Store, subject: Subject) -> List[Suggestion]:
    """Proven results on similar subjects that have never been tried here."""
    out: List[Suggestion] = []
    my_operators = {h.operator for h in store.list_hypotheses(subject.id)}
    for other in store.list_subjects():
        if other.id == subject.id:
            continue
        sim = similarity(subject, other)
        if sim < 0.3:
            continue
        for finding in store.list_findings(other.id, [Verdict.PROVEN]):
            hyp = store.get_hypothesis(finding.hypothesis_id)
            if hyp is None or hyp.operator in my_operators:
                continue
            score = 80.0 * sim * min(1.0, abs(finding.effect) / 0.01)
            out.append(Suggestion(
                title=f"Port '{hyp.title}' from {other.slug} to {subject.slug}",
                rationale=(
                    f"Proven there at {finding.effect:+.3%}; the two subjects overlap "
                    f"{sim:.0%} and this operator has never been tried here."
                ),
                score=score, kind="transfer", subject_slug=subject.slug,
                seed_prompt=f"Try the {hyp.operator} change that worked on {other.slug}: {hyp.title}",
                evidence={"source_subject": other.slug, "finding_id": finding.id},
            ))
    return out


def _market(store: Store, subject: Subject) -> List[Suggestion]:
    """Web leads with strong commercial signal that nothing is chasing."""
    leads = store.list_leads(subject.id)
    out: List[Suggestion] = []
    for lead in leads[:40]:
        s = lead.scores or {}
        commercial = float(s.get("commercial", 0.0))
        authority = float(s.get("authority", 0.0))
        if commercial < 0.4 or authority < 0.6:
            continue
        out.append(Suggestion(
            title=f"Chase the commercial angle in '{lead.title[:80]}'",
            rationale=(
                f"Commercial signal {commercial:.2f} from a {authority:.2f}-authority source "
                f"({lead.url}); no hypothesis on this subject targets it yet."
            ),
            score=50.0 * commercial * authority, kind="market", subject_slug=subject.slug,
            seed_prompt=f"Frame a subject around: {lead.title[:120]}",
            evidence={"lead_id": lead.id, "url": lead.url},
        ))
    return out[:3]


def _themes(store: Store, subjects: Sequence[Subject]) -> List[Suggestion]:
    """Words the human keeps using that no subject is actually about."""
    covered: set = set()
    for subj in store.list_subjects():
        covered |= subject_tokens(subj)

    counts: Counter = Counter()
    examples: Dict[str, str] = {}
    for prompt in store.list_prompts(limit=400):
        if prompt.role != "human":
            continue
        for token in set(tokenize(prompt.text)):
            counts[token] += 1
            examples.setdefault(token, prompt.text.strip())

    out: List[Suggestion] = []
    for token, count in counts.most_common(40):
        if token in covered or count < 3:
            continue
        out.append(Suggestion(
            title=f"Open a subject about '{token}'",
            rationale=(
                f"You have raised '{token}' in {count} prompts but no subject covers it. "
                f"Most recent mention: \"{examples.get(token, '')[:140]}\""
            ),
            score=20.0 * math.log1p(count), kind="theme",
            seed_prompt=f"Create a subject about {token} and propose how to measure it.",
            evidence={"token": token, "mentions": count},
        ))
    return out[:4]


# ---------------------------------------------------------------------------
# Optional LLM phrasing pass
# ---------------------------------------------------------------------------

def polish(suggestions: Sequence[Suggestion], llm: Optional[LLMClient] = None) -> List[Suggestion]:
    """Rewrite titles and rationales in the human's terms. Ranking is untouched."""
    llm = llm or LLMClient()
    items = list(suggestions)
    if not llm.available or not items:
        return items
    payload = [
        {"i": i, "title": s.title, "rationale": s.rationale, "kind": s.kind}
        for i, s in enumerate(items)
    ]
    raw = llm.complete(
        system=(
            "You sharpen research suggestions. Keep every item, keep its index and its meaning, "
            "keep the numbers. Return JSON only."
        ),
        user=(
            "Rewrite each suggestion so a busy researcher can act on it. "
            "Return a JSON array of {\"i\": <index>, \"title\": ..., \"rationale\": ...}.\n\n"
            + str(payload)
        ),
        max_tokens=1500,
    )
    parsed = extract_json(raw or "")
    if not isinstance(parsed, list):
        return items
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            items[idx].title = str(entry.get("title") or items[idx].title)[:200]
            items[idx].rationale = str(entry.get("rationale") or items[idx].rationale)[:500]
    return items
