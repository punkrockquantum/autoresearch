"""Business impact: which proven results are worth acting on, and why.

The scoring is an explicit formula over five inputs, each of which the platform
actually measured:

    confidence  — how far through the verdict ladder the finding got
    magnitude   — effect size against the subject's reference effect
    durability  — fraction of stress cells it survived
    demand      — market signal from the web leads
    effort      — declared cost of shipping it

Nothing here is an LLM's opinion, so two runs over the same evidence rank the
same way. If the subject declares a `value_model`, the score is also converted
into money; without one you get the ranking and the components, which is usually
what a decision needs anyway.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arp.models import Finding, Hypothesis, Lead, Subject, Verdict
from arp.store import Store
from arp.websearch import market_signal

# How much we trust a finding, by verdict. Refuted findings score zero by
# construction: the point of the ladder is that unproven things cannot be sold.
CONFIDENCE = {
    Verdict.PROVEN: 1.0,
    Verdict.SUPPORTED: 0.55,
    Verdict.CONTESTED: 0.35,
    Verdict.TESTING: 0.2,
    Verdict.PROPOSED: 0.05,
    Verdict.INCONCLUSIVE: 0.05,
    Verdict.REFUTED: 0.0,
}


@dataclass
class ImpactScore:
    total: float
    confidence: float
    magnitude: float
    durability: float
    demand: float
    effort: float
    monetary: Optional[float] = None
    currency: str = "USD"
    rationale: str = ""
    components: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 2),
            "confidence": round(self.confidence, 4),
            "magnitude": round(self.magnitude, 4),
            "durability": round(self.durability, 4),
            "demand": round(self.demand, 4),
            "effort": round(self.effort, 4),
            "monetary": self.monetary,
            "currency": self.currency,
            "rationale": self.rationale,
            **self.components,
        }


def durability_of(finding: Finding) -> float:
    """Fraction of exhaustive-grid cells the finding held up in."""
    cells = (finding.evidence.get("confirmation") or {}).get("cells") or []
    if not cells:
        return 0.0
    passed = sum(1 for c in cells if c.get("passed"))
    return passed / len(cells)


def score_finding(
    subject: Subject,
    finding: Finding,
    leads: Sequence[Lead] = (),
    hypothesis: Optional[Hypothesis] = None,
) -> ImpactScore:
    value_model: Dict[str, Any] = subject.config.get("value_model", {})
    reference = float(value_model.get("reference_effect", 0.01))  # 1% is "a big win" by default
    effort_days = float(value_model.get("effort_days", 5.0))
    if hypothesis is not None:
        effort_days = float((hypothesis.meta or {}).get("effort_days", effort_days))

    confidence = CONFIDENCE.get(finding.verdict, 0.1)
    # Use the conservative end of the interval: we sell the bound, not the mean.
    conservative = max(0.0, finding.ci_low if finding.ci_low > 0 else finding.effect * 0.5)
    magnitude = min(1.0, (conservative / reference) ** 0.7) if reference > 0 and conservative > 0 else 0.0
    durability = durability_of(finding)
    demand = market_signal(leads)
    effort = 1.0 / (1.0 + effort_days / 10.0)

    total = 100.0 * confidence * magnitude * (0.5 + 0.5 * durability) * (0.4 + 0.6 * demand) * effort

    monetary = None
    unit_value = value_model.get("value_per_relative_point")
    if unit_value:
        # e.g. "1.0 relative improvement in val_bpb is worth $2M/year of compute"
        monetary = round(float(unit_value) * conservative, 2)

    rationale = _rationale(subject, finding, confidence, magnitude, durability, demand, effort, leads)
    return ImpactScore(
        total=total,
        confidence=confidence,
        magnitude=magnitude,
        durability=durability,
        demand=demand,
        effort=effort,
        monetary=monetary,
        currency=str(value_model.get("currency", "USD")),
        rationale=rationale,
        components={
            "conservative_effect": round(conservative, 6),
            "reference_effect": reference,
            "effort_days": effort_days,
            "n_leads": len(leads),
        },
    )


def _rationale(
    subject: Subject, finding: Finding, confidence: float, magnitude: float,
    durability: float, demand: float, effort: float, leads: Sequence[Lead],
) -> str:
    bits = [
        f"{finding.verdict} finding worth {finding.effect:+.3%} on {subject.metric} "
        f"(conservative bound {finding.ci_low:+.3%})",
        f"held in {durability:.0%} of stress cells" if durability else "not yet stress-tested",
    ]
    if leads:
        top = max(leads, key=lambda l: float((l.scores or {}).get("authority", 0)))
        bits.append(f"market signal {demand:.2f} from {len(leads)} sources, strongest: {domain(top.url)}")
    else:
        bits.append("no external demand evidence gathered yet")
    bits.append(f"shipping effort discount {effort:.2f}")
    return "; ".join(bits)


def domain(url: str) -> str:
    from arp.websearch import domain_of

    return domain_of(url) or url


def rank_opportunities(
    store: Store, subject: Subject, verdicts: Sequence[str] = (Verdict.PROVEN, Verdict.SUPPORTED),
    persist: bool = True,
) -> List[Tuple[Finding, ImpactScore]]:
    """Score every finding worth scoring and return them best-first."""
    out: List[Tuple[Finding, ImpactScore]] = []
    for finding in store.list_findings(subject.id, verdicts):
        leads = store.list_leads(subject.id, finding.id)
        hypothesis = store.get_hypothesis(finding.hypothesis_id)
        score = score_finding(subject, finding, leads, hypothesis)
        if persist:
            finding.impact = score.as_dict()
            store.save_finding(finding)
        out.append((finding, score))
    out.sort(key=lambda pair: pair[1].total, reverse=True)
    return out


def business_brief(subject: Subject, finding: Finding, score: ImpactScore,
                   hypothesis: Optional[Hypothesis], leads: Sequence[Lead] = ()) -> str:
    """A short, plain-English write-up of one opportunity."""
    title = hypothesis.title if hypothesis else finding.hypothesis_id
    lines = [
        f"### {title}",
        "",
        f"**Impact score {score.total:.1f}/100** — {finding.verdict.upper()} "
        f"({finding.effect:+.3%} on {subject.metric}, interval "
        f"[{finding.ci_low:+.3%}, {finding.ci_high:+.3%}], n={finding.n_trials})",
        "",
        f"- Confidence: {score.confidence:.2f} ({finding.verdict})",
        f"- Magnitude: {score.magnitude:.2f} vs reference effect "
        f"{score.components.get('reference_effect')}",
        f"- Durability: {score.durability:.0%} of stress cells passed",
        f"- Demand: {score.demand:.2f} from {score.components.get('n_leads', 0)} web sources",
        f"- Effort discount: {score.effort:.2f} "
        f"({score.components.get('effort_days')} days assumed)",
    ]
    if score.monetary is not None:
        lines.append(f"- Estimated annual value: {score.monetary:,.0f} {score.currency}")
    if hypothesis and hypothesis.params:
        changed = {k: v for k, v in hypothesis.params.items()
                   if subject.baseline_params.get(k) != v}
        if changed:
            lines += ["", "**Change to ship:**", "", "```", *(f"{k} = {v}" for k, v in sorted(changed.items())), "```"]
    if leads:
        lines += ["", "**Evidence from the web:**", ""]
        ranked = sorted(leads, key=lambda l: float((l.scores or {}).get("authority", 0)), reverse=True)
        for lead in ranked[:5]:
            lines.append(f"- [{lead.title}]({lead.url}) — {domain(lead.url)}")
    lines += ["", f"_{score.rationale}_"]
    return "\n".join(lines)
