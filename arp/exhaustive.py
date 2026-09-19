"""The exhaustive confirmation stage: where a supported hypothesis becomes proven.

The screening stage answers "does this help, on the nominal setup?". That is not
enough to act on — most tuning wins are artefacts of the one configuration they
were found in. So a supported hypothesis is re-run across a *finite, enumerable*
grid of stress cells (scale, data slice, whatever the subject declares, plus any
axis a human attached to a challenge), with a fresh baseline measured inside
every cell.

PROVEN requires, with no exceptions and no agent judgement:
  1. every cell filled to the replicate floor, on both arms,
  2. no cell regressing beyond the tolerance,
  3. the pooled effect clearing the MDE at the stricter confirmation alpha,
  4. no open human challenges.

Anything else is REFUTED, INCONCLUSIVE, or still TESTING. Because the grid is
enumerable and the rule is arithmetic, the verdict is a deterministic function of
the trial table.
"""

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arp.config import DecisionPolicy
from arp.models import Hypothesis, Subject, Trial, Verdict
from arp.search import derive_seed, hypothesis_key
from arp.stats import EffectEstimate, estimate_effect, holm_bonferroni
from arp.store import Store


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def merged_axes(subject: Subject, extra_axes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    axes: Dict[str, Any] = dict(subject.stress_axes or {})
    for name, spec in (extra_axes or {}).items():
        axes[name] = spec
    return axes


def axis_labels(spec: Any) -> List[Any]:
    """An axis is either a list of labels or a mapping label -> parameter overrides."""
    if isinstance(spec, dict):
        return list(spec.keys())
    if isinstance(spec, (list, tuple)):
        return list(spec)
    return [spec]


def grid_cells(subject: Subject, extra_axes: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every cell of the confirmation grid, in a stable order."""
    axes = merged_axes(subject, extra_axes)
    if not axes:
        return [{}]
    names = sorted(axes)
    label_lists = [axis_labels(axes[n]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*label_lists)]


def cell_key(cell: Dict[str, Any]) -> str:
    return json.dumps(cell, sort_keys=True, default=str)


def cell_overrides(
    subject: Subject, cell: Dict[str, Any], extra_axes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parameter overrides implied by a cell (for axes declared as label -> params)."""
    axes = merged_axes(subject, extra_axes)
    overrides: Dict[str, Any] = {}
    for axis, label in (cell or {}).items():
        spec = axes.get(axis)
        if isinstance(spec, dict) and isinstance(spec.get(label), dict):
            overrides.update(spec[label])
    return overrides


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class TrialRequest:
    cell: Dict[str, Any]
    arm: str
    seed: int
    replicate: int


@dataclass
class CellResult:
    cell: Dict[str, Any]
    estimate: EffectEstimate
    n_baseline: int
    n_treatment: int
    passed: bool
    regressed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell": self.cell,
            "effect": round(self.estimate.effect, 6),
            "ci_low": round(self.estimate.ci_low, 6),
            "ci_high": round(self.estimate.ci_high, 6),
            "n_baseline": self.n_baseline,
            "n_treatment": self.n_treatment,
            "passed": self.passed,
            "regressed": self.regressed,
        }


@dataclass
class ConfirmationResult:
    status: str
    reason: str
    cells: List[CellResult] = field(default_factory=list)
    pooled: EffectEstimate = field(default_factory=EffectEstimate)
    pending: List[TrialRequest] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.pending

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "cells": [c.as_dict() for c in self.cells],
            "pooled": self.pooled.as_dict(),
            "pending": len(self.pending),
        }


def plan_confirmation(
    store: Store,
    subject: Subject,
    hypothesis: Hypothesis,
    policy: DecisionPolicy,
    salt: str,
    extra_axes: Optional[Dict[str, Any]] = None,
    replicates: Optional[int] = None,
    targets: Optional[Dict[str, int]] = None,
) -> List[TrialRequest]:
    """Which confirmation trials are still missing, in a deterministic order.

    `targets` raises the replicate count for named cells — that is how a cell
    whose interval is still too wide asks for more evidence instead of being
    written off.
    """
    replicates = replicates or policy.min_replicates
    targets = targets or {}
    trials = store.list_trials(subject_id=subject.id, stage="confirm")
    have: Dict[Tuple[str, str], int] = {}
    for t in trials:
        if t.arm == "treatment" and t.hypothesis_id != hypothesis.id:
            continue
        have[(cell_key(t.cell), t.arm)] = have.get((cell_key(t.cell), t.arm), 0) + 1

    requests: List[TrialRequest] = []
    for cell in grid_cells(subject, extra_axes):
        key = cell_key(cell)
        want = min(max(replicates, targets.get(key, replicates)), policy.max_replicates)
        for arm in ("baseline", "treatment"):
            done = have.get((key, arm), 0)
            for i in range(done, want):
                requests.append(
                    TrialRequest(
                        cell=cell,
                        arm=arm,
                        seed=derive_seed(
                            salt,
                            hypothesis_key(hypothesis) if arm == "treatment" else "baseline",
                            key, arm, i,
                        ),
                        replicate=i,
                    )
                )
    # Round-robin: replicate 0 of every cell and arm before replicate 1 of any.
    # An interrupted run then leaves balanced evidence everywhere instead of one
    # finished cell and nothing else.
    requests.sort(key=lambda r: (r.replicate, cell_key(r.cell), r.arm))
    return requests


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _values(trials: Sequence[Trial], key: str, arm: str, hypothesis_id: Optional[str]) -> List[float]:
    out = []
    for t in trials:
        if cell_key(t.cell) != key or t.arm != arm or t.metric_value is None:
            continue
        if arm == "treatment" and hypothesis_id and t.hypothesis_id != hypothesis_id:
            continue
        out.append(t.metric_value)
    return out


def evaluate_confirmation(
    store: Store,
    subject: Subject,
    hypothesis: Hypothesis,
    policy: DecisionPolicy,
    salt: str,
    extra_axes: Optional[Dict[str, Any]] = None,
    open_challenges: int = 0,
) -> ConfirmationResult:
    """Apply the four PROVEN conditions to whatever evidence exists right now."""
    effective = policy.tightened(open_challenges)
    replicates = effective.min_replicates
    trials = store.list_trials(subject_id=subject.id, stage="confirm")
    cells = grid_cells(subject, extra_axes)

    cell_results: List[CellResult] = []
    pooled_baseline: List[float] = []
    pooled_treatment: List[float] = []
    per_cell_values: List[Tuple[Dict[str, Any], List[float], List[float]]] = []
    incomplete: List[Dict[str, Any]] = []

    for cell in cells:
        key = cell_key(cell)
        base = _values(trials, key, "baseline", None)
        treat = _values(trials, key, "treatment", hypothesis.id)
        est = estimate_effect(base, treat, subject.direction, effective.alpha_confirm, effective.cs_rho)
        complete = len(base) >= replicates and len(treat) >= replicates
        if not complete:
            incomplete.append(cell)
        regressed = complete and est.ci_high <= -effective.regression_tolerance
        passed = complete and est.ci_low >= -effective.regression_tolerance
        cell_results.append(
            CellResult(cell=cell, estimate=est, n_baseline=len(base), n_treatment=len(treat),
                       passed=passed, regressed=regressed)
        )
        per_cell_values.append((cell, base, treat))

    # Pool the cells *stratified*: cells sit at different metric levels by
    # construction (a bigger model has a different loss), so throwing the raw
    # numbers into one bucket would charge the between-cell spread to the
    # treatment as noise. Rescaling each cell to the grand baseline mean removes
    # the cell effect and leaves the within-cell comparison intact.
    cell_means = [sum(base) / len(base) for _, base, _ in per_cell_values if base]
    grand_mean = sum(cell_means) / len(cell_means) if cell_means else 0.0
    for _, base, treat in per_cell_values:
        if not base:
            continue
        cell_mean = sum(base) / len(base)
        scale = grand_mean / cell_mean if abs(cell_mean) > 1e-12 else 1.0
        pooled_baseline.extend(b * scale for b in base)
        pooled_treatment.extend(t * scale for t in treat)

    pooled = estimate_effect(
        pooled_baseline, pooled_treatment, subject.direction, effective.alpha_confirm, effective.cs_rho
    )
    # Cells whose interval still spans the regression tolerance are *unresolved*,
    # not failed: the honest response to "we can't tell yet" is more replicates
    # in exactly those cells, up to the replicate budget.
    unresolved = [c for c in cell_results if not c.passed and not c.regressed]
    targets = {
        cell_key(c.cell): min(effective.max_replicates, max(c.n_baseline, c.n_treatment) + 2)
        for c in unresolved
        if not any(cell_key(c.cell) == cell_key(x) for x in incomplete)
    }
    pending = plan_confirmation(
        store, subject, hypothesis, effective, salt, extra_axes, replicates, targets
    )

    # A cell that has already regressed conclusively kills the thesis now; no
    # point paying for the rest of the grid.
    regressors = [c for c in cell_results if c.regressed]
    if regressors:
        where = ", ".join(cell_key(c.cell) for c in regressors)
        return ConfirmationResult(
            Verdict.REFUTED,
            f"regression in {len(regressors)} of {len(cells)} stress cell(s): {where}",
            cell_results, pooled, [],
        )

    if incomplete:
        # While a challenge is open the finding stays visibly CONTESTED, but the
        # work is the same work: fill the (now larger, now stricter) grid.
        return ConfirmationResult(
            Verdict.CONTESTED if open_challenges else Verdict.SUPPORTED,
            f"{len(incomplete)} of {len(cells)} stress cell(s) still need replicates "
            f"({len(pending)} trials queued"
            + (f", under {open_challenges} open challenge(s)" if open_challenges else "")
            + ")",
            cell_results, pooled, pending,
        )

    if unresolved and pending:
        where = ", ".join(cell_key(c.cell) for c in unresolved)
        return ConfirmationResult(
            Verdict.CONTESTED if open_challenges else Verdict.SUPPORTED,
            f"{len(unresolved)} cell(s) still cannot rule out a regression "
            f"({where}); {len(pending)} more trial(s) queued",
            cell_results, pooled, pending,
        )

    if unresolved:
        where = ", ".join(cell_key(c.cell) for c in unresolved)
        return ConfirmationResult(
            Verdict.INCONCLUSIVE,
            f"replicate budget spent without ruling out a regression in: {where}",
            cell_results, pooled, [],
        )

    if pooled.ci_low >= effective.mde:
        return ConfirmationResult(
            Verdict.PROVEN,
            f"holds in all {len(cells)} stress cells; pooled effect {pooled.effect:+.4%} "
            f"with lower bound {pooled.ci_low:+.4%} at alpha={effective.alpha_confirm:g}",
            cell_results, pooled, [],
        )

    if pooled.ci_high < effective.mde:
        return ConfirmationResult(
            Verdict.REFUTED,
            f"grid complete but pooled upper bound {pooled.ci_high:+.4%} cannot reach "
            f"the MDE of {effective.mde:.2%}",
            cell_results, pooled, [],
        )

    return ConfirmationResult(
        Verdict.INCONCLUSIVE,
        f"grid complete; pooled interval [{pooled.ci_low:+.4%}, {pooled.ci_high:+.4%}] "
        "straddles the decision threshold",
        cell_results, pooled, [],
    )


def family_wise_filter(store: Store, subject: Subject, policy: DecisionPolicy) -> Dict[str, bool]:
    """Holm-Bonferroni across a subject's findings: which survive multiplicity?

    Run enough hypotheses and some will clear any per-test bar by luck. This is
    reported alongside each finding so a reader can see whether a result is
    still standing once the whole family is accounted for.
    """
    findings = store.list_findings(subject.id, [Verdict.SUPPORTED, Verdict.PROVEN, Verdict.CONTESTED])
    if not findings:
        return {}
    mask = holm_bonferroni([f.p_value for f in findings], policy.alpha_confirm)
    return {f.id: survived for f, survived in zip(findings, mask)}
