"""The probabilistic-to-deterministic core.

The search over hypotheses is random; the *verdict* must not be. Everything in
this module is a pure function of the trial table plus a `DecisionPolicy`, so
two people running `arp` on the same data get the same answer — no agent
judgement anywhere in the decision path.

Three ingredients:

1. **Welch's t-test** for a classical, fixed-sample p-value (reported, and used
   for multiplicity control across many hypotheses).
2. **Anytime-valid confidence sequences** (normal-mixture boundary, Howard et
   al. 2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences").
   These stay valid under continuous peeking, which matters because the bandit
   looks at the numbers after every single trial.
3. **A sequential probability ratio test** (Wald) as an accelerator: it lets an
   obvious win or an obvious dud stop early without inflating error rates.

A hypothesis is only ever promoted to PROVEN by `arp.exhaustive`, which requires
every cell of a finite, enumerable stress grid to agree. Probabilistic search,
deterministic landing.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from arp.config import DecisionPolicy


# ---------------------------------------------------------------------------
# Distribution helpers (stdlib only)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """One-sided survival function P(T > t) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def student_t_two_sided_p(t: float, df: float) -> float:
    return min(1.0, 2.0 * student_t_sf(abs(t), df))


def norm_ppf(p: float) -> float:
    return statistics.NormalDist().inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def student_t_critical(alpha: float, df: float) -> float:
    """Two-sided critical value t* with P(|T| > t*) = alpha, by bisection.

    No closed form in the standard library, and bisection over a monotone tail
    is fast enough to run inside the decision loop.
    """
    if df <= 0:
        return float("inf")
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def small_sample_inflation(alpha: float, df: float) -> float:
    """How much to widen a bound that uses an *estimated* rather than known sigma.

    The confidence-sequence boundary assumes a known variance proxy. Feed it a
    sample standard deviation from five runs and it under-covers badly — which
    is exactly the regime research experiments live in. Scaling by the ratio of
    the t critical value to the normal one restores honest coverage and fades to
    1.0 as evidence accumulates.
    """
    if df <= 0:
        return float("inf")
    z = norm_ppf(1.0 - alpha / 2.0)
    if z <= 0:
        return 1.0
    return max(1.0, student_t_critical(alpha, df) / z)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@dataclass
class TTest:
    t: float
    df: float
    p_value: float
    diff: float
    stderr: float


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> TTest:
    """Welch's unequal-variance t-test on `a` minus `b`."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        diff = (statistics.fmean(a) if na else 0.0) - (statistics.fmean(b) if nb else 0.0)
        return TTest(t=0.0, df=0.0, p_value=1.0, diff=diff, stderr=float("inf"))
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        # Zero variance on both arms: any difference at all is exact.
        diff = ma - mb
        return TTest(t=0.0 if diff == 0 else math.copysign(float("inf"), diff),
                     df=float(na + nb - 2), p_value=0.0 if diff else 1.0, diff=diff, stderr=0.0)
    se = math.sqrt(se2)
    t = (ma - mb) / se
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return TTest(t=t, df=df, p_value=student_t_two_sided_p(t, df), diff=ma - mb, stderr=se)


def cs_radius(n: int, sigma: float, alpha: float, rho: float = 1.0) -> float:
    """Normal-mixture confidence-sequence radius for a mean after `n` samples.

    Valid at every n simultaneously, which is what makes it safe to peek after
    every trial and stop the moment the interval clears the threshold.
    """
    if n <= 0 or sigma <= 0:
        return float("inf")
    alpha = min(max(alpha, 1e-12), 0.5)
    inner = (2.0 * (n * rho + 1.0) / (n * n * rho)) * math.log(math.sqrt(n * rho + 1.0) / alpha)
    return sigma * math.sqrt(max(inner, 0.0))


def sprt_decision(
    values: Sequence[float], delta: float, sigma: float, alpha: float, beta: float = 0.1
) -> str:
    """Wald's SPRT for H0: mean == 0 vs H1: mean == delta, Gaussian with known sigma.

    Returns "accept_h1", "accept_h0" or "continue".
    """
    n = len(values)
    if n == 0 or sigma <= 0 or delta <= 0:
        return "continue"
    total = math.fsum(values)
    # Log-likelihood ratio for a Gaussian shift of `delta`.
    llr = (delta * total - 0.5 * n * delta * delta) / (sigma * sigma)
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    if llr >= upper:
        return "accept_h1"
    if llr <= lower:
        return "accept_h0"
    return "continue"


def holm_bonferroni(p_values: Sequence[float], alpha: float) -> List[bool]:
    """Holm step-down correction. Returns a rejection mask aligned with the input.

    Used to keep the family-wise error rate honest when a subject accumulates
    dozens of hypotheses — without it, "run enough experiments" manufactures
    significance on its own.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


# ---------------------------------------------------------------------------
# Effect estimation
# ---------------------------------------------------------------------------

@dataclass
class EffectEstimate:
    """Effect of a treatment arm against baseline, in relative (fractional) units.

    Positive always means "better", whatever the subject's optimisation
    direction is, so downstream code never has to branch on direction again.
    """

    effect: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float = 1.0
    n_baseline: int = 0
    n_treatment: int = 0
    baseline_mean: float = 0.0
    treatment_mean: float = 0.0
    sigma: float = 0.0
    t_stat: float = 0.0
    df: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_baseline": self.n_baseline,
            "n_treatment": self.n_treatment,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "sigma": self.sigma,
        }


def estimate_effect(
    baseline: Sequence[float],
    treatment: Sequence[float],
    direction: str = "minimize",
    alpha: float = 0.05,
    rho: float = 1.0,
) -> EffectEstimate:
    """Relative improvement of `treatment` over `baseline`, with an anytime-valid CI."""
    nb, nt = len(baseline), len(treatment)
    if nb == 0 or nt == 0:
        return EffectEstimate(n_baseline=nb, n_treatment=nt)

    mb = statistics.fmean(baseline)
    mt = statistics.fmean(treatment)
    scale = abs(mb) if abs(mb) > 1e-12 else 1.0
    raw_gain = (mb - mt) if direction == "minimize" else (mt - mb)
    effect = raw_gain / scale

    tt = welch_ttest(treatment, baseline)
    vb = statistics.variance(baseline) if nb > 1 else 0.0
    vt = statistics.variance(treatment) if nt > 1 else 0.0
    pooled_var = ((max(nb - 1, 0)) * vb + (max(nt - 1, 0)) * vt) / max(nb + nt - 2, 1)
    sigma = math.sqrt(max(pooled_var, 0.0))

    # Effective sample size for a difference of two means, rounded *down*: at the
    # three-or-four-replicate sizes real experiments run at, rounding up costs
    # real coverage, and being slightly slow to call a win is the cheaper error.
    n_eff = int(max(1, math.floor(1.0 / (1.0 / max(nb, 1) + 1.0 / max(nt, 1)))))
    inflation = small_sample_inflation(alpha, df=max(nb + nt - 2, 0))
    radius_abs = cs_radius(n_eff, sigma, alpha, rho) * inflation if sigma > 0 else 0.0
    if not math.isfinite(radius_abs):
        radius_abs = abs(raw_gain) + scale  # degenerate: interval covers everything useful
    radius = radius_abs / scale

    return EffectEstimate(
        effect=effect,
        ci_low=effect - radius,
        ci_high=effect + radius,
        p_value=tt.p_value,
        n_baseline=nb,
        n_treatment=nt,
        baseline_mean=mb,
        treatment_mean=mt,
        sigma=sigma / scale,
        t_stat=tt.t,
        df=tt.df,
    )


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    status: str                      # a Verdict value
    reason: str
    estimate: EffectEstimate
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def conclusive(self) -> bool:
        from arp.models import Verdict

        return self.status in (Verdict.SUPPORTED, Verdict.REFUTED, Verdict.INCONCLUSIVE)


def decide(
    baseline: Sequence[float],
    treatment: Sequence[float],
    direction: str,
    policy: DecisionPolicy,
    open_challenges: int = 0,
) -> Decision:
    """Screening verdict for one hypothesis. Pure function of its arguments.

    Order of checks matters and is fixed:
      1. not enough evidence yet -> keep testing
      2. lower bound clears the minimum detectable effect -> SUPPORTED
      3. upper bound is below zero by more than the MDE -> REFUTED
      4. whole interval sits inside the region of practical equivalence -> INCONCLUSIVE
      5. SPRT boundary crossed -> SUPPORTED / REFUTED
      6. replicate budget exhausted -> REFUTED if it cannot be a worthwhile win,
         otherwise INCONCLUSIVE
      7. otherwise -> keep testing
    """
    from arp.models import Verdict

    policy = policy.tightened(open_challenges)
    est = estimate_effect(baseline, treatment, direction, policy.alpha, policy.cs_rho)
    stats: Dict[str, float] = {
        "alpha": policy.alpha,
        "mde": policy.mde,
        "rope": policy.rope,
        "min_replicates": policy.min_replicates,
        "open_challenges": open_challenges,
    }

    if est.n_baseline < policy.min_replicates or est.n_treatment < policy.min_replicates:
        return Decision(
            Verdict.TESTING,
            f"need {policy.min_replicates} replicates per arm "
            f"(have {est.n_baseline} baseline / {est.n_treatment} treatment)",
            est,
            stats,
        )

    if est.ci_low >= policy.mde:
        return Decision(
            Verdict.SUPPORTED,
            f"anytime-valid lower bound {est.ci_low:+.4%} clears the MDE of {policy.mde:.2%}",
            est,
            stats,
        )

    if est.ci_high <= -policy.mde:
        return Decision(
            Verdict.REFUTED,
            f"anytime-valid upper bound {est.ci_high:+.4%} is a regression beyond the MDE",
            est,
            stats,
        )

    if est.ci_low >= -policy.rope and est.ci_high <= policy.rope:
        return Decision(
            Verdict.INCONCLUSIVE,
            f"interval [{est.ci_low:+.4%}, {est.ci_high:+.4%}] lies inside the "
            f"±{policy.rope:.2%} region of practical equivalence: no real effect",
            est,
            stats,
        )

    # SPRT accelerator on the per-replicate gains, measured against the baseline
    # mean. Two guards, both learned the hard way:
    #   * the baseline mean is itself an estimate, so inflate sigma by its
    #     uncertainty — otherwise a fluke-high baseline manufactures a winner;
    #   * the SPRT may only accelerate a decision the anytime-valid interval has
    #     not contradicted. It speeds decisions up; it never overrules them.
    if est.sigma > 0:
        gains = [
            ((est.baseline_mean - v) if direction == "minimize" else (v - est.baseline_mean))
            / (abs(est.baseline_mean) if abs(est.baseline_mean) > 1e-12 else 1.0)
            for v in treatment
        ]
        sigma_eff = est.sigma * math.sqrt(1.0 + est.n_treatment / max(est.n_baseline, 1))
        sprt = sprt_decision(gains, delta=policy.mde, sigma=sigma_eff, alpha=policy.alpha)
        stats["sprt"] = {"accept_h1": 1.0, "accept_h0": -1.0, "continue": 0.0}[sprt]
        if sprt == "accept_h1" and est.ci_low > 0:
            return Decision(
                Verdict.SUPPORTED,
                f"SPRT crossed the H1 boundary with the interval clear of zero "
                f"({est.ci_low:+.4%} lower bound)",
                est, stats,
            )
        if sprt == "accept_h0" and est.ci_high < policy.mde and est.n_treatment >= policy.min_replicates * 2:
            return Decision(
                Verdict.INCONCLUSIVE, "SPRT crossed the H0 boundary: effect indistinguishable from zero",
                est, stats,
            )

    if est.n_treatment >= policy.max_replicates:
        if est.ci_high < policy.mde:
            return Decision(
                Verdict.REFUTED,
                f"replicate budget spent; upper bound {est.ci_high:+.4%} cannot reach the MDE",
                est,
                stats,
            )
        return Decision(
            Verdict.INCONCLUSIVE,
            f"replicate budget spent with interval [{est.ci_low:+.4%}, {est.ci_high:+.4%}] "
            "still straddling the decision threshold",
            est,
            stats,
        )

    return Decision(
        Verdict.TESTING,
        f"effect {est.effect:+.4%}, interval [{est.ci_low:+.4%}, {est.ci_high:+.4%}] "
        "not yet decisive",
        est,
        stats,
    )


def required_replicates(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Classical per-arm sample size for detecting `mde` at the given power.

    Used to tell the human up front how much compute a question will cost,
    rather than discovering it 200 trials later.
    """
    if mde <= 0 or sigma <= 0:
        return 0
    z_a = norm_ppf(1.0 - alpha / 2.0)
    z_b = norm_ppf(power)
    return int(math.ceil(2.0 * ((z_a + z_b) * sigma / mde) ** 2))


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    """(mean, stdev) with a sane answer for n < 2."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)
