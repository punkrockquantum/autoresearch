"""Probabilistic allocation: which hypothesis gets the next unit of compute.

Thompson sampling over a Normal-Inverse-Gamma posterior on each hypothesis's
per-trial gain. Sampling is driven by a `random.Random` seeded from the run
salt, and every trial's own seed is derived by hashing (run salt, hypothesis,
arm, replicate) — so a whole run replays identically from its salt alone. The
search is random; the record of it is not.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from arp.config import SearchPolicy
from arp.models import Hypothesis, Subject


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from any tuple of values.

    Used instead of `random.randint` so a trial's seed is a pure function of
    what it is a trial *of*: rerunning the same cell reproduces the same run.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def hypothesis_key(hypothesis: Hypothesis) -> str:
    """A stable identity for a hypothesis: what it changes, not which row it is.

    Database ids are random per row, so seeding from them would make two runs of
    the same search irreproducible. Seeding from the change itself means "the
    same experiment" really does get the same seed, in this database or the next.
    """
    params = json.dumps(hypothesis.params, sort_keys=True, default=str)
    return f"{hypothesis.operator}:{params}"


@dataclass
class ArmStats:
    """Observed per-trial gains for one hypothesis (relative units, higher is better)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0          # sum of squared deviations
    last_seen_step: int = -1

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "ArmStats":
        stats = cls()
        for v in values:
            stats.update(v)
        return stats

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


@dataclass
class Allocation:
    hypothesis: Optional[Hypothesis]
    arm: str                       # "baseline" | "treatment"
    seed: int
    stage: str = "screen"
    cell: Dict[str, object] = field(default_factory=dict)
    reason: str = ""
    sampled_value: float = 0.0


class ThompsonAllocator:
    """Samples a plausible gain for every open hypothesis and runs the best one.

    Three things nudge the draw, in this order:
      * the posterior from this hypothesis's own trials (strongest signal),
      * the evolved capability prior for its operator family — what has worked
        on this subject, and on subjects like it,
      * a novelty bonus for operators never tried here, so the frontier keeps
        widening instead of grinding on the same knob.
    """

    def __init__(self, policy: Optional[SearchPolicy] = None, salt: str = "arp"):
        self.policy = policy or SearchPolicy()
        self.salt = salt
        self.rng = random.Random(derive_seed("allocator", salt))

    # -- posterior ---------------------------------------------------------

    def sample_gain(self, stats: ArmStats, prior_mean: float = 0.0, prior_var: float = 1e-4) -> float:
        """Draw one sample of the hypothesis's true mean gain (NIG posterior)."""
        kappa0, alpha0 = 1.0, 2.0
        beta0 = max(prior_var, 1e-12) * alpha0
        n, mean = stats.n, stats.mean
        if n == 0:
            kappa_n, mu_n, alpha_n, beta_n = kappa0, prior_mean, alpha0, beta0
        else:
            kappa_n = kappa0 + n
            mu_n = (kappa0 * prior_mean + n * mean) / kappa_n
            alpha_n = alpha0 + n / 2.0
            beta_n = beta0 + 0.5 * stats.m2 + (kappa0 * n * (mean - prior_mean) ** 2) / (2.0 * kappa_n)
        # sigma^2 ~ InvGamma(alpha_n, beta_n); mu | sigma^2 ~ N(mu_n, sigma^2 / kappa_n)
        gamma = self.rng.gammavariate(alpha_n, 1.0)
        sigma2 = beta_n / gamma if gamma > 0 else beta_n
        return self.rng.gauss(mu_n, max(sigma2 / kappa_n, 1e-18) ** 0.5)

    # -- allocation --------------------------------------------------------

    def allocate(
        self,
        subject: Subject,
        hypotheses: Sequence[Hypothesis],
        stats_by_hyp: Dict[str, ArmStats],
        priors_by_operator: Optional[Dict[str, float]] = None,
        tried_operators: Optional[set] = None,
        step: int = 0,
        baseline_n: int = 0,
        treatment_n: int = 0,
    ) -> Allocation:
        """Pick the next trial to run."""
        priors_by_operator = priors_by_operator or {}
        tried_operators = tried_operators or set()

        # Nothing is comparable until the baseline itself is measured properly —
        # a single lucky baseline run makes every treatment look like a winner.
        # And because the precision of a difference is capped by the *smaller*
        # arm, starving the baseline caps the precision of every comparison at
        # once: hence the ratio rule, not just a periodic refresh. The periodic
        # refresh still earns its place — the baseline drifts (hardware, data,
        # upstream code) and a number from hours ago may no longer be true.
        needs_baseline = (
            baseline_n < self.policy.min_baseline
            or baseline_n < self.policy.baseline_ratio * treatment_n
            or (step > 0 and step % self.policy.baseline_refresh_every == 0)
        )
        if needs_baseline:
            return Allocation(
                hypothesis=None,
                arm="baseline",
                seed=derive_seed(self.salt, subject.slug, "baseline", baseline_n),
                reason="baseline refresh" if baseline_n else "establish baseline",
            )

        if not hypotheses:
            return Allocation(None, "baseline", derive_seed(self.salt, subject.slug, "baseline", baseline_n),
                              reason="no open hypotheses; holding the baseline")

        best: Optional[Allocation] = None
        best_value = -float("inf")
        for hyp in hypotheses:
            stats = stats_by_hyp.get(hyp.id, ArmStats())
            prior = priors_by_operator.get(hyp.operator, 0.0)
            value = self.sample_gain(stats, prior_mean=prior)
            if hyp.operator not in tried_operators and stats.n == 0:
                value += self.policy.novelty_bonus * abs(prior or 0.01)
            if value > best_value:
                best_value = value
                best = Allocation(
                    hypothesis=hyp,
                    arm="treatment",
                    seed=derive_seed(self.salt, subject.slug, hypothesis_key(hyp), stats.n),
                    reason=f"Thompson draw {value:+.4%} (n={stats.n}, operator={hyp.operator})",
                    sampled_value=value,
                )

        # Exploration floor: occasionally ignore the draw and take the least-tested
        # hypothesis, so one early lucky arm cannot starve the rest.
        if self.rng.random() < self.policy.exploration_floor:
            least = min(hypotheses, key=lambda h: stats_by_hyp.get(h.id, ArmStats()).n)
            n = stats_by_hyp.get(least.id, ArmStats()).n
            return Allocation(
                hypothesis=least,
                arm="treatment",
                seed=derive_seed(self.salt, subject.slug, hypothesis_key(least), n),
                reason=f"exploration floor: least-tested hypothesis (n={n})",
            )

        assert best is not None
        return best


def confirmation_seeds(salt: str, hypothesis_id: str, cell_key: str, replicates: int) -> List[int]:
    """Seeds for one cell of the exhaustive grid — reproducible, and distinct per cell."""
    return [derive_seed(salt, hypothesis_id, cell_key, "confirm", i) for i in range(replicates)]
