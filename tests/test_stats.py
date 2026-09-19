"""The decision layer is the part that must not be wrong."""

import random
import unittest

from arp.config import DecisionPolicy
from arp.models import Verdict
from arp.stats import (
    cs_radius,
    decide,
    estimate_effect,
    holm_bonferroni,
    regularized_incomplete_beta,
    required_replicates,
    sprt_decision,
    student_t_two_sided_p,
    welch_ttest,
)


class TestDistributions(unittest.TestCase):
    def test_incomplete_beta_endpoints(self):
        self.assertAlmostEqual(regularized_incomplete_beta(2, 3, 0.0), 0.0)
        self.assertAlmostEqual(regularized_incomplete_beta(2, 3, 1.0), 1.0)

    def test_incomplete_beta_symmetry(self):
        # I_x(a,b) == 1 - I_{1-x}(b,a)
        for a, b, x in ((2.0, 3.0, 0.3), (0.5, 4.0, 0.75), (5.0, 5.0, 0.5)):
            self.assertAlmostEqual(
                regularized_incomplete_beta(a, b, x),
                1.0 - regularized_incomplete_beta(b, a, 1.0 - x),
                places=10,
            )

    def test_t_distribution_against_known_values(self):
        # Two-sided p for t=2.228 with df=10 is ~0.05 (textbook critical value).
        self.assertAlmostEqual(student_t_two_sided_p(2.228, 10), 0.05, places=3)
        # Large df approaches the normal: t=1.96 -> 0.05.
        self.assertAlmostEqual(student_t_two_sided_p(1.96, 100000), 0.05, places=3)
        self.assertAlmostEqual(student_t_two_sided_p(0.0, 5), 1.0, places=6)


class TestWelch(unittest.TestCase):
    def test_identical_samples_are_not_significant(self):
        sample = [1.0, 1.1, 0.9, 1.05, 0.95]
        result = welch_ttest(sample, sample)
        self.assertAlmostEqual(result.diff, 0.0)
        self.assertGreater(result.p_value, 0.9)

    def test_separated_samples_are_significant(self):
        a = [10.0, 10.1, 9.9, 10.2, 9.8]
        b = [1.0, 1.1, 0.9, 1.2, 0.8]
        result = welch_ttest(a, b)
        self.assertLess(result.p_value, 1e-6)
        self.assertAlmostEqual(result.diff, 9.0, places=6)

    def test_too_few_samples_is_inconclusive_not_a_crash(self):
        result = welch_ttest([1.0], [2.0])
        self.assertEqual(result.p_value, 1.0)


class TestConfidenceSequence(unittest.TestCase):
    def test_radius_shrinks_with_more_data(self):
        radii = [cs_radius(n, sigma=1.0, alpha=0.05) for n in (5, 20, 100, 1000)]
        self.assertTrue(all(a > b for a, b in zip(radii, radii[1:])), radii)

    def test_radius_grows_with_noise_and_confidence(self):
        self.assertGreater(cs_radius(20, 2.0, 0.05), cs_radius(20, 1.0, 0.05))
        self.assertGreater(cs_radius(20, 1.0, 0.001), cs_radius(20, 1.0, 0.05))

    def test_coverage_under_continuous_peeking(self):
        """The whole point of an anytime-valid bound: peeking must not break it.

        Over many simulated streams, the fraction where the running interval ever
        excludes the true mean must stay under alpha — checked with the known
        variance proxy the boundary assumes.
        """
        rng = random.Random(20260917)
        alpha, misses, runs = 0.05, 0, 300
        for _ in range(runs):
            total, n = 0.0, 0
            for _ in range(80):
                total += rng.gauss(0.0, 1.0)
                n += 1
                if abs(total / n) > cs_radius(n, 1.0, alpha):
                    misses += 1
                    break
        self.assertLess(misses / runs, alpha)

    def test_two_sample_interval_covers_a_null_effect(self):
        """Same guarantee, but through the estimator the platform actually uses.

        Sigma is estimated from a handful of runs here, which is why
        `estimate_effect` inflates the radius for small samples.
        """
        rng = random.Random(4242)
        alpha, misses, runs = 0.05, 0, 200
        for _ in range(runs):
            baseline, treatment = [], []
            for _ in range(20):
                baseline.append(1.0 + rng.gauss(0, 0.01))
                treatment.append(1.0 + rng.gauss(0, 0.01))
                if len(treatment) < 3:
                    continue
                est = estimate_effect(baseline, treatment, "minimize", alpha)
                if est.ci_low > 0.0 or est.ci_high < 0.0:
                    misses += 1
                    break
        self.assertLess(misses / runs, alpha)


class TestSPRT(unittest.TestCase):
    def test_accepts_h1_on_a_clear_effect(self):
        values = [0.05] * 30
        self.assertEqual(sprt_decision(values, delta=0.02, sigma=0.01, alpha=0.05), "accept_h1")

    def test_accepts_h0_on_pure_noise(self):
        rng = random.Random(7)
        values = [rng.gauss(0.0, 0.01) for _ in range(200)]
        self.assertEqual(sprt_decision(values, delta=0.02, sigma=0.01, alpha=0.05), "accept_h0")

    def test_continues_when_undecided(self):
        self.assertEqual(sprt_decision([0.01], delta=0.02, sigma=0.05, alpha=0.05), "continue")


class TestHolm(unittest.TestCase):
    def test_step_down_rejects_only_the_small_ones(self):
        mask = holm_bonferroni([0.001, 0.04, 0.6], alpha=0.05)
        self.assertEqual(mask, [True, False, False])

    def test_single_test_is_plain_alpha(self):
        self.assertEqual(holm_bonferroni([0.04], alpha=0.05), [True])
        self.assertEqual(holm_bonferroni([0.06], alpha=0.05), [False])


class TestEffect(unittest.TestCase):
    def test_direction_is_applied(self):
        baseline, treatment = [1.0] * 5, [0.9] * 5
        down = estimate_effect(baseline, treatment, "minimize")
        up = estimate_effect(baseline, treatment, "maximize")
        self.assertAlmostEqual(down.effect, 0.1, places=6)
        self.assertAlmostEqual(up.effect, -0.1, places=6)

    def test_empty_arms_do_not_crash(self):
        self.assertEqual(estimate_effect([], [], "minimize").effect, 0.0)


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy()

    def test_holds_out_until_the_replicate_floor(self):
        decision = decide([1.0, 1.0], [0.5, 0.5], "minimize", self.policy)
        self.assertEqual(decision.status, Verdict.TESTING)
        self.assertIn("replicates", decision.reason)

    def test_supports_a_large_real_effect(self):
        rng = random.Random(1)
        baseline = [1.0 + rng.gauss(0, 0.002) for _ in range(8)]
        treatment = [0.95 + rng.gauss(0, 0.002) for _ in range(8)]
        decision = decide(baseline, treatment, "minimize", self.policy)
        self.assertEqual(decision.status, Verdict.SUPPORTED)
        self.assertGreater(decision.estimate.ci_low, self.policy.mde)

    def test_refutes_a_regression(self):
        rng = random.Random(2)
        baseline = [1.0 + rng.gauss(0, 0.002) for _ in range(8)]
        treatment = [1.05 + rng.gauss(0, 0.002) for _ in range(8)]
        self.assertEqual(decide(baseline, treatment, "minimize", self.policy).status, Verdict.REFUTED)

    def test_null_effect_never_gets_supported(self):
        """The failure mode that matters: noise must not become a discovery."""
        rng = random.Random(20260917)
        supported = 0
        for _ in range(120):
            baseline = [1.0 + rng.gauss(0, 0.01) for _ in range(12)]
            treatment = [1.0 + rng.gauss(0, 0.01) for _ in range(12)]
            if decide(baseline, treatment, "minimize", self.policy).status == Verdict.SUPPORTED:
                supported += 1
        self.assertLessEqual(supported / 120, self.policy.alpha)

    def test_challenge_tightens_the_bar(self):
        rng = random.Random(3)
        baseline = [1.0 + rng.gauss(0, 0.002) for _ in range(6)]
        treatment = [0.985 + rng.gauss(0, 0.002) for _ in range(6)]
        relaxed = decide(baseline, treatment, "minimize", self.policy, open_challenges=0)
        strict = decide(baseline, treatment, "minimize", self.policy, open_challenges=2)
        self.assertEqual(relaxed.status, Verdict.SUPPORTED)
        self.assertEqual(strict.status, Verdict.TESTING)
        self.assertLess(strict.estimate.ci_low, relaxed.estimate.ci_low)

    def test_decision_is_deterministic(self):
        baseline = [1.0, 1.01, 0.99, 1.02, 0.98]
        treatment = [0.97, 0.96, 0.98, 0.95, 0.97]
        first = decide(baseline, treatment, "minimize", self.policy)
        second = decide(baseline, treatment, "minimize", self.policy)
        self.assertEqual((first.status, first.reason), (second.status, second.reason))


class TestPowerHelper(unittest.TestCase):
    def test_required_replicates_scales_with_noise(self):
        self.assertGreater(required_replicates(0.02, 0.005), required_replicates(0.01, 0.005))
        self.assertEqual(required_replicates(0.0, 0.005), 0)


if __name__ == "__main__":
    unittest.main()
