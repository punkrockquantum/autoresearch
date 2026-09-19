"""The exhaustive stage: what it takes to move from 'supported' to 'proven'."""

import random
import unittest

from arp.config import DecisionPolicy
from arp.exhaustive import (
    cell_key,
    cell_overrides,
    evaluate_confirmation,
    grid_cells,
    plan_confirmation,
)
from arp.models import Hypothesis, Subject, Trial, Verdict
from arp.store import Store

POLICY = DecisionPolicy(min_replicates=3, alpha_confirm=0.05, mde=0.002)


def make_subject() -> Subject:
    return Subject(
        slug="grid", title="Grid subject", metric="loss", direction="minimize",
        config={
            "param_space": {"lr": {"type": "float", "default": 0.01, "group": "lr"}},
            "baseline_params": {"lr": 0.01},
            "stress_axes": {"scale": {"small": {"lr": 0.005}, "large": {}}},
        },
    )


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.subject = make_subject()

    def test_cells_are_the_cartesian_product_in_a_stable_order(self):
        cells = grid_cells(self.subject, {"data": ["a", "b"]})
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells, grid_cells(self.subject, {"data": ["a", "b"]}))
        self.assertIn({"scale": "small", "data": "a"}, cells)

    def test_no_axes_means_one_cell(self):
        bare = Subject(slug="bare", title="t", metric="m")
        self.assertEqual(grid_cells(bare), [{}])

    def test_axis_labels_can_carry_parameter_overrides(self):
        self.assertEqual(cell_overrides(self.subject, {"scale": "small"}), {"lr": 0.005})
        self.assertEqual(cell_overrides(self.subject, {"scale": "large"}), {})

    def test_cell_key_is_order_insensitive(self):
        self.assertEqual(cell_key({"a": 1, "b": 2}), cell_key({"b": 2, "a": 1}))


class TestConfirmation(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(make_subject())
        self.hyp = self.store.add_hypothesis(
            Hypothesis(subject_id=self.subject.id, title="lr up", operator="tune:lr", params={"lr": 0.02})
        )

    def tearDown(self):
        self.store.close()

    def fill(self, cell, baseline_values, treatment_values):
        for value in baseline_values:
            self.store.add_trial(Trial(subject_id=self.subject.id, hypothesis_id=None, arm="baseline",
                                       seed=0, metric_value=value, stage="confirm", cell=cell))
        for value in treatment_values:
            self.store.add_trial(Trial(subject_id=self.subject.id, hypothesis_id=self.hyp.id,
                                       arm="treatment", seed=0, metric_value=value, stage="confirm",
                                       cell=cell))

    def evaluate(self, open_challenges=0):
        return evaluate_confirmation(self.store, self.subject, self.hyp, POLICY, "salt",
                                     open_challenges=open_challenges)

    def test_plan_covers_every_cell_and_arm(self):
        pending = plan_confirmation(self.store, self.subject, self.hyp, POLICY, "salt")
        self.assertEqual(len(pending), 2 * 2 * POLICY.min_replicates)
        self.assertEqual(len({r.seed for r in pending}), len(pending))
        # Round-robin: the first pass touches every cell and arm once.
        first_pass = pending[:4]
        self.assertEqual(len({(cell_key(r.cell), r.arm) for r in first_pass}), 4)

    def test_plan_shrinks_as_trials_land(self):
        before = len(plan_confirmation(self.store, self.subject, self.hyp, POLICY, "salt"))
        self.fill({"scale": "small"}, [1.0], [])
        after = len(plan_confirmation(self.store, self.subject, self.hyp, POLICY, "salt"))
        self.assertEqual(after, before - 1)

    def test_incomplete_grid_stays_supported(self):
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999], [0.9, 0.901, 0.899])
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.SUPPORTED)
        self.assertTrue(result.pending)

    def test_a_win_everywhere_is_proven(self):
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999, 1.0], [0.9, 0.901, 0.899, 0.9])
        self.fill({"scale": "large"}, [2.0, 2.002, 1.998, 2.0], [1.8, 1.802, 1.798, 1.8])
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.PROVEN)
        self.assertTrue(all(c.passed for c in result.cells))
        self.assertGreater(result.pooled.effect, 0.05)

    def test_a_regression_in_one_cell_refutes_it(self):
        """The reason the stage exists: 'works at one scale' is not a result."""
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999, 1.0], [0.9, 0.901, 0.899, 0.9])
        self.fill({"scale": "large"}, [2.0, 2.002, 1.998, 2.0], [2.2, 2.202, 2.198, 2.2])
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.REFUTED)
        self.assertTrue(any(c.regressed for c in result.cells))

    def test_an_unresolved_cell_buys_more_evidence_instead_of_being_written_off(self):
        """'We can't tell yet' must mean more replicates, not a refutation."""
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999, 1.0], [1.0, 0.9995, 1.0005, 1.0])
        self.fill({"scale": "large"}, [2.0, 2.001, 1.999, 2.0], [2.0, 1.9995, 2.0005, 2.0])
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.SUPPORTED)
        self.assertTrue(result.pending)
        self.assertFalse(any(c.regressed for c in result.cells))

    def test_a_null_effect_is_refuted_once_the_budget_is_spent(self):
        rng = random.Random(99)
        for cell, level in (({"scale": "small"}, 1.0), ({"scale": "large"}, 2.0)):
            self.fill(
                cell,
                [level + rng.gauss(0, 0.0005 * level) for _ in range(POLICY.max_replicates)],
                [level + rng.gauss(0, 0.0005 * level) for _ in range(POLICY.max_replicates)],
            )
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.REFUTED)
        self.assertFalse(result.pending)
        self.assertIn("MDE", result.reason)

    def test_open_challenge_blocks_proof_and_raises_the_bar(self):
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999, 1.0], [0.9, 0.901, 0.899, 0.9])
        self.fill({"scale": "large"}, [2.0, 2.002, 1.998, 2.0], [1.8, 1.802, 1.798, 1.8])
        self.assertEqual(self.evaluate().status, Verdict.PROVEN)
        contested = self.evaluate(open_challenges=1)
        self.assertEqual(contested.status, Verdict.CONTESTED)
        self.assertTrue(contested.pending, "a challenge must buy more evidence, not just a label")

    def test_stratified_pooling_ignores_level_differences_between_cells(self):
        """Cells sit at different metric levels; that spread is not treatment noise."""
        self.fill({"scale": "small"}, [1.0, 1.0, 1.0, 1.0], [0.95, 0.95, 0.95, 0.95])
        self.fill({"scale": "large"}, [10.0, 10.0, 10.0, 10.0], [9.5, 9.5, 9.5, 9.5])
        result = self.evaluate()
        self.assertEqual(result.status, Verdict.PROVEN)
        self.assertAlmostEqual(result.pooled.effect, 0.05, places=6)

    def test_evaluation_is_deterministic(self):
        self.fill({"scale": "small"}, [1.0, 1.001, 0.999, 1.0], [0.9, 0.901, 0.899, 0.9])
        self.fill({"scale": "large"}, [2.0, 2.002, 1.998, 2.0], [1.8, 1.802, 1.798, 1.8])
        first, second = self.evaluate(), self.evaluate()
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.reason, second.reason)


if __name__ == "__main__":
    unittest.main()
