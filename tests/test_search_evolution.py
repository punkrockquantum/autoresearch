"""Allocation, determinism, and the capability memory that evolves with use."""

import random
import unittest

from arp.config import SearchPolicy
from arp.evolution import (
    CapabilityMemory,
    choose_operator,
    mutate,
    operator_catalog,
    param_groups,
    similarity,
)
from arp.models import Hypothesis, Subject, Verdict
from arp.search import ArmStats, ThompsonAllocator, derive_seed
from arp.store import Store


def make_subject(slug="s", **kwargs) -> Subject:
    config = {
        "param_space": {
            "lr": {"type": "float", "default": 0.01, "low": 0.001, "high": 1.0, "group": "lr", "step": 0.3},
            "depth": {"type": "int", "default": 8, "low": 2, "high": 32, "step": 2, "group": "architecture"},
            "act": {"type": "choice", "default": "gelu", "values": ["gelu", "relu"], "group": "architecture"},
        },
        "baseline_params": {"lr": 0.01, "depth": 8, "act": "gelu"},
    }
    config.update(kwargs.pop("config", {}))
    return Subject(slug=slug, title=kwargs.pop("title", "Subject about learning rates"),
                   metric=kwargs.pop("metric", "loss"), config=config, **kwargs)


class TestSeeds(unittest.TestCase):
    def test_seeds_are_pure_functions_of_their_inputs(self):
        self.assertEqual(derive_seed("a", "b", 1), derive_seed("a", "b", 1))
        self.assertNotEqual(derive_seed("a", "b", 1), derive_seed("a", "b", 2))
        self.assertTrue(0 <= derive_seed("x") < 2 ** 31)


class TestArmStats(unittest.TestCase):
    def test_running_moments_match_the_batch_calculation(self):
        values = [0.01, -0.02, 0.03, 0.005]
        stats = ArmStats.from_values(values)
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        self.assertEqual(stats.n, 4)
        self.assertAlmostEqual(stats.mean, mean, places=12)
        self.assertAlmostEqual(stats.variance, var, places=12)


class TestAllocator(unittest.TestCase):
    def setUp(self):
        self.subject = make_subject()
        self.hypotheses = [
            Hypothesis(subject_id=self.subject.id, title="a", operator="tune:lr", params={"lr": 0.02}),
            Hypothesis(subject_id=self.subject.id, title="b", operator="tune:architecture",
                       params={"depth": 10}),
        ]

    def test_measures_the_baseline_before_comparing_anything(self):
        alloc = ThompsonAllocator(SearchPolicy(), salt="t").allocate(
            self.subject, self.hypotheses, {}, baseline_n=0
        )
        self.assertEqual(alloc.arm, "baseline")

    def test_keeps_the_baseline_from_starving(self):
        alloc = ThompsonAllocator(SearchPolicy(), salt="t").allocate(
            self.subject, self.hypotheses, {}, baseline_n=3, treatment_n=40, step=3
        )
        self.assertEqual(alloc.arm, "baseline")

    def test_same_salt_gives_the_same_sequence(self):
        def sequence():
            allocator = ThompsonAllocator(SearchPolicy(), salt="fixed")
            stats = {h.id: ArmStats.from_values([0.01, 0.02]) for h in self.hypotheses}
            picks = []
            for i in range(1, 15):
                alloc = allocator.allocate(self.subject, self.hypotheses, stats, baseline_n=5,
                                           treatment_n=5, step=i)
                picks.append(alloc.hypothesis.id if alloc.hypothesis else "baseline")
            return picks

        self.assertEqual(sequence(), sequence())

    def test_prefers_the_arm_with_better_evidence(self):
        allocator = ThompsonAllocator(SearchPolicy(exploration_floor=0.0), salt="t")
        stats = {
            self.hypotheses[0].id: ArmStats.from_values([0.05] * 12),
            self.hypotheses[1].id: ArmStats.from_values([-0.05] * 12),
        }
        picks = [
            allocator.allocate(self.subject, self.hypotheses, stats, baseline_n=10,
                               treatment_n=10, step=i).hypothesis.id
            for i in range(1, 40) if i % 12
        ]
        winner = self.hypotheses[0].id
        self.assertGreater(picks.count(winner) / len(picks), 0.8)


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.subject = make_subject()
        self.rng = random.Random(0)

    def test_groups_and_catalog_come_from_the_param_space(self):
        self.assertEqual(set(param_groups(self.subject)), {"lr", "architecture"})
        catalog = operator_catalog(self.subject)
        self.assertIn("tune:lr", catalog)
        self.assertIn("tune:architecture", catalog)
        self.assertIn("combine", catalog)

    def test_float_mutation_stays_inside_bounds(self):
        for _ in range(200):
            params, _ = mutate(self.subject, "tune:lr", self.subject.baseline_params, self.rng, 3.0)
            self.assertGreaterEqual(params["lr"], 0.001)
            self.assertLessEqual(params["lr"], 1.0)

    def test_int_and_choice_mutations_are_valid(self):
        seen_int, seen_choice = set(), set()
        for _ in range(200):
            params, _ = mutate(self.subject, "tune:architecture", self.subject.baseline_params, self.rng)
            seen_int.add(params["depth"])
            seen_choice.add(params["act"])
        self.assertTrue(all(2 <= d <= 32 for d in seen_int))
        self.assertTrue(seen_choice <= {"gelu", "relu"})

    def test_revert_restores_every_default(self):
        params, note = mutate(self.subject, "revert", {"lr": 0.5, "depth": 20, "act": "relu"}, self.rng)
        self.assertEqual(params, {"lr": 0.01, "depth": 8, "act": "gelu"})
        self.assertIn("default", note)

    def test_simplify_resets_the_furthest_parameter(self):
        params, _ = mutate(self.subject, "simplify", {"lr": 0.01, "depth": 30, "act": "gelu"}, self.rng)
        self.assertEqual(params["depth"], 8)

    def test_direction_hints_bias_which_way_a_knob_moves(self):
        ups = downs = 0
        for seed in range(200):
            params, _ = mutate(self.subject, "tune:lr", {"lr": 0.01}, random.Random(seed),
                               direction_hints={"lr": 2.0})
            if params["lr"] > 0.01:
                ups += 1
            else:
                downs += 1
        self.assertGreater(ups, downs * 3)
        self.assertGreater(downs, 0, "a hint must bias the direction, never lock it in")

    def test_no_hint_is_an_even_coin(self):
        ups = sum(
            1 for seed in range(300)
            if mutate(self.subject, "tune:lr", {"lr": 0.01}, random.Random(seed))[0]["lr"] > 0.01
        )
        self.assertGreater(ups, 100)
        self.assertLess(ups, 200)

    def test_boldness_widens_the_step(self):
        rng_a, rng_b = random.Random(5), random.Random(5)
        timid, _ = mutate(self.subject, "tune:lr", self.subject.baseline_params, rng_a, 1.0)
        bold, _ = mutate(self.subject, "tune:lr", self.subject.baseline_params, rng_b, 3.0)
        self.assertGreater(abs(bold["lr"] - 0.01), abs(timid["lr"] - 0.01))


class TestCapabilityMemory(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(make_subject())
        self.memory = CapabilityMemory(self.store)

    def tearDown(self):
        self.store.close()

    def test_priors_move_with_outcomes(self):
        self.assertEqual(self.memory.prior_for(self.subject, "tune:lr"), 0.0)
        self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.02)
        self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.01)
        self.memory.record_verdict(self.subject, "tune:architecture", Verdict.REFUTED, -0.03)
        self.assertGreater(self.memory.prior_for(self.subject, "tune:lr"), 0.0)
        self.assertLess(self.memory.prior_for(self.subject, "tune:architecture"), 0.0)
        self.assertGreater(self.memory.success_rate(self.subject, "tune:lr"),
                           self.memory.success_rate(self.subject, "tune:architecture"))

    def test_operator_choice_follows_what_worked(self):
        for _ in range(12):
            self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.02)
            self.memory.record_verdict(self.subject, "tune:architecture", Verdict.REFUTED, -0.02)
        rng = random.Random(3)
        picks = [choose_operator(self.memory, self.subject, rng) for _ in range(60)]
        self.assertGreater(picks.count("tune:lr"), picks.count("tune:architecture"))

    def test_similarity_recognises_related_subjects(self):
        twin = make_subject("s2", title="Another subject about learning rates")
        stranger = Subject(slug="s3", title="Espresso grind size and extraction yield",
                           metric="yield", direction="maximize")
        self.assertGreater(similarity(self.subject, twin), similarity(self.subject, stranger))

    def test_new_subjects_inherit_from_similar_ones(self):
        """The 'evolves ability' claim, made concrete: experience must transfer.

        Transfer happens live through `prior_for` even before seeding; seeding
        writes it into the new subject's own capability row, which is what the
        operator sampler reads.
        """
        for _ in range(8):
            self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.03)
            self.memory.record_verdict(self.subject, "tune:architecture", Verdict.REFUTED, -0.02)
        newcomer = self.store.add_subject(make_subject("s-new", title="Subject about learning rates again"))

        self.assertIsNone(self.store.get_capability("tune:lr", newcomer.id))
        self.assertGreater(self.memory.prior_for(newcomer, "tune:lr"), 0.0)

        self.assertGreater(self.memory.seed_from_similar(newcomer), 0)
        inherited = self.store.get_capability("tune:lr", newcomer.id)
        self.assertIsNotNone(inherited)
        self.assertGreater(inherited.alpha, 1.0)
        self.assertGreater(inherited.mean_reward, 0.0)
        # Inherited evidence is discounted: it never counts as first-hand.
        self.assertLess(inherited.n, 8)

        rng = random.Random(11)
        picks = [choose_operator(self.memory, newcomer, rng) for _ in range(60)]
        self.assertGreater(picks.count("tune:lr"), picks.count("tune:architecture"))

    def test_unrelated_subjects_do_not_inherit(self):
        for _ in range(8):
            self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.03)
        unrelated = self.store.add_subject(
            Subject(slug="coffee", title="Espresso grind size and extraction yield",
                    metric="yield", direction="maximize",
                    config={"param_space": {"grind": {"type": "int", "default": 20, "group": "grind"}}})
        )
        self.assertEqual(self.memory.seed_from_similar(unrelated), 0)

    def test_skill_profile_summarises_the_subject(self):
        self.memory.record_verdict(self.subject, "tune:lr", Verdict.PROVEN, 0.02)
        profile = self.memory.skill_profile(self.subject)
        self.assertEqual(profile.subject_id, self.subject.id)
        self.assertIn("tune:lr", profile.top_operators())
        self.assertGreaterEqual(profile.level, 0.0)


if __name__ == "__main__":
    unittest.main()
