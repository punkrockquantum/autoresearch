"""End-to-end behaviour against a simulated response surface with a known truth.

These are the tests that say whether the platform works: given a knob that
genuinely helps and a knob that does nothing, does it prove the first, refuse to
prove the second, and survive a human who refuses to believe either?
"""

import unittest

from arp.config import DecisionPolicy, PlatformConfig, SearchPolicy
from arp.interventions import apply_prompt, open_challenge, unadopt
from arp.models import Subject, Verdict
from arp.orchestrator import Orchestrator
from arp.store import Store


def simulated_subject(slug: str = "sim") -> Subject:
    """`lr` really helps; `momentum` does nothing at all."""
    return Subject(
        slug=slug,
        title="Simulated optimiser tuning",
        metric="loss",
        direction="minimize",
        runner="simulated",
        description="Synthetic subject with a known response surface.",
        config={
            "param_space": {
                "lr": {"type": "float", "default": 0.01, "low": 0.001, "high": 0.5,
                       "log": True, "group": "lr", "step": 0.35},
                "momentum": {"type": "float", "default": 0.9, "low": 0.5, "high": 0.99,
                             "log": False, "scale": 0.1, "group": "momentum", "step": 0.2},
            },
            "baseline_params": {"lr": 0.01, "momentum": 0.9},
            "stress_axes": {"scale": {"nominal": {}, "larger": {}}},
            "simulation": {
                "baseline": 1.0,
                "noise": 0.002,
                "coefs": {"lr": -0.04, "momentum": 0.0},
                "curvature": {"lr": 0.02},
            },
            "value_model": {"reference_effect": 0.01, "effort_days": 2},
        },
    )


def config_for(tmp_dir: str = ".") -> PlatformConfig:
    return PlatformConfig(
        db_path=":memory:",
        work_dir=tmp_dir,
        decision=DecisionPolicy(min_replicates=3, max_replicates=16),
        search=SearchPolicy(max_open_hypotheses=6),
    )


class TestOrchestratorLoop(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(simulated_subject())
        self.config = config_for()
        self.orch = Orchestrator(self.store, self.subject, self.config, salt="test", search_web=False)

    def tearDown(self):
        self.store.close()

    def test_a_short_run_records_trials_and_findings(self):
        summary = self.orch.run(steps=12)
        self.assertEqual(len(summary.steps), 12)
        self.assertGreater(self.store.count_trials(self.subject.id), 0)
        self.assertTrue(self.store.list_hypotheses(self.subject.id))
        self.assertTrue(all(s.action in ("screen", "confirm", "baseline") for s in summary.steps))

    def test_the_baseline_is_measured_before_any_comparison(self):
        self.orch.run(steps=3)
        baseline = self.store.list_trials(subject_id=self.subject.id, arm="baseline")
        self.assertGreaterEqual(len(baseline), self.config.search.min_baseline)

    def test_it_proves_a_real_effect_and_adopts_it(self):
        self.orch.run(steps=140)
        proven = self.store.list_findings(self.subject.id, [Verdict.PROVEN])
        self.assertTrue(proven, "a 4%-per-log-unit effect should be findable in 140 trials")
        for finding in proven:
            self.assertGreater(finding.effect, 0.0)
            self.assertGreater(finding.ci_low, 0.0)
            self.assertIsNotNone(finding.proven_at)
            hypothesis = self.store.get_hypothesis(finding.hypothesis_id)
            self.assertIn("lr", hypothesis.params)

        subject = self.store.get_subject(self.subject.slug)
        self.assertGreater(subject.config["epoch"], 0)
        self.assertGreater(subject.baseline_params["lr"], 0.01, "a proven win must move the baseline")
        self.assertTrue(subject.config["adopted"])

    def test_it_never_proves_a_knob_that_does_nothing(self):
        """The expensive failure mode: noise dressed up as a discovery."""
        self.orch.run(steps=140)
        for finding in self.store.list_findings(self.subject.id, [Verdict.PROVEN]):
            hypothesis = self.store.get_hypothesis(finding.hypothesis_id)
            self.assertNotIn(
                "momentum", hypothesis.params,
                f"momentum has no effect in the simulation but was proven: {finding.reason}",
            )

    def test_it_learns_which_way_to_turn_a_knob(self):
        """After a result, the next proposal should not be a coin flip."""
        self.orch.run(steps=60)
        hints = self.orch.proposer.direction_hints(self.store.get_subject("sim"))
        self.assertIn("lr", hints)
        # Turning lr *up* is what helps in this simulation.
        self.assertGreater(hints["lr"], 0.0)

    def test_hypotheses_are_stored_as_deltas_not_whole_configurations(self):
        self.orch.run(steps=40)
        for hypothesis in self.store.list_hypotheses(self.subject.id):
            if hypothesis.operator.startswith("tune:"):
                self.assertLessEqual(len(hypothesis.params), 1, hypothesis.title)

    def test_a_run_replays_identically_from_its_salt(self):
        def trace():
            store = Store(":memory:")
            subject = store.add_subject(simulated_subject())
            orch = Orchestrator(store, subject, config_for(), salt="fixed-salt", search_web=False)
            summary = orch.run(steps=45)
            out = [(s.action, s.arm, round(s.metric, 9) if s.metric is not None else None)
                   for s in summary.steps]
            store.close()
            return out

        self.assertEqual(trace(), trace())

    def test_a_different_salt_explores_differently(self):
        def trace(salt):
            store = Store(":memory:")
            subject = store.add_subject(simulated_subject())
            orch = Orchestrator(store, subject, config_for(), salt=salt, search_web=False)
            summary = orch.run(steps=45)
            out = [s.metric for s in summary.steps]
            store.close()
            return out

        self.assertNotEqual(trace("salt-a"), trace("salt-b"))

    def test_state_reports_what_is_going_on(self):
        self.orch.run(steps=20)
        state = self.orch.state()
        self.assertEqual(state["subject"], "sim")
        self.assertEqual(state["salt"], "test")
        self.assertGreater(state["trials"], 0)
        self.assertIn("findings", state)
        self.assertGreaterEqual(state["skill_level"], 0.0)

    def test_it_resumes_from_the_stored_state(self):
        self.orch.run(steps=20)
        trials_before = self.store.count_trials(self.subject.id)
        resumed = Orchestrator(self.store, self.store.get_subject("sim"), self.config,
                               salt="test", search_web=False)
        resumed.run(steps=10)
        self.assertGreater(self.store.count_trials(self.subject.id), trials_before)

    def test_a_time_budget_stops_the_loop(self):
        summary = self.orch.run(steps=10_000, budget_seconds=0.4)
        self.assertLess(len(summary.steps), 10_000)
        self.assertLessEqual(summary.elapsed_s, 5.0)


class TestCrashHandling(unittest.TestCase):
    def test_a_hypothesis_that_always_crashes_is_refuted_not_retried_forever(self):
        store = Store(":memory:")
        subject = simulated_subject("crashy")
        # The simulated runner "crashes" on out-of-range parameters, like an OOM.
        subject.config["param_space"]["lr"]["high"] = 0.011
        subject = store.add_subject(subject)
        orch = Orchestrator(store, subject, config_for(), salt="crash", search_web=False)
        from arp.models import Hypothesis

        hypothesis = store.add_hypothesis(
            Hypothesis(subject_id=subject.id, title="way too high", operator="tune:lr",
                       params={"lr": 5.0})
        )
        for _ in range(12):
            orch.step()
        refreshed = store.get_hypothesis(hypothesis.id)
        finding = store.finding_for_hypothesis(hypothesis.id)
        self.assertEqual(refreshed.status, Verdict.REFUTED)
        self.assertIn("runner failed", finding.reason)
        failures = [t for t in store.list_trials(subject_id=subject.id, ok_only=False) if not t.ok]
        self.assertLessEqual(len(failures), 4, "a broken idea must not soak up the whole budget")
        store.close()


class TestHumanInTheLoop(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(simulated_subject())
        self.config = config_for()
        self.orch = Orchestrator(self.store, self.subject, self.config, salt="hitl", search_web=False)

    def tearDown(self):
        self.store.close()

    def test_a_prompt_becomes_a_real_hypothesis_that_gets_tested(self):
        outcome = apply_prompt(self.store, self.subject, "please try lr=0.08")
        self.assertEqual(outcome["kind"], "hypothesis")
        hypothesis = self.store.get_hypothesis(outcome["hypothesis_id"])
        self.assertEqual(hypothesis.params, {"lr": 0.08})
        self.assertEqual(hypothesis.origin, "human")

        self.orch.run(steps=25)
        trials = self.store.list_trials(subject_id=self.subject.id, hypothesis_id=hypothesis.id)
        self.assertTrue(trials, "a human's idea must actually be run, not just filed")

    def test_constraints_and_steers_are_kept_for_the_planner(self):
        apply_prompt(self.store, self.subject, "never let peak memory exceed 40GB")
        apply_prompt(self.store, self.subject, "concentrate on the schedule from here")
        subject = self.store.get_subject(self.subject.slug)
        self.assertTrue(subject.config.get("constraints"))
        self.assertTrue(subject.config.get("focus"))
        self.assertEqual(len(self.store.list_prompts(subject.id)), 2)

    def test_a_challenge_reopens_a_proven_finding_under_a_stricter_bar(self):
        self.orch.run(steps=140)
        proven = self.store.list_findings(self.subject.id, [Verdict.PROVEN])
        self.assertTrue(proven)
        finding = proven[0]

        subject = self.store.get_subject(self.subject.slug)
        challenge = open_challenge(self.store, subject, finding, "I don't believe it")
        reopened = self.store.get_finding(finding.id)
        self.assertEqual(reopened.verdict, Verdict.CONTESTED)
        self.assertIsNone(reopened.proven_at)
        self.assertTrue(challenge.axis, "a challenge must widen the grid, not just add a label")
        self.assertEqual(len(self.store.open_challenges(finding.id)), 1)

        # Challenging an adopted change rolls it back out of the baseline, so the
        # re-test compares the change against its absence rather than itself.
        subject = self.store.get_subject(self.subject.slug)
        hypothesis = self.store.get_hypothesis(finding.hypothesis_id)
        for name, value in hypothesis.params.items():
            self.assertNotEqual(subject.baseline_params.get(name), value)

    def test_the_loop_settles_a_challenge_and_closes_it(self):
        self.orch.run(steps=140)
        proven = self.store.list_findings(self.subject.id, [Verdict.PROVEN])
        self.assertTrue(proven)
        finding = proven[0]
        subject = self.store.get_subject(self.subject.slug)
        open_challenge(self.store, subject, finding, "prove it again, properly")

        orch = Orchestrator(self.store, self.store.get_subject(self.subject.slug), self.config,
                            salt="hitl", search_web=False)
        orch.run(steps=120)

        settled = self.store.get_finding(finding.id)
        self.assertIn(settled.verdict, Verdict.TERMINAL,
                      f"a challenge must be settled, not left hanging: {settled.reason}")
        self.assertEqual(self.store.open_challenges(finding.id), [])
        challenge = self.store.list_challenges(finding_id=finding.id)[0]
        self.assertEqual(challenge.status, "resolved")
        self.assertIn(settled.verdict, challenge.resolution)

    def test_unadopt_refuses_to_undo_a_superseding_change(self):
        subject = self.store.get_subject(self.subject.slug)
        subject.config["adopted"] = [
            {"finding_id": "fnd_old", "title": "first lr change", "previous": {"lr": 0.01}},
            {"finding_id": "fnd_new", "title": "second lr change", "previous": {"lr": 0.02}},
        ]
        subject.config["baseline_params"] = {"lr": 0.04, "momentum": 0.9}
        self.store.save_subject(subject)
        from arp.models import Finding

        old = self.store.save_finding(Finding(subject_id=subject.id, hypothesis_id="h",
                                              id="fnd_old"))
        note = unadopt(self.store, subject, old)
        self.assertIn("superseded", note)
        self.assertEqual(self.store.get_subject(subject.slug).baseline_params["lr"], 0.04)


if __name__ == "__main__":
    unittest.main()
