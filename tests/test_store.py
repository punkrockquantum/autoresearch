"""The store is boring on purpose; these tests keep it that way."""

import os
import tempfile
import unittest

from arp.models import (
    Capability,
    Challenge,
    Finding,
    Hypothesis,
    Lead,
    Prompt,
    Run,
    Subject,
    Trial,
    Verdict,
)
from arp.store import Store


def a_subject(slug: str = "s1") -> Subject:
    return Subject(slug=slug, title="Test subject", metric="loss", direction="minimize",
                   config={"param_space": {"lr": {"type": "float", "default": 0.1, "group": "lr"}}})


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(a_subject())

    def tearDown(self):
        self.store.close()

    def test_subject_round_trips_with_nested_config(self):
        loaded = self.store.get_subject("s1")
        self.assertEqual(loaded.id, self.subject.id)
        self.assertEqual(loaded.param_space["lr"]["default"], 0.1)
        self.assertEqual(self.store.get_subject(self.subject.id).slug, "s1")

    def test_subject_slug_is_unique(self):
        with self.assertRaises(Exception):
            self.store.add_subject(a_subject())

    def test_trials_filter_by_arm_stage_and_success(self):
        hyp = self.store.add_hypothesis(Hypothesis(subject_id=self.subject.id, title="t", operator="tune:lr"))
        self.store.add_trial(Trial(subject_id=self.subject.id, hypothesis_id=None, arm="baseline",
                                   seed=1, metric_value=1.0))
        self.store.add_trial(Trial(subject_id=self.subject.id, hypothesis_id=hyp.id, arm="treatment",
                                   seed=2, metric_value=0.9, stage="confirm"))
        self.store.add_trial(Trial(subject_id=self.subject.id, hypothesis_id=hyp.id, arm="treatment",
                                   seed=3, metric_value=None, ok=False, error="boom"))
        self.assertEqual(len(self.store.list_trials(subject_id=self.subject.id)), 2)
        self.assertEqual(len(self.store.list_trials(subject_id=self.subject.id, ok_only=False)), 3)
        self.assertEqual(len(self.store.list_trials(subject_id=self.subject.id, stage="confirm")), 1)
        self.assertEqual(self.store.baseline_values(self.subject.id), [1.0])
        self.assertEqual(self.store.count_trials(self.subject.id), 3)

    def test_finding_is_unique_per_hypothesis(self):
        hyp = self.store.add_hypothesis(Hypothesis(subject_id=self.subject.id, title="t", operator="op"))
        finding = self.store.save_finding(Finding(subject_id=self.subject.id, hypothesis_id=hyp.id))
        finding.verdict = Verdict.PROVEN
        finding.effect = 0.05
        self.store.save_finding(finding)
        again = self.store.finding_for_hypothesis(hyp.id)
        self.assertEqual(again.id, finding.id)
        self.assertEqual(again.verdict, Verdict.PROVEN)
        self.assertEqual(len(self.store.list_findings(self.subject.id)), 1)

    def test_challenges_track_open_state(self):
        finding = self.store.save_finding(Finding(subject_id=self.subject.id, hypothesis_id="h1"))
        challenge = self.store.add_challenge(
            Challenge(subject_id=self.subject.id, finding_id=finding.id, reason="doubt",
                      axis={"seed": [1, 2]})
        )
        self.assertEqual(len(self.store.open_challenges(finding.id)), 1)
        self.assertEqual(self.store.list_challenges(finding_id=finding.id)[0].axis, {"seed": [1, 2]})
        challenge.status = "resolved"
        self.store.save_challenge(challenge)
        self.assertEqual(self.store.open_challenges(finding.id), [])

    def test_capabilities_upsert_on_subject_and_operator(self):
        self.store.save_capability(Capability(operator="tune:lr", subject_id=self.subject.id, alpha=2.0))
        self.store.save_capability(Capability(operator="tune:lr", subject_id=self.subject.id, alpha=5.0, n=3))
        caps = self.store.list_capabilities(self.subject.id)
        self.assertEqual(len(caps), 1)
        self.assertEqual(caps[0].alpha, 5.0)
        self.assertEqual(caps[0].n, 3)

    def test_leads_deduplicate_by_url(self):
        for _ in range(3):
            self.store.add_lead(Lead(subject_id=self.subject.id, title="t", url="https://x.test/a"))
        self.assertEqual(len(self.store.list_leads(self.subject.id)), 1)

    def test_prompts_and_runs_persist(self):
        self.store.add_prompt(Prompt(subject_id=self.subject.id, text="hello", kind="steer"))
        run = self.store.add_run(Run(subject_id=self.subject.id, salt="s", steps_requested=5))
        run.status = "finished"
        self.store.save_run(run)
        self.assertEqual(self.store.list_prompts(self.subject.id)[0].text, "hello")
        self.assertEqual(self.store.list_runs(self.subject.id)[0].status, "finished")

    def test_survives_reopening_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.db")
            with Store(path) as store:
                store.add_subject(a_subject("persisted"))
            with Store(path) as store:
                self.assertIsNotNone(store.get_subject("persisted"))


if __name__ == "__main__":
    unittest.main()
