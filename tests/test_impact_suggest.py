"""Business-impact scoring, web-lead handling, and the suggestion engine."""

import unittest

from arp.impact import CONFIDENCE, business_brief, durability_of, rank_opportunities, score_finding
from arp.models import Finding, Hypothesis, Lead, Prompt, Subject, Verdict
from arp.store import Store
from arp.suggest import suggest
from arp.websearch import authority_of, build_queries, domain_of, market_signal, score_hit, SearchHit


def make_subject(slug="impact") -> Subject:
    return Subject(
        slug=slug, title="Reduce inference cost per token", metric="cost", direction="minimize",
        description="Serving cost per million tokens on production hardware.",
        config={
            "param_space": {"batch": {"type": "int", "default": 8, "group": "batching"}},
            "baseline_params": {"batch": 8},
            "value_model": {"reference_effect": 0.01, "effort_days": 5,
                            "value_per_relative_point": 1_000_000, "currency": "USD"},
        },
    )


def a_finding(subject_id, hypothesis_id, verdict=Verdict.PROVEN, effect=0.02, cells_passed=2,
              cells_total=2) -> Finding:
    return Finding(
        subject_id=subject_id, hypothesis_id=hypothesis_id, verdict=verdict,
        effect=effect, ci_low=effect * 0.8, ci_high=effect * 1.2, p_value=0.001, n_trials=12,
        evidence={"confirmation": {"cells": [
            {"passed": i < cells_passed} for i in range(cells_total)
        ]}},
    )


class TestWebScoring(unittest.TestCase):
    def test_domain_and_authority(self):
        self.assertEqual(domain_of("https://www.arxiv.org/abs/1234"), "arxiv.org")
        self.assertGreater(authority_of("https://arxiv.org/abs/1"), authority_of("https://reddit.com/r/x"))
        self.assertGreater(authority_of("https://cs.stanford.edu/paper"), 0.7)
        self.assertEqual(authority_of("https://some-random-blog.example"), 0.4)

    def test_queries_are_deterministic_and_deduplicated(self):
        subject = make_subject()
        first = build_queries(subject)
        self.assertEqual(first, build_queries(subject))
        self.assertEqual(len(first), len(set(first)))

    def test_hit_scoring_rewards_relevance_and_commercial_language(self):
        subject = make_subject()
        commercial = SearchHit(title="Inference cost per token cut 40%",
                               snippet="Production serving cost savings and throughput ROI",
                               url="https://arxiv.org/abs/1")
        irrelevant = SearchHit(title="Sourdough starter tips", snippet="flour and water",
                               url="https://medium.com/x")
        good = score_hit(subject, commercial, "inference cost")
        bad = score_hit(subject, irrelevant, "inference cost")
        self.assertGreater(good["relevance"], bad["relevance"])
        self.assertGreater(good["commercial"], bad["commercial"])
        self.assertGreater(good["authority"], bad["authority"])

    def test_market_signal_weights_by_authority(self):
        strong = [Lead(subject_id="s", title="t", url="https://arxiv.org/a",
                       scores={"authority": 0.9, "commercial": 0.9, "relevance": 0.9, "recency": 1.0})]
        weak = [Lead(subject_id="s", title="t", url="https://reddit.com/a",
                     scores={"authority": 0.3, "commercial": 0.1, "relevance": 0.1, "recency": 0.2})]
        self.assertGreater(market_signal(strong), market_signal(weak))
        self.assertEqual(market_signal([]), 0.0)


class TestImpact(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(make_subject())
        self.hyp = self.store.add_hypothesis(
            Hypothesis(subject_id=self.subject.id, title="bigger batch", operator="tune:batching",
                       params={"batch": 16})
        )

    def tearDown(self):
        self.store.close()

    def test_unproven_findings_score_below_proven_ones(self):
        proven = score_finding(self.subject, a_finding(self.subject.id, self.hyp.id, Verdict.PROVEN))
        supported = score_finding(self.subject, a_finding(self.subject.id, self.hyp.id, Verdict.SUPPORTED))
        refuted = score_finding(self.subject, a_finding(self.subject.id, self.hyp.id, Verdict.REFUTED))
        self.assertGreater(proven.total, supported.total)
        self.assertEqual(refuted.total, 0.0)
        self.assertEqual(CONFIDENCE[Verdict.REFUTED], 0.0)

    def test_durability_counts_the_stress_cells(self):
        self.assertEqual(durability_of(a_finding("s", "h", cells_passed=2, cells_total=2)), 1.0)
        self.assertEqual(durability_of(a_finding("s", "h", cells_passed=1, cells_total=2)), 0.5)
        self.assertEqual(durability_of(Finding(subject_id="s", hypothesis_id="h")), 0.0)

    def test_market_evidence_raises_the_score(self):
        finding = a_finding(self.subject.id, self.hyp.id)
        leads = [Lead(subject_id=self.subject.id, title="cost", url="https://arxiv.org/a",
                      scores={"authority": 0.9, "commercial": 0.9, "relevance": 0.9, "recency": 1.0})]
        self.assertGreater(
            score_finding(self.subject, finding, leads).total,
            score_finding(self.subject, finding, []).total,
        )

    def test_money_follows_the_conservative_bound(self):
        score = score_finding(self.subject, a_finding(self.subject.id, self.hyp.id, effect=0.02))
        # value_per_relative_point 1e6 x ci_low (0.016) = 16,000
        self.assertAlmostEqual(score.monetary, 16000.0, places=2)
        self.assertEqual(score.currency, "USD")

    def test_scoring_is_deterministic(self):
        finding = a_finding(self.subject.id, self.hyp.id)
        self.assertEqual(score_finding(self.subject, finding).as_dict(),
                         score_finding(self.subject, finding).as_dict())

    def test_ranking_persists_scores_onto_findings(self):
        self.store.save_finding(a_finding(self.subject.id, self.hyp.id))
        ranked = rank_opportunities(self.store, self.subject)
        self.assertEqual(len(ranked), 1)
        stored = self.store.list_findings(self.subject.id)[0]
        self.assertIn("total", stored.impact)

    def test_brief_names_the_change_to_ship(self):
        finding = a_finding(self.subject.id, self.hyp.id)
        score = score_finding(self.subject, finding)
        text = business_brief(self.subject, finding, score, self.hyp)
        self.assertIn("bigger batch", text)
        self.assertIn("batch = 16", text)
        self.assertIn("Impact score", text)


class TestSuggest(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.subject = self.store.add_subject(make_subject())

    def tearDown(self):
        self.store.close()

    def test_nothing_to_suggest_on_an_empty_platform(self):
        self.assertEqual(suggest(self.store, self.subject), [])

    def test_near_misses_are_surfaced_for_a_second_look(self):
        hyp = self.store.add_hypothesis(
            Hypothesis(subject_id=self.subject.id, title="batch 16", operator="tune:batching",
                       params={"batch": 16})
        )
        self.store.save_finding(Finding(
            subject_id=self.subject.id, hypothesis_id=hyp.id, verdict=Verdict.INCONCLUSIVE,
            effect=0.004, ci_low=-0.001, ci_high=0.009,
        ))
        ideas = suggest(self.store, self.subject)
        near = [i for i in ideas if i.kind == "near_miss"]
        self.assertTrue(near)
        self.assertIn("batch 16", near[0].title)
        self.assertTrue(near[0].seed_prompt)

    def test_proven_results_suggest_transfer_to_similar_subjects(self):
        source = self.store.add_subject(make_subject("cost-twin"))
        hyp = self.store.add_hypothesis(
            Hypothesis(subject_id=source.id, title="batch 32", operator="tune:batching",
                       params={"batch": 32})
        )
        self.store.save_finding(Finding(subject_id=source.id, hypothesis_id=hyp.id,
                                        verdict=Verdict.PROVEN, effect=0.03, ci_low=0.02, ci_high=0.04))
        ideas = suggest(self.store, self.subject)
        transfers = [i for i in ideas if i.kind == "transfer"]
        self.assertTrue(transfers)
        self.assertIn("cost-twin", transfers[0].title)

    def test_repeated_prompt_themes_become_subject_ideas(self):
        """'Prompt a subject based on previous inputs and chats', concretely."""
        for _ in range(4):
            self.store.add_prompt(Prompt(subject_id=self.subject.id,
                                         text="what about quantisation for serving?"))
        ideas = suggest(self.store, self.subject)
        themes = [i for i in ideas if i.kind == "theme"]
        self.assertTrue(themes)
        self.assertTrue(any("quantisation" in i.title for i in themes))

    def test_suggestions_are_ranked_and_capped(self):
        for i in range(6):
            hyp = self.store.add_hypothesis(
                Hypothesis(subject_id=self.subject.id, title=f"h{i}", operator="tune:batching",
                           params={"batch": 10 + i})
            )
            self.store.save_finding(Finding(
                subject_id=self.subject.id, hypothesis_id=hyp.id, verdict=Verdict.INCONCLUSIVE,
                effect=0.001 * i, ci_low=-0.001, ci_high=0.002 * (i + 1),
            ))
        ideas = suggest(self.store, self.subject, limit=3)
        self.assertEqual(len(ideas), 3)
        self.assertEqual([i.score for i in ideas], sorted((i.score for i in ideas), reverse=True))


if __name__ == "__main__":
    unittest.main()
