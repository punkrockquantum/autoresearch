"""Turning a sentence (and prior chat) into a runnable research subject."""

import io
import contextlib
import json
import os
import tempfile
import unittest

from arp.authoring import (
    create_subject,
    draft_subject,
    infer_direction,
    infer_metric,
    nearest_subject,
    related_prompts,
    slugify,
    starter_config,
    validate_param_space,
)
from arp.cli import demo_subject, main
from arp.config import PlatformConfig
from arp.models import Prompt, Subject
from arp.orchestrator import Orchestrator
from arp.store import Store


class TestInference(unittest.TestCase):
    def test_direction_comes_from_the_verb(self):
        self.assertEqual(infer_direction("minimise serving cost"), "minimize")
        self.assertEqual(infer_direction("maximise win rate"), "maximize")
        self.assertEqual(infer_direction("improve accuracy"), "maximize")
        self.assertEqual(infer_direction("reduce latency"), "minimize")
        self.assertEqual(infer_direction("something about widgets"), "minimize")

    def test_the_first_verb_wins_when_both_appear(self):
        self.assertEqual(infer_direction("reduce cost to increase margin"), "minimize")
        self.assertEqual(infer_direction("increase margin by reducing cost"), "maximize")

    def test_known_metrics_are_recognised(self):
        self.assertEqual(infer_metric("get val_bpb down"), "val_bpb")
        self.assertEqual(infer_metric("cut cost per million tokens"), "cost_per_million_tokens")
        self.assertEqual(infer_metric("improve accuracy on the eval set"), "accuracy")

    def test_an_unknown_metric_is_taken_from_the_phrasing(self):
        self.assertEqual(infer_metric("minimise widget jitter"), "widget_jitter")
        self.assertEqual(infer_metric("do something vague"), "score")

    def test_slugs_are_short_and_unique(self):
        self.assertEqual(slugify("Reduce the serving cost per token"), "reduce-serving-cost-token")
        self.assertEqual(slugify("Reduce the serving cost per token", {"reduce-serving-cost-token"}),
                         "reduce-serving-cost-token-2")
        self.assertTrue(slugify("!!!"))


class TestParamSpaceValidation(unittest.TestCase):
    def test_good_specs_survive(self):
        clean = validate_param_space({
            "lr": {"type": "float", "default": 0.01, "low": 0.001, "high": 0.1, "group": "lr"},
            "depth": {"type": "int", "default": 8, "low": 2, "high": 32, "step": 2, "group": "arch"},
            "act": {"type": "choice", "default": "gelu", "values": ["gelu", "relu"], "group": "arch"},
        })
        self.assertEqual(set(clean), {"lr", "depth", "act"})
        self.assertTrue(clean["lr"]["log"])

    def test_malformed_specs_are_dropped_not_repaired(self):
        """A plausible-looking but broken spec would produce trials measuring nothing."""
        clean = validate_param_space({
            "bad_type": {"type": "tensor", "default": 1},
            "no_default": {"type": "float", "low": 0, "high": 1},
            "inverted": {"type": "float", "default": 5, "low": 10, "high": 1},
            "out_of_range": {"type": "int", "default": 99, "low": 1, "high": 10},
            "one_choice": {"type": "choice", "default": "a", "values": ["a"]},
            "not a name": {"type": "float", "default": 1.0},
            "fine": {"type": "float", "default": 1.0, "low": 0.1, "high": 10.0},
        })
        self.assertEqual(set(clean), {"fine"})

    def test_log_scale_is_disabled_for_non_positive_defaults(self):
        clean = validate_param_space({"x": {"type": "float", "default": 0.0, "log": True}})
        self.assertFalse(clean["x"]["log"])


class TestDrafting(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.demo = self.store.add_subject(demo_subject())

    def tearDown(self):
        self.store.close()

    def test_an_unrelated_subject_is_drafted_empty_and_says_so(self):
        draft = draft_subject(self.store, "improve espresso extraction yield by adjusting grind size")
        self.assertEqual(draft.source, "empty")
        self.assertFalse(draft.ready)
        self.assertEqual(draft.subject.direction, "maximize")
        self.assertTrue(any("no parameter space" in n for n in draft.notes))

    def test_an_explicit_template_is_copied_whole(self):
        draft = draft_subject(self.store, "a second optimiser subject", template_slug="demo-optimizer")
        self.assertEqual(draft.source, "template")
        self.assertTrue(draft.ready)
        self.assertEqual(set(draft.subject.param_space), set(self.demo.param_space))
        self.assertEqual(draft.subject.stress_axes, self.demo.stress_axes)
        self.assertEqual(draft.subject.baseline_params["lr"], 0.01)

    def test_a_similar_subject_is_found_without_being_named(self):
        draft = draft_subject(
            self.store, "tune a synthetic optimiser to minimise simulated validation loss again"
        )
        self.assertEqual(draft.source, "similar")
        self.assertTrue(draft.ready)
        self.assertTrue(any("copied from 'demo-optimizer'" in n for n in draft.notes))

    def test_copying_a_template_carries_the_runner_it_needs(self):
        """A parameter space without its runner is a subject that fails every trial."""
        draft = draft_subject(self.store, "another optimiser subject", template_slug="demo-optimizer")
        self.assertEqual(draft.subject.runner, "simulated")
        self.assertIn("simulation", draft.subject.config)
        self.assertEqual(draft.subject.metric, self.demo.metric)

    def test_an_explicit_runner_is_respected_over_the_template(self):
        draft = draft_subject(self.store, "another optimiser subject",
                              template_slug="demo-optimizer", runner="command")
        self.assertEqual(draft.subject.runner, "command")

    def test_explicit_flags_beat_inference(self):
        draft = draft_subject(self.store, "reduce widget latency", metric="p99_ms",
                              direction="maximize", slug="custom-slug")
        self.assertEqual(draft.subject.metric, "p99_ms")
        self.assertEqual(draft.subject.direction, "maximize")
        self.assertEqual(draft.subject.slug, "custom-slug")

    def test_an_unknown_template_is_an_error(self):
        with self.assertRaises(KeyError):
            draft_subject(self.store, "x", template_slug="nope")

    def test_prior_chat_is_found_and_carried_over(self):
        """'Prompt a subject based on previous inputs and chats', end to end."""
        for _ in range(3):
            self.store.add_prompt(Prompt(
                subject_id=self.demo.id,
                text="we should look at quantisation to reduce serving cost per token",
            ))
        found = related_prompts(self.store, "quantisation for serving cost")
        self.assertTrue(found)

        draft = draft_subject(self.store, "reduce serving cost per token with quantisation")
        self.assertEqual(draft.context_prompts, 3)
        create_subject(self.store, draft)
        carried = [p for p in self.store.list_prompts(draft.subject.id) if p.kind == "context"]
        self.assertEqual(len(carried), 3)
        self.assertIn("carried_from_subject", carried[0].meta)

    def test_creation_warm_starts_the_capability_memory(self):
        from arp.evolution import CapabilityMemory
        from arp.models import Verdict

        memory = CapabilityMemory(self.store)
        for _ in range(8):
            memory.record_verdict(self.demo, "tune:lr", Verdict.PROVEN, 0.03)
        draft = create_subject(
            self.store,
            draft_subject(self.store, "a second optimiser subject", template_slug="demo-optimizer"),
        )
        self.assertIsNotNone(self.store.get_capability("tune:lr", draft.subject.id))

    def test_a_drafted_subject_actually_runs(self):
        draft = create_subject(
            self.store,
            draft_subject(self.store, "a second optimiser subject", template_slug="demo-optimizer"),
        )
        orch = Orchestrator(self.store, draft.subject,
                            PlatformConfig(db_path=":memory:", work_dir="/tmp"),
                            salt="draft", search_web=False)
        summary = orch.run(steps=20)
        self.assertEqual(summary.aborted, "")
        self.assertTrue(all(s.ok for s in summary.steps), [s.note for s in summary.steps if not s.ok])
        self.assertGreater(self.store.count_trials(draft.subject.id), 0)

    def test_starter_config_is_valid_input_for_an_update(self):
        draft = draft_subject(self.store, "improve espresso extraction yield")
        config = starter_config(draft.subject)
        self.assertIn("param_space", config)
        self.assertTrue(validate_param_space(config["param_space"]))
        self.assertEqual(config["runner_config"]["metric_key"], draft.subject.metric)

    def test_nearest_subject_reports_no_match_on_an_empty_platform(self):
        empty = Store(":memory:")
        found, score = nearest_subject(empty, Subject(slug="x", title="anything"))
        self.assertIsNone(found)
        self.assertEqual(score, 0.0)
        empty.close()


class TestBrokenSetupAborts(unittest.TestCase):
    def test_a_runner_that_cannot_run_stops_the_loop_quickly(self):
        """A broken setup is not a research result; it must not eat the budget."""
        store = Store(":memory:")
        subject = store.add_subject(Subject(
            slug="broken", title="Broken subject", metric="score", runner="command",
            config={"param_space": {"x": {"type": "int", "default": 1, "low": 0, "high": 5,
                                          "group": "g"}},
                    "baseline_params": {"x": 1},
                    "runner_config": {"command": "exit 7"}},
        ))
        orch = Orchestrator(store, subject, PlatformConfig(db_path=":memory:", work_dir="/tmp"),
                            salt="broken", search_web=False)
        summary = orch.run(steps=50)
        self.assertTrue(summary.aborted)
        self.assertLessEqual(len(summary.steps), 6)
        self.assertIn("exit 7", summary.aborted)
        run = store.list_runs(subject.id)[0]
        self.assertEqual(run.status, "failed")
        store.close()


class TestAuthoringCLI(unittest.TestCase):
    def run_cli(self, *args, expect: int = 0) -> str:
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = main(list(args))
        self.assertEqual(code, expect, buf.getvalue() + err.getvalue())
        return buf.getvalue() + err.getvalue()

    def test_new_and_update_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "subject", "add", "--preset", "demo")

            cloned = self.run_cli("--db", db, "subject", "new", "a", "second", "optimiser",
                                  "subject", "--from", "demo-optimizer")
            self.assertIn("source: template", cloned)
            self.assertIn("ready to run", cloned)

            drafted = self.run_cli("--db", db, "subject", "new", "improve", "espresso",
                                   "extraction", "yield")
            self.assertIn("not runnable yet", drafted)
            path = [l.strip() for l in drafted.splitlines() if l.strip().endswith(".config.json")][0]
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                config = json.load(f)
            config["runner_config"]["command"] = "echo espresso_extraction_yield: 1.0"
            with open(path, "w") as f:
                json.dump(config, f)

            updated = self.run_cli("--db", db, "subject", "update",
                                   "improve-espresso-extraction-yield", "--config", path)
            self.assertIn("knob(s)", updated)
            self.assertNotIn("warning", updated)

    def test_repeating_yourself_makes_a_second_subject_not_an_error(self):
        """Two similar sentences are two questions, so they get two slugs."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            first = self.run_cli("--db", db, "subject", "new", "reduce", "widget", "latency")
            second = self.run_cli("--db", db, "subject", "new", "reduce", "widget", "latency")
            self.assertIn("reduce-widget-latency ", first)
            self.assertIn("reduce-widget-latency-2 ", second)

    def test_an_explicitly_reused_slug_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "subject", "new", "reduce", "widget", "latency",
                         "--slug", "widgets")
            with self.assertRaises(SystemExit):
                main(["--db", db, "subject", "new", "something", "else", "--slug", "widgets"])

    def test_working_files_stay_next_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            out = self.run_cli("--db", db, "subject", "new", "improve", "espresso", "yield")
            path = [l.strip() for l in out.splitlines() if l.strip().endswith(".config.json")][0]
            self.assertTrue(path.startswith(tmp), path)

    def test_a_broken_subject_run_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "subject", "new", "reduce", "widget", "latency",
                         "--runner", "command")
            config = os.path.join(tmp, "c.json")
            with open(config, "w") as f:
                json.dump({"param_space": {"x": {"type": "int", "default": 1, "low": 0,
                                                 "high": 3, "group": "g"}},
                           "baseline_params": {"x": 1},
                           "runner_config": {"command": "exit 9"}}, f)
            self.run_cli("--db", db, "subject", "update", "reduce-widget-latency",
                         "--config", config)
            self.run_cli("--db", db, "run", "reduce-widget-latency", "--steps", "30",
                         "--no-web", "--quiet", expect=1)


if __name__ == "__main__":
    unittest.main()
