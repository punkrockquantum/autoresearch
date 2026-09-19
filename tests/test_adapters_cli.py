"""The bridge to this repo's `train.py`, the runners, and the command line."""

import io
import os
import contextlib
import tempfile
import unittest

from arp.adapters.autoresearch_train import (
    autoresearch_subject,
    patch_constants,
    patch_seed,
    read_results_tsv,
)
from arp.agents import classify_prompt, extract_json, parse_param_hints
from arp.cli import demo_subject, main
from arp.models import Subject, TrialSpec
from arp.report import markdown_to_html
from arp.runners import CommandRunner, SimulatedRunner, build_runner, parse_metrics

TRAIN_SNIPPET = """\
import torch

ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
WINDOW_PATTERN = "SSSL" # sliding window pattern
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
DEPTH = 8               # number of transformer layers

torch.manual_seed(42)
torch.cuda.manual_seed(42)
"""


class TestTrainAdapter(unittest.TestCase):
    def test_patches_constants_and_keeps_comments_aligned(self):
        patched = patch_constants(TRAIN_SNIPPET, {"DEPTH": 12, "MATRIX_LR": 0.06})
        self.assertIn("DEPTH = 12", patched)
        self.assertIn("# number of transformer layers", patched)
        self.assertIn("MATRIX_LR = 0.06", patched)
        self.assertIn("ASPECT_RATIO = 64", patched)

        def comment_column(text, name):
            line = next(l for l in text.splitlines() if l.startswith(name + " ="))
            return line.index("#")

        for name in ("DEPTH", "MATRIX_LR"):
            self.assertEqual(comment_column(patched, name), comment_column(TRAIN_SNIPPET, name))

    def test_patches_string_constants_as_literals(self):
        patched = patch_constants(TRAIN_SNIPPET, {"WINDOW_PATTERN": "L"})
        self.assertIn("WINDOW_PATTERN = 'L'", patched)

    def test_an_unknown_constant_is_an_error_not_a_silent_no_op(self):
        """A typo must not become an experiment that quietly measured nothing."""
        with self.assertRaises(KeyError) as ctx:
            patch_constants(TRAIN_SNIPPET, {"NO_SUCH_CONSTANT": 1})
        self.assertIn("NO_SUCH_CONSTANT", str(ctx.exception))

    def test_seeds_are_repointed_so_replicates_really_differ(self):
        patched = patch_seed(TRAIN_SNIPPET, 1234)
        self.assertIn("torch.manual_seed(1234)", patched)
        self.assertIn("torch.cuda.manual_seed(1234)", patched)
        self.assertNotIn("42", patched)

    def test_the_subject_mirrors_the_constants_the_agent_may_touch(self):
        subject = autoresearch_subject()
        self.assertEqual(subject.metric, "val_bpb")
        self.assertEqual(subject.direction, "minimize")
        self.assertEqual(subject.runner, "train")
        for name in ("MATRIX_LR", "DEPTH", "WINDOW_PATTERN", "DEVICE_BATCH_SIZE"):
            self.assertIn(name, subject.param_space)
        self.assertIn("scale", subject.stress_axes)

    def test_every_declared_parameter_exists_in_the_real_train_py(self):
        """The adapter's parameter space must not drift from the actual file."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")
        with open(path) as f:
            source = f.read()
        subject = autoresearch_subject()
        params = {name: subject.param_space[name].get("default") for name in subject.param_space}
        patch_constants(source, params)  # raises if any constant is missing

    def test_results_tsv_import_tolerates_junk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.tsv")
            with open(path, "w") as f:
                f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
                f.write("a1b2c3d\t0.997900\t44.0\tkeep\tbaseline\n")
                f.write("broken-row\n")
                f.write("c3d4e5f\tnot-a-number\t44.0\tcrash\tbad\n")
            rows = read_results_tsv(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["val_bpb"], 0.9979)
        self.assertEqual(rows[1]["val_bpb"], 0.0)
        self.assertEqual(read_results_tsv("/nonexistent/results.tsv"), [])


class TestRunners(unittest.TestCase):
    def test_metric_parsing_matches_the_train_summary_block(self):
        parsed = parse_metrics("""
---
val_bpb:          0.997900
training_seconds: 300.1
peak_vram_mb:     45060.2
not a metric line
depth:            8
""")
        self.assertAlmostEqual(parsed["val_bpb"], 0.9979)
        self.assertAlmostEqual(parsed["peak_vram_mb"], 45060.2)
        self.assertEqual(parsed["depth"], 8)
        self.assertNotIn("not", parsed)

    def test_simulated_runner_is_seeded_and_reproducible(self):
        subject = demo_subject()
        runner = SimulatedRunner(subject, ".")
        spec = TrialSpec(subject=subject, params={"lr": 0.02}, seed=7)
        first, second = runner.run(spec), runner.run(spec)
        self.assertTrue(first.ok)
        self.assertEqual(first.metric, second.metric)

    def test_simulated_runner_respects_the_direction_of_a_real_effect(self):
        subject = demo_subject()
        runner = SimulatedRunner(subject, ".")
        better = [runner.run(TrialSpec(subject=subject, params={"lr": 0.03}, seed=s)).metric
                  for s in range(30)]
        worse = [runner.run(TrialSpec(subject=subject, params={"lr": 0.003}, seed=s)).metric
                 for s in range(30)]
        self.assertLess(sum(better) / len(better), sum(worse) / len(worse))

    def test_out_of_range_parameters_fail_like_an_oom(self):
        subject = demo_subject()
        result = SimulatedRunner(subject, ".").run(
            TrialSpec(subject=subject, params={"lr": 99.0}, seed=1)
        )
        self.assertFalse(result.ok)
        self.assertIn("outside", result.error)

    def test_baseline_arm_uses_the_spec_not_the_stale_subject(self):
        """A baseline trial must measure the *current* baseline, not the defaults."""
        subject = demo_subject()
        runner = SimulatedRunner(subject, ".")
        default_arm = runner.run(TrialSpec(subject=subject, params={}, seed=3, arm="baseline"))
        moved_arm = runner.run(TrialSpec(subject=subject, params={"lr": 0.2}, seed=3, arm="baseline"))
        self.assertNotEqual(default_arm.metric, moved_arm.metric)

    def test_command_runner_reads_metrics_from_a_shell_command(self):
        subject = Subject(slug="cmd", title="Shell subject", metric="score", direction="maximize",
                          runner="command",
                          config={"param_space": {"x": {"type": "int", "default": 2, "group": "g"}},
                                  "runner_config": {"command": "echo score: {x}"}})
        with tempfile.TemporaryDirectory() as tmp:
            result = CommandRunner(subject, tmp).run(
                TrialSpec(subject=subject, params={"x": 5}, seed=1)
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.metric, 5.0)

    def test_command_runner_reports_a_failing_command(self):
        subject = Subject(slug="cmd2", title="Broken", metric="score",
                          config={"runner_config": {"command": "exit 3"}})
        with tempfile.TemporaryDirectory() as tmp:
            result = CommandRunner(subject, tmp).run(TrialSpec(subject=subject, params={}, seed=1))
        self.assertFalse(result.ok)
        self.assertIn("exit 3", result.error)

    def test_registry_builds_the_declared_runner(self):
        self.assertIsInstance(build_runner(demo_subject(), "."), SimulatedRunner)
        with self.assertRaises(KeyError):
            build_runner(demo_subject(), ".", override="nope")


class TestPromptParsing(unittest.TestCase):
    def test_intents_are_classified_without_a_model(self):
        self.assertEqual(classify_prompt("I don't believe that result"), "challenge")
        self.assertEqual(classify_prompt("try DEPTH=12"), "hypothesis")
        self.assertEqual(classify_prompt("never exceed 40GB of VRAM"), "constraint")
        self.assertEqual(classify_prompt("why did that regress?"), "question")
        self.assertEqual(classify_prompt("architecture work from here"), "steer")

    def test_parameter_hints_are_pulled_out_of_plain_english(self):
        subject = autoresearch_subject()
        self.assertEqual(parse_param_hints(subject, "try DEPTH=12"), {"DEPTH": 12})
        self.assertEqual(parse_param_hints(subject, "set MATRIX_LR to 0.06"), {"MATRIX_LR": 0.06})
        self.assertEqual(parse_param_hints(subject, "use WINDOW_PATTERN = L"), {"WINDOW_PATTERN": "L"})
        self.assertEqual(parse_param_hints(subject, "make it better somehow"), {})

    def test_json_extraction_survives_chatty_models(self):
        self.assertEqual(extract_json('Sure!\n```json\n[{"a": 1}]\n```'), [{"a": 1}])
        self.assertEqual(extract_json('here you go: {"a": 2} hope that helps'), {"a": 2})
        self.assertIsNone(extract_json("no json here"))


class TestReportRendering(unittest.TestCase):
    def test_markdown_renders_to_self_contained_html(self):
        html = markdown_to_html("# Title\n\n- one\n- two\n\n```\ncode\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
                                title="T")
        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<li>one</li>", html)
        self.assertIn("<pre>", html)
        self.assertIn("<table>", html)
        self.assertIn("prefers-color-scheme", html)

    def test_html_escapes_content(self):
        self.assertIn("&lt;script&gt;", markdown_to_html("<script>alert(1)</script>"))


class TestCLI(unittest.TestCase):
    def run_cli(self, *args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(list(args))
        self.assertEqual(code, 0, buf.getvalue())
        return buf.getvalue()

    def test_full_command_line_journey(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "init")
            self.run_cli("--db", db, "subject", "add", "--preset", "demo")
            listing = self.run_cli("--db", db, "subject", "list")
            self.assertIn("demo-optimizer", listing)

            self.run_cli("--db", db, "run", "demo-optimizer", "--steps", "25", "--no-web", "--quiet")
            self.assertIn("trials", self.run_cli("--db", db, "status", "demo-optimizer"))
            self.assertTrue(self.run_cli("--db", db, "findings", "demo-optimizer"))

            prompt_out = self.run_cli("--db", db, "prompt", "demo-optimizer", "try lr=0.04")
            self.assertIn("hypothesis", prompt_out)

            report = self.run_cli("--db", db, "report", "demo-optimizer")
            self.assertIn("Where things stand", report)

            html_path = os.path.join(tmp, "r.html")
            self.run_cli("--db", db, "report", "demo-optimizer", "--html", html_path)
            self.assertTrue(os.path.exists(html_path))

            self.run_cli("--db", db, "impact", "demo-optimizer")
            self.run_cli("--db", db, "suggest")
            self.run_cli("--db", db, "leads", "demo-optimizer")

    def test_unknown_subject_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "init")
            with self.assertRaises(SystemExit):
                main(["--db", db, "status", "nope"])

    def test_duplicate_subject_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "p.db")
            self.run_cli("--db", db, "subject", "add", "--preset", "demo")
            with self.assertRaises(SystemExit):
                main(["--db", db, "subject", "add", "--preset", "demo"])


if __name__ == "__main__":
    unittest.main()
