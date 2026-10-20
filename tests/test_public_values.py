"""Regression checks for selecting fresh versus bundled public evaluations.

Run with: python3 -m unittest discover -s tests -p 'test_*.py'
Only the standard library is required; all writes stay in temporary directories.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPER_FILES = ("public_values.py", "data/public_evaluation.json",
               "data/public_values.tex", "table_public.tex")


class PublicValuesSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.repo = self.workspace / "release"
        self.paper = self.repo / "paper"
        for name in PAPER_FILES:
            target = self.paper / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "paper" / name, target)
        self.snapshot = self.paper / "data/public_evaluation.json"
        self.original = json.loads(self.snapshot.read_text())
        self.default_runs = self.repo / "data/public/walmart-amazon/runs/product-v1-dgx"

    def write_evaluation(self, runs, marker, n=1000):
        evaluation = dict(self.original, evaluated_at_utc=marker, n=n)
        source = runs / "test/evaluation.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(evaluation))
        return evaluation

    def run_generator(self, *args, runs=None, cwd=None):
        env = os.environ.copy()
        env.pop("PUBLIC_RUNS_DIR", None)
        if runs is not None:
            env["PUBLIC_RUNS_DIR"] = str(runs)
        return subprocess.run([sys.executable, str(self.paper / "public_values.py"), *args],
                              cwd=cwd or self.workspace, env=env, capture_output=True, text=True)

    def saved_outputs(self):
        return {name: (self.paper / name).read_bytes() for name in PAPER_FILES[1:]}

    def test_default_source_is_under_repository_root_from_another_cwd(self):
        fresh = self.write_evaluation(self.default_runs, "fresh-default-run")
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.snapshot.read_text()), fresh)
        checked = self.run_generator("--check")
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_override_selects_absolute_or_cwd_relative_runs(self):
        self.write_evaluation(self.default_runs, "default-must-not-win")
        working = self.workspace / "caller"
        working.mkdir()
        for override in (self.workspace / "absolute-runs", Path("relative-runs")):
            with self.subTest(override=str(override)):
                runs = override if override.is_absolute() else working / override
                fresh = self.write_evaluation(runs, f"override-{override.name}")
                result = self.run_generator(runs=override, cwd=working)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(self.snapshot.read_text()), fresh)

    def test_missing_fresh_source_never_falls_back_to_snapshot(self):
        before = self.saved_outputs()
        for runs in (None, self.workspace / "missing-override"):
            for args in ((), ("--check",)):
                with self.subTest(runs=runs, args=args):
                    result = self.run_generator(*args, runs=runs)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("No fresh public evaluation", result.stderr)
                    self.assertIn("--from-json", result.stderr)
                    self.assertEqual(self.saved_outputs(), before)

    def test_explicit_snapshot_check_needs_no_fresh_run(self):
        before = self.saved_outputs()
        result = self.run_generator("--from-json", "--check", runs="missing-runs")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("matches saved outputs", result.stdout)
        self.assertEqual(self.saved_outputs(), before)

    def test_incomplete_fresh_evaluation_does_not_overwrite_snapshot(self):
        before = self.saved_outputs()
        self.write_evaluation(self.default_runs, "incomplete-run", n=999)
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not the complete 1,000-query test", result.stderr)
        self.assertEqual(self.saved_outputs(), before)


if __name__ == "__main__":
    unittest.main()
