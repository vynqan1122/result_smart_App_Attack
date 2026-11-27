"""Offline checks for tracked source syntax and historical benchmark artifacts."""

import ast
import csv
import math
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "appendix_outputs" / "attack_summary.csv"
EXPECTED_FIELDS = [
    "dataset", "backbone", "training_mode", "baseline", "attack_algo",
    "model_name", "epsilon", "num_images", "mean_success_rate",
    "min_success_rate", "max_success_rate", "path",
]
RATE_FIELDS = ["mean_success_rate", "min_success_rate", "max_success_rate"]
TOLERANCE = 1e-9


class RepositoryChecks(unittest.TestCase):
    def test_tracked_python_syntax(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT
        ).decode("utf-8").split("\0")
        paths = [name for name in tracked if name]
        self.assertTrue(paths, "No tracked Python files found")
        for name in paths:
            with self.subTest(path=name):
                path = ROOT / name
                ast.parse(path.read_bytes(), filename=name)

    def test_appendix_summary(self):
        with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, EXPECTED_FIELDS)
            rows = list(reader)
        self.assertTrue(rows, "Benchmark summary is empty")
        for index, row in enumerate(rows, start=2):
            with self.subTest(csv_line=index):
                self.assertNotIn(None, row, "Unexpected extra CSV columns")
                for field in EXPECTED_FIELDS:
                    self.assertTrue(row.get(field), f"Missing {field}")
                self.assertGreater(int(row["num_images"]), 0)
                epsilon = float(row["epsilon"])
                self.assertTrue(math.isfinite(epsilon))
                self.assertGreaterEqual(epsilon, 0)
                rates = {field: float(row[field]) for field in RATE_FIELDS}
                for field, value in rates.items():
                    self.assertTrue(math.isfinite(value), field)
                    self.assertGreaterEqual(value, -TOLERANCE, field)
                    self.assertLessEqual(value, 1 + TOLERANCE, field)
                self.assertLessEqual(
                    rates["min_success_rate"],
                    rates["mean_success_rate"] + TOLERANCE,
                )
                self.assertLessEqual(
                    rates["mean_success_rate"],
                    rates["max_success_rate"] + TOLERANCE,
                )

    def test_appendix_figure(self):
        path = ROOT / "appendix_outputs" / "attack_success_summary.png"
        with path.open("rb") as handle:
            self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertGreater(path.stat().st_size, 8)

    def test_required_documentation(self):
        for name in [
            "README.md", "README.vi.md", "CONTRIBUTING.md", "ROADMAP.md",
            "CITATION.cff", "LICENSING.md", "SECURITY.md", "docs/BENCHMARKS.md",
        ]:
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
