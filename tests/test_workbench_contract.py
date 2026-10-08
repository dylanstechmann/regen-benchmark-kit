"""The metrics report keeps the shape that regen-workbench's ``regenbench-metrics/1`` adapter reads.

regen-workbench (tools/frozen_evaluation.py, adapt_regenbench_metrics) binds a frozen plan to this
report. These tests pin the fields it requires, so a rename here fails in this repository first.
When a sibling regen-workbench checkout is present the adapter itself is also run on a real report.
"""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from regenbench.cli import main

REQUIRED = ("task", "dataset_sha256", "group_metadata", "configuration", "folds", "models", "environment")
ADAPTER = Path(__file__).resolve().parents[2] / "regen-workbench" / "tools" / "frozen_evaluation.py"


class WorkbenchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        table = root / "features.csv"
        with table.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sample_id", "target", "source_well", "batch_id", "f_signal"])
            for group in range(4):
                for i in range(6):
                    x = group * 100 + i
                    writer.writerow([f"{group}-{i}", 2 * x + 1, f"well{group}", "one-study", x])
        main(["regress", str(table), "--group-by", "source_well", "--out", str(root / "out")])
        cls.report = json.loads((root / "out" / "metrics.json").read_text(encoding="utf-8"))
        main(["demo", "--out", str(root / "demo.csv")])
        main(["run", str(root / "demo.csv"), "--group-by", "donor_id", "--folds", "3",
              "--bootstrap-draws", "100", "--out", str(root / "run_out")])
        cls.classification = json.loads((root / "run_out" / "metrics.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_required_fields_are_present(self):
        for key in REQUIRED:
            self.assertIn(key, self.report)

    def test_grouping_columns_and_fold_groups_resolve(self):
        columns = self.report["configuration"]["group_by"]
        self.assertEqual(columns, ["source_well"])
        self.assertRegex(self.report["dataset_sha256"], r"^[0-9a-f]{64}$")
        known = set(self.report["group_metadata"])
        for fold in self.report["folds"]:
            self.assertTrue(set(fold["train_groups"]) | set(fold["test_groups"]) <= known)
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))

    def test_classification_run_reports_carry_the_adapter_fields_too(self):
        # Added 2026-10-08 after the first version of this test found they were missing.
        for key in REQUIRED:
            self.assertIn(key, self.classification)
        self.assertEqual(self.classification["task"], "classification")
        known = set(self.classification["group_metadata"])
        for fold in self.classification["folds"]:
            self.assertTrue(set(fold["train_groups"]) | set(fold["test_groups"]) <= known)
        self.assertTrue(self.classification["group_metadata"][sorted(known)[0]]["donor_id"][0].startswith("d"))

    @unittest.skipUnless(ADAPTER.is_file(), "sibling regen-workbench checkout not present")
    def test_workbench_adapter_accepts_the_report(self):
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location("frozen_evaluation_contract", ADAPTER)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        for report in (self.report, self.classification):
            normalized = module.adapt_regenbench_metrics(report)
            self.assertEqual(normalized["adapter"], "regenbench-metrics/1")
            self.assertEqual(normalized["input_sha256"], [report["dataset_sha256"]])
            self.assertEqual(len(normalized["folds"]), len(report["folds"]))


if __name__ == "__main__":
    unittest.main()
