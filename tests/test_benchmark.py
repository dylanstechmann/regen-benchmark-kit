import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from regenbench.benchmark import evaluate, group_accuracy_interval, save_results
from regenbench.cli import write_demo
from regenbench.data import connected_groups, load_table


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "demo.csv"
        write_demo(self.path)

    def test_joint_groups_use_transitive_connections(self):
        rows = [{"donor": "a", "plate": "1"}, {"donor": "a", "plate": "2"},
                {"donor": "b", "plate": "2"}, {"donor": "c", "plate": "3"}]
        groups = connected_groups(rows, ["donor", "plate"])
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])

    def test_oof_group_disjoint_predictions_and_reproducibility(self):
        data = load_table(self.path, ["donor_id", "batch_id"])
        report, rows = evaluate(data, folds=5, bootstrap_draws=100)
        again, same_rows = evaluate(data, folds=5, bootstrap_draws=100)
        self.assertEqual(report, again)
        self.assertEqual(rows, same_rows)
        self.assertEqual(len(rows), len(data.y))
        self.assertEqual(len({r["sample_id"] for r in rows}), len(data.y))
        for fold in report["folds"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
        self.assertGreater(report["models"]["logistic"]["balanced_accuracy"], 0.7)
        self.assertAlmostEqual(report["models"]["majority"]["balanced_accuracy"], 0.5)
        out = Path(self.tmp.name) / "results"
        save_results(report, rows, out)
        self.assertTrue((out / "predictions.csv").exists())
        with self.assertRaises(FileExistsError):
            save_results(report, rows, out)

    def test_unblocked_batches_are_reported(self):
        report, _ = evaluate(load_table(self.path, ["donor_id"]), bootstrap_draws=100)
        self.assertTrue(any(report["unblocked_overlap_counts"]["batch_id"]))

    def test_metadata_never_enters_feature_matrix(self):
        data = load_table(self.path, ["donor_id"])
        self.assertEqual(data.features, ["f_signal", "f_nuisance"])
        self.assertEqual(data.x.shape[1], 2)

    def test_reject_malformed_and_nonfinite_tables(self):
        for body in ["sample_id,label,group_id,f_x\na,A,g,nan\nb,B,h,1\n",
                     "sample_id,label,group_id,f_x\na,A,g,1\na,B,h,2\n",
                     "sample_id,label,group_id,f_x\na,A,,1\nb,B,h,2\n",
                     "sample_id,label,group_id,f_x\na,A,g,1,2\nb,B,h,2\n"]:
            self.path.write_text(body)
            with self.assertRaises(ValueError):
                load_table(self.path, ["group_id"])

    def test_single_connected_component_cannot_be_split(self):
        with self.path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["batch_id"] = "one-plate"
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "independent groups"):
            evaluate(load_table(self.path, ["donor_id", "batch_id"]))

    def test_group_interval_weights_independent_units_equally(self):
        y = np.array(["a"] * 12)
        pred = np.array(["a"] * 10 + ["b"] * 2)
        groups = np.array(["large"] * 10 + ["small-1", "small-2"])
        report = group_accuracy_interval(y, pred, groups, draws=100)
        self.assertAlmostEqual(report["estimate"], 1 / 3)
        self.assertIsNotNone(report["ci95"])

    def test_training_only_scaler_fit(self):
        from unittest.mock import patch
        from sklearn.preprocessing import StandardScaler
        data = load_table(self.path, ["donor_id"])
        captured = []
        original = StandardScaler.fit

        def recording_fit(scaler, x, *args, **kwargs):
            captured.append(x.copy())
            return original(scaler, x, *args, **kwargs)

        with patch.object(StandardScaler, "fit", recording_fit):
            report, _ = evaluate(data, bootstrap_draws=100)
        self.assertEqual(len(captured), 5)
        for fold, fitted in zip(report["folds"], captured):
            expected = data.x[np.isin(data.groups, fold["train_groups"])]
            np.testing.assert_array_equal(fitted, expected)


if __name__ == "__main__":
    unittest.main()
