import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from regenbench.data import load_table
from regenbench.regression import evaluate_regression, regression_metrics, save_regression_results


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "features.csv"
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "target", "source_well", "batch_id", "f_signal"])
            for group in range(3):
                for i in range(6):
                    x = group * 100 + i
                    writer.writerow([f"{group}-{i}", 2 * x + 1, f"well{group}", "one-study", x])

    def test_metrics_have_known_values_and_undefined_r2(self):
        metrics = regression_metrics([0, 1, 2], [0, 2, 2])
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.5)
        self.assertAlmostEqual(metrics["mean_signed_error"], 1 / 3)
        self.assertIsNone(regression_metrics([2, 2], [1, 3])["r2"])
        for truth, pred in [([], []), ([1], [np.nan]), ([1], [1, 2])]:
            with self.assertRaises(ValueError):
                regression_metrics(truth, pred)

    def test_complete_well_holdout_training_only_scaler_and_baseline(self):
        data = load_table(self.path, ["source_well"], task="regression")
        captured = []
        original = StandardScaler.fit

        def record(scaler, x, *args, **kwargs):
            captured.append(x.copy())
            return original(scaler, x, *args, **kwargs)

        with patch.object(StandardScaler, "fit", record):
            report, predictions = evaluate_regression(data)
        self.assertEqual(len(captured), 3)
        self.assertEqual(len(predictions), 18)
        for fold, fitted in zip(report["folds"], captured):
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
            train = np.isin(data.groups, fold["train_groups"])
            np.testing.assert_array_equal(fitted, data.x[train])
            for row in predictions:
                if row["fold"] == fold["fold"]:
                    self.assertAlmostEqual(row["mean_baseline"], data.y[train].mean())
        self.assertEqual(report["unblocked_overlap_counts"]["batch_id"], [1, 1, 1])
        self.assertEqual(report["folds"][0]["feature_importance_methods"]["ridge"],
                         "absolute_standardized_coefficient_mean_across_classes")
        self.assertLess(report["models"]["ridge"]["mae"], report["models"]["mean_baseline"]["mae"])
        self.assertEqual((report, predictions), evaluate_regression(data))

    def test_target_and_metadata_never_enter_features(self):
        data = load_table(self.path, ["source_well"], task="regression")
        self.assertEqual(data.features, ["f_signal"])
        for groups, target in [(["f_signal"], "target"), (["source_well"], "f_signal"),
                               (["source_well"], "source_well")]:
            with self.assertRaises(ValueError):
                load_table(self.path, groups, task="regression", target_column=target)
        for value in ["nan", "inf", "bad", ""]:
            self.path.write_text(f"sample_id,target,source_well,f_x\na,{value},w,1\n")
            with self.assertRaises(ValueError):
                load_table(self.path, ["source_well"], task="regression")

    def test_invalid_folds_and_single_group(self):
        data = load_table(self.path, ["source_well"], task="regression")
        for folds in [1, 4, 2.5, True]:
            with self.assertRaises(ValueError):
                evaluate_regression(data, folds=folds)
        report, rows = evaluate_regression(data, folds=2)
        self.assertEqual(len(report["folds"]), 2)
        self.assertEqual(len(rows), 18)
        with self.assertRaisesRegex(ValueError, "two independent groups"):
            evaluate_regression(load_table(self.path, ["batch_id"], task="regression"))

    def test_results_are_strict_json_and_never_overwritten(self):
        data = load_table(self.path, ["source_well"], task="regression")
        data.y[:] = 2
        report, rows = evaluate_regression(data)
        out = Path(self.tmp.name) / "results"
        save_regression_results(report, rows, out)
        saved = json.loads((out / "metrics.json").read_text())
        self.assertIsNone(saved["models"]["ridge"]["r2"])
        self.assertIn("undefined", (out / "REPORT.md").read_text())
        with self.assertRaises(FileExistsError):
            save_regression_results(report, rows, out)

    def test_random_forest_regression_and_feature_importances(self):
        data = load_table(self.path, ["source_well"], task="regression")
        report, rows = evaluate_regression(data, folds=2)
        self.assertIn("random_forest", report["models"])
        self.assertLess(report["models"]["random_forest"]["mae"], 200)
        self.assertIn("mean_feature_importances", report)
        self.assertIn("random_forest", report["mean_feature_importances"])
        self.assertIn("f_signal", report["mean_feature_importances"]["random_forest"])

        # Per-fold feature importances
        for fold in report["folds"]:
            self.assertIn("feature_importances", fold)
            self.assertIn("random_forest", fold["feature_importances"])
            self.assertIn("ridge", fold["feature_importances"])

        # Verify export to disk
        out = Path(self.tmp.name) / "rf_regression_results"
        save_regression_results(report, rows, out)
        self.assertTrue((out / "feature_importances.csv").exists())
        with (out / "feature_importances.csv").open() as f:
            reader = list(csv.DictReader(f))
            self.assertGreater(len(reader), 0)
            self.assertIn("fold", reader[0])
            self.assertIn("model", reader[0])
            self.assertIn("feature", reader[0])
            self.assertIn("importance", reader[0])
            self.assertEqual({row["method"] for row in reader if row["model"] == "random_forest"},
                             {"training_impurity_decrease"})


if __name__ == "__main__":
    unittest.main()
