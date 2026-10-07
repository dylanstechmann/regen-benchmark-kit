import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from regenbench.data import load_table
from regenbench.regression import (
    evaluate_regression,
    group_error_intervals,
    paired_group_error_comparisons,
    regression_metrics,
    save_regression_results,
)


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


class GroupedRegressionUncertaintyTests(unittest.TestCase):
    """Grouped error intervals weight groups equally and stay silent when unsupported."""

    def test_groups_weigh_equally_regardless_of_row_count(self):
        # Group "a" contributes nine rows with zero error; "b" and "c" one row each
        # with error 3. A row mean would be 0.3; the equal-group mean is 2.0.
        y = np.array([0.0] * 9 + [0.0, 0.0])
        predictions = np.array([0.0] * 9 + [3.0, 3.0])
        groups = np.array(["a"] * 9 + ["b", "c"])
        result = group_error_intervals(y, predictions, groups, seed=0)
        self.assertEqual(result["n_groups"], 3)
        self.assertAlmostEqual(result["mae"]["estimate"], 2.0)
        self.assertAlmostEqual(float(np.mean(np.abs(predictions - y))), 6 / 11)
        self.assertIn("not a row bootstrap", result["method"])

    def test_interval_requires_three_groups_and_says_why(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        predictions = np.array([1.0, 1.0, 2.0, 2.0])
        two = group_error_intervals(y, predictions, np.array(["a", "a", "b", "b"]), seed=0)
        self.assertFalse(two["interval_available"])
        self.assertIsNone(two["mae"]["ci95"])
        self.assertIn("at least 3", two["interval_unavailable_reason"])
        self.assertAlmostEqual(two["mae"]["estimate"], 1.5)

        three = group_error_intervals(y, predictions, np.array(["a", "b", "c", "c"]), seed=0)
        self.assertTrue(three["interval_available"])
        self.assertIsNone(three["interval_unavailable_reason"])
        low, high = three["mae"]["ci95"]
        self.assertLessEqual(low, three["mae"]["estimate"])
        self.assertLessEqual(three["mae"]["estimate"], high)

    def test_intervals_are_deterministic_for_a_seed(self):
        y = np.zeros(6)
        predictions = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        groups = np.array(["a", "b", "c", "d", "e", "f"])
        first = group_error_intervals(y, predictions, groups, seed=7)
        self.assertEqual(first, group_error_intervals(y, predictions, groups, seed=7))
        self.assertNotEqual(first["mae"]["ci95"],
                            group_error_intervals(y, predictions, groups, seed=8)["mae"]["ci95"])

    def test_perfect_predictions_give_a_zero_width_interval(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array(["a", "b", "c", "d"])
        result = group_error_intervals(y, y.copy(), groups, seed=0)
        self.assertEqual(result["mae"]["estimate"], 0.0)
        self.assertEqual(result["mae"]["ci95"], [0.0, 0.0])

    def test_malformed_interval_inputs_are_rejected(self):
        y = np.array([1.0, 2.0])
        for predictions, groups in (
            (np.array([1.0]), np.array(["a", "b"])),
            (np.array([1.0, 2.0]), np.array(["a"])),
            (np.array([1.0, np.nan]), np.array(["a", "b"])),
            (np.array([]), np.array([])),
        ):
            with self.subTest(predictions=predictions, groups=groups):
                with self.assertRaises(ValueError):
                    group_error_intervals(y if len(y) == len(predictions) else y, predictions, groups)

    def test_paired_comparison_uses_the_same_groups_and_excludes_the_baseline(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        groups = np.array(["a", "b", "c", "d"])
        predictions = {
            "mean_baseline": np.array([4.0, 4.0, 4.0, 4.0]),
            "ridge": np.array([1.0, 1.0, 1.0, 1.0]),
            "worse": np.array([9.0, 9.0, 9.0, 9.0]),
        }
        result = paired_group_error_comparisons(y, predictions, groups, seed=0)
        self.assertEqual(result["baseline_model"], "mean_baseline")
        self.assertNotIn("mean_baseline", result["models"])
        self.assertTrue(result["negative_difference_favors_candidate"])
        self.assertAlmostEqual(result["models"]["ridge"]["mae"]["estimate_difference"], -3.0)
        self.assertAlmostEqual(result["models"]["worse"]["mae"]["estimate_difference"], 5.0)
        low, high = result["models"]["ridge"]["mae"]["ci95"]
        self.assertLess(high, 0.0)
        self.assertLessEqual(low, high)

    def test_paired_comparison_withholds_intervals_below_three_groups(self):
        y = np.array([0.0, 0.0])
        groups = np.array(["a", "b"])
        result = paired_group_error_comparisons(
            y, {"mean_baseline": np.array([2.0, 2.0]), "ridge": np.array([1.0, 1.0])}, groups)
        self.assertFalse(result["interval_available"])
        self.assertIsNone(result["models"]["ridge"]["mae"]["ci95"])
        self.assertAlmostEqual(result["models"]["ridge"]["mae"]["estimate_difference"], -1.0)

    def test_paired_comparison_validates_baseline_and_shapes(self):
        y = np.array([0.0, 0.0, 0.0])
        groups = np.array(["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "baseline"):
            paired_group_error_comparisons(y, {"ridge": np.zeros(3)}, groups)
        with self.assertRaisesRegex(ValueError, "matched and finite"):
            paired_group_error_comparisons(
                y, {"mean_baseline": np.zeros(3), "ridge": np.array([0.0, np.nan, 0.0])}, groups)

    def test_evaluation_report_carries_intervals_and_honest_warnings(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "features.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "target", "source_well", "f_signal"])
            for group in range(4):
                for i in range(5):
                    x = group * 10 + i
                    writer.writerow([f"{group}-{i}", 2 * x + 1, f"well{group}", x])
        data = load_table(path, ["source_well"], task="regression")
        report, _rows = evaluate_regression(data, seed=0)

        intervals = report["models"]["ridge"]["group_error_intervals"]
        self.assertEqual(intervals["n_groups"], 4)
        self.assertTrue(intervals["interval_available"])
        self.assertEqual(len(intervals["mae"]["ci95"]), 2)
        self.assertEqual(len(intervals["rmse"]["ci95"]), 2)
        paired = report["paired_group_error_comparisons"]
        self.assertEqual(paired["baseline_model"], "mean_baseline")
        self.assertIn("ridge", paired["models"])
        self.assertTrue(any("not row bootstraps" in warning for warning in report["warnings"]))
        self.assertTrue(any("wide and unstable" in warning for warning in report["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_two_group_evaluation_reports_no_interval_and_explains(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "features.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "target", "source_well", "f_signal"])
            for group in range(2):
                for i in range(5):
                    x = group * 10 + i
                    writer.writerow([f"{group}-{i}", 2 * x + 1, f"well{group}", x])
        data = load_table(path, ["source_well"], task="regression")
        report, _rows = evaluate_regression(data, seed=0)
        intervals = report["models"]["ridge"]["group_error_intervals"]
        self.assertFalse(intervals["interval_available"])
        self.assertIsNone(intervals["mae"]["ci95"])
        self.assertFalse(report["paired_group_error_comparisons"]["interval_available"])
        self.assertTrue(any("no interval is reported" in warning for warning in report["warnings"]))
