"""Fixed regression baselines with complete experimental groups held out."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regenbench import __version__
from regenbench.data import Dataset, extract_feature_importances, feature_importance_method, unblocked_overlaps


def regression_metrics(y, predictions):
    y, predictions = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    if (y.ndim != 1 or not y.size or y.shape != predictions.shape
            or not np.isfinite(y).all() or not np.isfinite(predictions).all()):
        raise ValueError("metrics require matched, nonempty finite target/prediction vectors")
    return {"mae": float(mean_absolute_error(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "r2": None if len(y) < 2 or np.ptp(y) == 0 else float(r2_score(y, predictions)),
            "mean_signed_error": float(np.mean(predictions - y))}


def group_error_intervals(y, predictions, groups, *, seed=0, draws=2000):
    """Equal-group-weighted MAE/RMSE intervals over fixed out-of-fold predictions.

    Each group's error is computed first, so groups receive equal weight however
    many rows they contributed. The bootstrap resamples whole groups and is
    conditional on the already fitted predictions: it does not refit models and
    carries no uncertainty from model selection. Fewer than three groups yields no
    interval, because resampling two groups describes nothing.
    """
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    groups = np.asarray(groups)
    if y.ndim != 1 or y.shape != predictions.shape or len(groups) != len(y) or not y.size:
        raise ValueError("intervals require matched, nonempty target/prediction/group vectors")
    if not np.isfinite(y).all() or not np.isfinite(predictions).all():
        raise ValueError("intervals require finite targets and predictions")
    unique = np.unique(groups)
    absolute = np.array([np.mean(np.abs(predictions[groups == group] - y[groups == group]))
                         for group in unique])
    squared = np.array([np.sqrt(np.mean((predictions[groups == group] - y[groups == group]) ** 2))
                        for group in unique])
    result = {"n_groups": len(unique),
              "mae": {"estimate": float(absolute.mean()), "ci95": None},
              "rmse": {"estimate": float(squared.mean()), "ci95": None},
              "method": "Equal-group-weighted percentile bootstrap of fixed out-of-fold errors; "
                        "conditional on fitted predictions, not a row bootstrap.",
              "interval_available": False,
              "interval_unavailable_reason": None}
    if len(unique) >= 3:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(unique), size=(draws, len(unique)))
        result["mae"]["ci95"] = np.quantile(absolute[samples].mean(axis=1), [0.025, 0.975]).tolist()
        result["rmse"]["ci95"] = np.quantile(squared[samples].mean(axis=1), [0.025, 0.975]).tolist()
        result["interval_available"] = True
    else:
        result["interval_unavailable_reason"] = (
            f"{len(unique)} independent group(s): at least 3 are required before resampling groups "
            "describes anything.")
    return result


def paired_group_error_comparisons(y, prediction_by_model, groups, *, baseline="mean_baseline",
                                   seed=0, draws=2000):
    """Compare each model's grouped error with the baseline using paired group resamples.

    Each group's error difference is computed before resampling, so groups weigh
    equally and every model comparison uses the same sampled groups. Negative
    differences favor the candidate because both quantities are errors.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not y.size or len(groups) != len(y) or not len(unique):
        raise ValueError("outcomes and groups must be nonempty and aligned")
    if baseline not in prediction_by_model:
        raise ValueError(f"baseline model {baseline!r} is unavailable")

    scores = {}
    for name, predictions in prediction_by_model.items():
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != y.shape or not np.isfinite(predictions).all():
            raise ValueError(f"model {name!r} predictions must be matched and finite")
        scores[name] = {
            "mae": np.array([np.mean(np.abs(predictions[groups == group] - y[groups == group]))
                             for group in unique]),
            "rmse": np.array([np.sqrt(np.mean((predictions[groups == group] - y[groups == group]) ** 2))
                              for group in unique]),
        }

    result = {"baseline_model": baseline, "n_groups": len(unique),
              "method": "Paired equal-group percentile bootstrap of fixed out-of-fold error differences; "
                        "conditional on fitted predictions.",
              "negative_difference_favors_candidate": True,
              "interval_available": len(unique) >= 3,
              "models": {}}
    samples = None
    if len(unique) >= 3:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(unique), size=(draws, len(unique)))
    for name, model_scores in scores.items():
        if name == baseline:
            continue
        result["models"][name] = {}
        for metric, values in model_scores.items():
            differences = values - scores[baseline][metric]
            entry = {"estimate_difference": float(differences.mean()), "ci95": None}
            if samples is not None:
                entry["ci95"] = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975]).tolist()
            result["models"][name][metric] = entry
    return result


def evaluate_regression(data: Dataset, *, folds=None, seed=0):
    if data.task != "regression":
        raise ValueError("load the table with task='regression'")
    n_groups = len(set(data.groups))
    if n_groups < 2:
        raise ValueError("regression evaluation needs at least two independent groups")
    if folds is not None and (isinstance(folds, bool) or not isinstance(folds, int)
                              or not 2 <= folds <= n_groups):
        raise ValueError("folds must be between 2 and the number of groups")
    splitter = LeaveOneGroupOut() if folds is None else GroupKFold(n_splits=folds)
    partitions = list(splitter.split(data.x, data.y, data.groups))
    factory = {
        "mean_baseline": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=seed),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=seed),
    }
    predictions = {name: np.full(len(data.y), np.nan) for name in factory}
    fold_ids = np.full(len(data.y), -1)
    records = []
    for fold, (train, test) in enumerate(partitions):
        if set(data.groups[train]) & set(data.groups[test]):
            raise ValueError("group overlap")
        fold_ids[test] = fold
        record = {"fold": fold, "n_train": len(train), "n_test": len(test),
                  "train_groups": sorted(set(data.groups[train])),
                  "test_groups": sorted(set(data.groups[test])), "models": {},
                  "feature_importances": {}, "feature_importance_methods": {}}
        for name, create in factory.items():
            model = create().fit(data.x[train], data.y[train])
            pred = model.predict(data.x[test])
            record["models"][name] = regression_metrics(data.y[test], pred)
            predictions[name][test] = pred
            imp = extract_feature_importances(model, data.features)
            if imp is not None:
                record["feature_importances"][name] = imp
                record["feature_importance_methods"][name] = feature_importance_method(model)
        records.append(record)
    group_metadata = {str(group): {column: sorted({row[column] for i, row in enumerate(data.rows)
                                                   if data.groups[i] == group})
                                   for column in data.group_columns}
                      for group in np.unique(data.groups)}
    models = {}
    for name, pred in predictions.items():
        per_group = {str(group): regression_metrics(data.y[data.groups == group], pred[data.groups == group])
                     for group in np.unique(data.groups)}
        models[name] = {**regression_metrics(data.y, pred), "per_group": per_group,
                        "equal_group_mae": float(np.mean([m["mae"] for m in per_group.values()])),
                        "group_error_intervals": group_error_intervals(
                            data.y, pred, data.groups, seed=seed)}
    paired_errors = paired_group_error_comparisons(data.y, predictions, data.groups, seed=seed)
    overlaps = unblocked_overlaps(data, partitions)
    warnings = ["Out-of-fold development benchmark; no hyperparameter selection was performed.",
                "Models and scalers fit only on each training fold. Features must already be free of upstream leakage.",
                "Grouping metadata defines the holdout; it does not establish biological independence.",
                "Equal-group MAE/RMSE intervals resample whole groups over fixed out-of-fold predictions. They are conditional on those predictions, exclude uncertainty from model selection, and are not row bootstraps.",
                "Paired differences against the training-fold mean baseline use the same resampled groups for every model; negative values favor the candidate because both quantities are errors.",
                "Predictions are not clipped to a target range; out-of-range predictions remain visible.",
                "Feature contributions describe fitted models, not biological mechanisms. Impurity decreases can favor continuous/high-cardinality features; absolute coefficients omit direction and depend on correlated features. Values from different model types are not comparable."]
    if n_groups < 3:
        warnings.append(f"Only {n_groups} groups: no interval is reported, because resampling fewer than three groups describes nothing.")
    if n_groups < 5:
        warnings.append(f"Only {n_groups} groups: results are exploratory and cannot establish broad generalization.")
        warnings.append("With few groups a group bootstrap interval is wide and unstable; read it as a spread across the available groups, not a population interval.")
    for column, counts in overlaps.items():
        if any(counts):
            warnings.append(f"Unblocked {column} overlaps train/test; this is not a held-out-{column} result.")
    mean_importances = {}
    for name in factory:
        fold_imps = [r["feature_importances"][name] for r in records if name in r.get("feature_importances", {})]
        if fold_imps:
            mean_importances[name] = {feat: float(np.mean([fi[feat] for fi in fold_imps]))
                                      for feat in data.features}
    report = {"schema_version": 1, "task": "regression", "dataset_sha256": data.sha256,
              "n_samples": len(data.y), "n_groups": n_groups, "group_metadata": group_metadata,
              "configuration": {"split": "leave_one_group_out" if folds is None else "group_k_fold",
                                "folds": len(records), "seed": seed, "group_by": data.group_columns,
                                "target": data.target_column, "features": data.features,
                                "ridge_alpha": 1.0,
                                "hist_gradient_boosting": {"learning_rate": 0.05, "max_iter": 100,
                                    "max_leaf_nodes": 15, "min_samples_leaf": 20,
                                    "l2_regularization": 1.0, "early_stopping": False},
                                "random_forest": {"n_estimators": 100, "max_depth": 5}},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__, "regenbench": __version__},
              "models": models, "folds": records, "mean_feature_importances": mean_importances,
              "paired_group_error_comparisons": paired_errors,
              "unblocked_overlap_counts": overlaps, "warnings": warnings}
    rows = [{"sample_id": row["sample_id"], "target": float(data.y[i]),
             "group": str(data.groups[i]), "fold": int(fold_ids[i]),
             **{name: float(pred[i]) for name, pred in predictions.items()}}
            for i, row in enumerate(data.rows)]
    return report, rows


def save_regression_results(report, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    importance_rows = []
    for f in report.get("folds", []):
        for mod, imps in f.get("feature_importances", {}).items():
            for feat, val in imps.items():
                importance_rows.append({"fold": f["fold"], "model": mod, "feature": feat, "importance": val,
                                        "method": f.get("feature_importance_methods", {}).get(mod, "unrecorded")})
    if importance_rows:
        with (output / "feature_importances.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fold", "model", "feature", "importance", "method"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(importance_rows)
    lines = ["# Grouped regression benchmark", "", f"Input SHA-256: `{report['dataset_sha256']}`", "",
             f"{report['n_samples']} rows / {report['n_groups']} source groups.", "",
             "Target: `" + report["configuration"]["target"] + "`. Errors are in the target's units.", "",
             "| Model | MAE | RMSE | R² | Equal-group MAE |", "|---|---:|---:|---:|---:|"]
    for name, score in report["models"].items():
        r2 = "undefined" if score["r2"] is None else f"{score['r2']:.4f}"
        lines.append(f"| {name} | {score['mae']:.4f} | {score['rmse']:.4f} | {r2} | {score['equal_group_mae']:.4f} |")
    lines.extend(["", "## Holdout groups", "", "| Held-out group | Model | MAE | R² |",
                  "|---|---|---:|---:|"])
    for group, metadata in report["group_metadata"].items():
        label = "; ".join(f"{key}={','.join(values)}" for key, values in metadata.items())
        for name, model in report["models"].items():
            score = model["per_group"][group]
            r2 = "undefined" if score["r2"] is None else f"{score['r2']:.4f}"
            lines.append(f"| {label} | {name} | {score['mae']:.4f} | {r2} |")
    if report.get("mean_feature_importances"):
        lines.extend(["", "## Feature importances (mean across folds)", "",
                      "| Model | Top features |", "|---|---|"])
        for mod, imps in report["mean_feature_importances"].items():
            top = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{k}: {v:.4f}" for k, v in top)
            lines.append(f"| {mod} | {top_str} |")
    lines.extend(["", "## Interpretation", "", *[f"- {w}" for w in report["warnings"]], ""])
    (output / "REPORT.md").write_text("\n".join(lines))
