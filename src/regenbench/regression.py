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
                        "equal_group_mae": float(np.mean([m["mae"] for m in per_group.values()]))}
    overlaps = unblocked_overlaps(data, partitions)
    warnings = ["Out-of-fold development benchmark; no hyperparameter selection was performed.",
                "Models and scalers fit only on each training fold. Features must already be free of upstream leakage.",
                "Grouping metadata defines the holdout; it does not establish biological independence.",
                "No confidence interval is reported. Inspect the per-group results and the number of source groups.",
                "Predictions are not clipped to a target range; out-of-range predictions remain visible.",
                "Feature contributions describe fitted models, not biological mechanisms. Impurity decreases can favor continuous/high-cardinality features; absolute coefficients omit direction and depend on correlated features. Values from different model types are not comparable."]
    if n_groups < 5:
        warnings.append(f"Only {n_groups} groups: results are exploratory and cannot establish broad generalization.")
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
