"""Out-of-fold baselines with fitted preprocessing inside each training fold."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regenbench import __version__
from regenbench.data import Dataset, extract_feature_importances


def metrics(y, pred, classes):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
    }


def group_accuracy_interval(y, pred, groups, *, seed=0, draws=2000):
    """Equal-group-weighted accuracy, percentile bootstrap over whole groups.

    Conditional on the already fitted OOF predictions; does not refit models or
    estimate uncertainty from model selection. Not a cell/row bootstrap.
    """
    unique = np.unique(groups)
    scores = np.array([np.mean(y[groups == g] == pred[groups == g]) for g in unique])
    result = {"estimate": float(scores.mean()), "n_groups": len(unique), "ci95": None,
              "method": "equal-group accuracy; percentile group bootstrap of fixed OOF predictions"}
    if len(unique) >= 3:
        rng = np.random.default_rng(seed)
        boot = np.array([rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(draws)])
        result["ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
    return result


def evaluate(data: Dataset, *, folds=5, seed=0, bootstrap_draws=2000):
    if data.task != "classification":
        raise ValueError("use evaluate_regression for regression targets")
    if not isinstance(folds, int) or folds < 2 or folds > len(set(data.groups)):
        raise ValueError("folds must be between 2 and the number of independent groups")
    if not isinstance(bootstrap_draws, int) or bootstrap_draws < 100:
        raise ValueError("use at least 100 bootstrap draws")
    classes = sorted(set(data.y))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    partitions = list(splitter.split(data.x, data.y, data.groups))
    # Validate every fold before fitting; do not search seeds for a favorable score.
    for train, test in partitions:
        if set(data.groups[train]) & set(data.groups[test]):
            raise ValueError("group overlap")
        if set(data.y[train]) != set(classes) or set(data.y[test]) != set(classes):
            raise ValueError("a fold lacks a class; collect more independent groups or reduce --folds")
    models = {"majority": lambda: DummyClassifier(strategy="most_frequent"),
              "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                  C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)),
              "random_forest": lambda: RandomForestClassifier(
                  n_estimators=100, max_depth=5, class_weight="balanced_subsample", random_state=seed)}
    predictions = {name: np.empty(len(data.y), dtype=object) for name in models}
    fold_ids = np.full(len(data.y), -1)
    fold_reports = []
    for fold, (train, test) in enumerate(partitions):
        fold_ids[test] = fold
        record = {"fold": fold, "n_train": len(train), "n_test": len(test),
                  "train_groups": sorted(set(data.groups[train])),
                  "test_groups": sorted(set(data.groups[test])), "models": {},
                  "feature_importances": {}}
        for name, factory in models.items():
            model = factory().fit(data.x[train], data.y[train])
            pred = model.predict(data.x[test])
            predictions[name][test] = pred
            record["models"][name] = metrics(data.y[test], pred, classes)
            imp = extract_feature_importances(model, data.features)
            if imp is not None:
                record["feature_importances"][name] = imp
        fold_reports.append(record)
    warnings = [
        "OOF evaluation is development evidence. Reserve an external dataset for a final claim.",
        "Group bootstrap intervals condition on fitted predictions and exclude model-selection uncertainty.",
        "Only selected grouping columns are blocked; choose the units appropriate to the intended generalization.",
    ]
    overlaps = {}
    for column in ["donor_id", "batch_id", "plate_id", "group_id"]:
        if column in data.rows[0] and column not in data.group_columns:
            counts = []
            for train, test in partitions:
                a = {data.rows[i][column] for i in train} - {""}
                b = {data.rows[i][column] for i in test} - {""}
                counts.append(len(a & b))
            overlaps[column] = counts
            if any(counts):
                warnings.append(f"Unblocked {column} values overlap train/test; this is not a held-out-{column} result.")
    mean_importances = {}
    for name in models:
        fold_imps = [f["feature_importances"][name] for f in fold_reports if name in f.get("feature_importances", {})]
        if fold_imps:
            mean_importances[name] = {feat: round(float(np.mean([fi[feat] for fi in fold_imps])), 6)
                                      for feat in data.features}
    report = {
        "schema_version": 1, "dataset_sha256": data.sha256,
        "configuration": {"folds": folds, "seed": seed, "group_by": data.group_columns,
                          "bootstrap_draws": bootstrap_draws, "features": data.features,
                          "logistic_C": 1.0, "logistic_class_weight": "balanced",
                          "random_forest": {"n_estimators": 100, "max_depth": 5}},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "regenbench": __version__},
        "n_samples": len(data.y), "n_groups": len(set(data.groups)), "classes": classes,
        "models": {}, "folds": fold_reports, "mean_feature_importances": mean_importances,
        "unblocked_overlap_counts": overlaps, "warnings": warnings,
    }
    for name, pred in predictions.items():
        report["models"][name] = metrics(data.y, pred, classes)
        report["models"][name]["group_accuracy"] = group_accuracy_interval(
            data.y, pred, data.groups, seed=seed, draws=bootstrap_draws)
    rows = [{"sample_id": row["sample_id"], "label": str(data.y[i]),
             "group": str(data.groups[i]), "fold": int(fold_ids[i]),
             **{name: str(pred[i]) for name, pred in predictions.items()}}
            for i, row in enumerate(data.rows)]
    return report, rows


def save_results(report, predictions, output):
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
                importance_rows.append({"fold": f["fold"], "model": mod, "feature": feat, "importance": val})
    if importance_rows:
        with (output / "feature_importances.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fold", "model", "feature", "importance"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(importance_rows)
    lines = ["# Grouped benchmark", "", f"Input SHA-256: `{report['dataset_sha256']}`", "",
             f"{report['n_samples']} samples / {report['n_groups']} independent groups.", "",
             "| Model | Accuracy | Balanced accuracy | Macro F1 | Equal-group accuracy |",
             "|---|---:|---:|---:|---:|"]
    for name, score in report["models"].items():
        lines.append(f"| {name} | {score['accuracy']:.3f} | {score['balanced_accuracy']:.3f} | "
                     f"{score['macro_f1']:.3f} | {score['group_accuracy']['estimate']:.3f} |")
    if report.get("mean_feature_importances"):
        lines.extend(["", "## Feature importances (mean across folds)", "",
                      "| Model | Top features |", "|---|---|"])
        for mod, imps in report["mean_feature_importances"].items():
            top = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{k}: {v:.4f}" for k, v in top)
            lines.append(f"| {mod} | {top_str} |")
    lines.extend(["", "## Interpretation", "", *[f"- {w}" for w in report["warnings"]], ""])
    (output / "REPORT.md").write_text("\n".join(lines))
