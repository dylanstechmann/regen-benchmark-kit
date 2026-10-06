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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regenbench import __version__
from regenbench.data import Dataset, extract_feature_importances, feature_importance_method, unblocked_overlaps


def probability_metrics(y, probabilities, classes):
    """Proper scores and descriptive top-label calibration summaries."""
    y = np.asarray(y, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and aligned to the declared class order")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each probability row must be nonnegative and sum to one")
    class_index = {label: i for i, label in enumerate(classes)}
    try:
        truth = np.eye(len(classes), dtype=float)[[class_index[label] for label in y]]
    except KeyError as exc:
        raise ValueError("a probability outcome is missing from the declared class order") from exc
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == truth.argmax(axis=1)
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = (confidence >= low) & ((confidence < high) if index < 9 else (confidence <= high))
        if selected.any():
            bins.append({"lower": low, "upper": high, "n": int(selected.sum()),
                         "mean_confidence": float(confidence[selected].mean()),
                         "observed_accuracy": float(correct[selected].mean())})
    ece = float(sum(item["n"] * abs(item["mean_confidence"] - item["observed_accuracy"])
                    for item in bins) / len(y))
    abstention = []
    for threshold in (0.5, 0.7, 0.9):
        selected = confidence >= threshold
        abstention.append({"minimum_confidence": threshold, "n_covered": int(selected.sum()),
                           "coverage": float(selected.mean()),
                           "accuracy_when_covered": float(correct[selected].mean()) if selected.any() else None})
    return {"multiclass_brier_score": float(np.mean(np.sum((probabilities - truth) ** 2, axis=1))),
            "log_loss": float(log_loss(y, probabilities, labels=classes)),
            "top_label_expected_calibration_error_10_bins": ece,
            "top_label_reliability_bins": bins,
            "confidence_coverage": abstention,
            "calibration_note": "Pooled out-of-fold descriptive bins; binning and finite sample size affect these estimates. This is not external calibration."}


def group_score_intervals(y, probabilities, groups, classes, *, seed=0, draws=2000):
    """Equal-group-weighted Brier/log-loss intervals over fixed OOF predictions."""
    y = np.asarray(y, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups)
    class_index = {label: i for i, label in enumerate(classes)}
    truth = np.eye(len(classes), dtype=float)[[class_index[label] for label in y]]
    unique = np.unique(groups)
    brier = np.array([np.mean(np.sum((probabilities[groups == group] - truth[groups == group]) ** 2, axis=1))
                      for group in unique])
    losses = np.array([-np.mean(np.log(np.clip(probabilities[groups == group, [class_index[label] for label in y[groups == group]]],
                                                1e-15, 1.0))) for group in unique])
    result = {"n_groups": len(unique), "multiclass_brier_score": {"estimate": float(brier.mean()), "ci95": None},
              "log_loss": {"estimate": float(losses.mean()), "ci95": None},
              "method": "Equal-group-weighted percentile bootstrap of fixed out-of-fold scores; conditional on fitted predictions."}
    if len(unique) >= 3:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(unique), size=(draws, len(unique)))
        result["multiclass_brier_score"]["ci95"] = np.quantile(brier[samples].mean(axis=1), [0.025, 0.975]).tolist()
        result["log_loss"]["ci95"] = np.quantile(losses[samples].mean(axis=1), [0.025, 0.975]).tolist()
    return result


def paired_group_probability_comparisons(y, probability_by_model, groups, classes,
                                        *, baseline="majority", seed=0, draws=2000):
    """Compare OOF probability scores with the baseline using paired group resamples.

    Each group's mean score difference is computed before resampling, so groups
    receive equal weight and every model comparison uses the same sampled groups.
    Negative differences favor the candidate because both scores are losses.
    """
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(y) == 0 or len(groups) != len(y) or len(unique) == 0:
        raise ValueError("outcomes and groups must be nonempty and aligned")
    if baseline not in probability_by_model:
        raise ValueError(f"baseline model {baseline!r} is unavailable")

    class_index = {label: i for i, label in enumerate(classes)}
    try:
        truth = np.eye(len(classes), dtype=float)[[class_index[label] for label in y]]
    except KeyError as exc:
        raise ValueError("a probability outcome is missing from the declared class order") from exc

    scores = {}
    for name, probabilities in probability_by_model.items():
        probability_metrics(y, probabilities, classes)
        probabilities = np.asarray(probabilities, dtype=float)
        scores[name] = {
            "multiclass_brier_score": np.asarray([
                np.mean(np.sum((probabilities[groups == group] - truth[groups == group]) ** 2, axis=1))
                for group in unique
            ]),
            "log_loss": np.asarray([
                -np.mean(np.log(np.clip(
                    probabilities[groups == group,
                                  [class_index[label] for label in y[groups == group]]],
                    1e-15, 1.0)))
                for group in unique
            ]),
        }

    result = {
        "baseline_model": baseline,
        "n_groups": len(unique),
        "method": "Paired equal-group percentile bootstrap of fixed out-of-fold score differences; conditional on fitted predictions.",
        "negative_difference_favors_candidate": True,
        "models": {},
    }
    if len(unique) >= 3:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(unique), size=(draws, len(unique)))
    else:
        samples = None
    for name, model_scores in scores.items():
        if name == baseline:
            continue
        result["models"][name] = {}
        for metric, group_values in model_scores.items():
            differences = group_values - scores[baseline][metric]
            entry = {"estimate_difference": float(differences.mean()), "ci95": None}
            if samples is not None:
                entry["ci95"] = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975]).tolist()
            result["models"][name][metric] = entry
    return result


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
    probabilities = {name: np.full((len(data.y), len(classes)), np.nan, dtype=float) for name in models}
    fold_ids = np.full(len(data.y), -1)
    fold_reports = []
    for fold, (train, test) in enumerate(partitions):
        fold_ids[test] = fold
        record = {"fold": fold, "n_train": len(train), "n_test": len(test),
                  "train_groups": sorted(set(data.groups[train])),
                  "test_groups": sorted(set(data.groups[test])), "models": {},
                  "feature_importances": {}, "feature_importance_methods": {}}
        for name, factory in models.items():
            model = factory().fit(data.x[train], data.y[train])
            pred = model.predict(data.x[test])
            predictions[name][test] = pred
            proba = model.predict_proba(data.x[test])
            model_classes = list(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
            if model_classes != classes:
                raise ValueError("model probability columns differ from the fixed class order")
            probabilities[name][test] = proba
            record["models"][name] = {**metrics(data.y[test], pred, classes),
                                       **probability_metrics(data.y[test], proba, classes)}
            imp = extract_feature_importances(model, data.features)
            if imp is not None:
                record["feature_importances"][name] = imp
                record["feature_importance_methods"][name] = feature_importance_method(model)
        fold_reports.append(record)
    warnings = [
        "OOF evaluation is development evidence. Reserve an external dataset for a final claim.",
        "Group bootstrap intervals condition on fitted predictions and exclude model-selection uncertainty.",
        "Only selected grouping columns are blocked; choose the units appropriate to the intended generalization.",
        "Feature contributions describe fitted models, not biological mechanisms. Impurity decreases can favor continuous/high-cardinality features; absolute coefficients omit direction and depend on correlated features. Values from different model types are not comparable.",
    ]
    overlaps = unblocked_overlaps(data, partitions)
    for column, counts in overlaps.items():
        if any(counts):
            warnings.append(f"Unblocked {column} values overlap train/test; this is not a held-out-{column} result.")
    mean_importances = {}
    for name in models:
        fold_imps = [f["feature_importances"][name] for f in fold_reports if name in f.get("feature_importances", {})]
        if fold_imps:
            mean_importances[name] = {feat: float(np.mean([fi[feat] for fi in fold_imps]))
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
        report["models"][name].update(probability_metrics(data.y, probabilities[name], classes))
        report["models"][name]["group_probability_scores"] = group_score_intervals(
            data.y, probabilities[name], data.groups, classes, seed=seed, draws=bootstrap_draws)
    report["paired_group_probability_comparisons"] = paired_group_probability_comparisons(
        data.y, probabilities, data.groups, classes, baseline="majority",
        seed=seed, draws=bootstrap_draws)
    rows = [{"sample_id": row["sample_id"], "label": str(data.y[i]),
             "group": str(data.groups[i]), "fold": int(fold_ids[i]),
             **{name: str(pred[i]) for name, pred in predictions.items()},
             **{f"{name}_prob_{label}": float(probabilities[name][i, class_index])
                for name in models for class_index, label in enumerate(classes)}}
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
                importance_rows.append({"fold": f["fold"], "model": mod, "feature": feat, "importance": val,
                                        "method": f.get("feature_importance_methods", {}).get(mod, "unrecorded")})
    if importance_rows:
        with (output / "feature_importances.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["fold", "model", "feature", "importance", "method"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(importance_rows)
    paired = report.get("paired_group_probability_comparisons", {}).get("models", {})

    def format_difference(value):
        if value is None:
            return "—"
        estimate = value["estimate_difference"]
        ci = value["ci95"]
        return f"{estimate:.3f}" if ci is None else f"{estimate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    lines = ["# Grouped benchmark", "", f"Input SHA-256: `{report['dataset_sha256']}`", "",
             f"{report['n_samples']} samples / {report['n_groups']} independent groups.", "",
             "| Model | Accuracy | Balanced accuracy | Macro F1 | Equal-group accuracy | Brier | Log loss | Brier Δ vs majority | Log-loss Δ vs majority |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, score in report["models"].items():
        comparison = paired.get(name, {})
        lines.append(f"| {name} | {score['accuracy']:.3f} | {score['balanced_accuracy']:.3f} | "
                     f"{score['macro_f1']:.3f} | {score['group_accuracy']['estimate']:.3f} | "
                     f"{score['multiclass_brier_score']:.3f} | {score['log_loss']:.3f} | "
                     f"{format_difference(comparison.get('multiclass_brier_score'))} | "
                     f"{format_difference(comparison.get('log_loss'))} |")
    lines.extend(["", "Probability score differences are candidate minus majority; negative values favor the candidate. "
                  "Intervals use paired, equal-group resampling of fixed out-of-fold predictions and are omitted with fewer than three groups."])
    if report.get("mean_feature_importances"):
        lines.extend(["", "## Feature importances (mean across folds)", "",
                      "| Model | Top features |", "|---|---|"])
        for mod, imps in report["mean_feature_importances"].items():
            top = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join(f"{k}: {v:.4f}" for k, v in top)
            lines.append(f"| {mod} | {top_str} |")
    lines.extend(["", "## Interpretation", "", *[f"- {w}" for w in report["warnings"]], ""])
    (output / "REPORT.md").write_text("\n".join(lines))
