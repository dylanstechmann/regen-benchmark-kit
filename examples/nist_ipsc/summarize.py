"""Build a study-specific report and figure from saved out-of-fold predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regenbench.regression import regression_metrics


def summarize(data_dir, results_dir):
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    destinations = [results_dir / name for name in ["STUDY_REPORT.md", "diagnostics.json", "predictions.png"]]
    if any(path.exists() for path in destinations):
        raise FileExistsError("study summary outputs already exist; use a new results directory")
    metrics = json.loads((results_dir / "metrics.json").read_text())
    provenance = json.loads((data_dir / "provenance.json").read_text())
    digest = hashlib.sha256((data_dir / "features.csv").read_bytes()).hexdigest()
    if digest != metrics["dataset_sha256"] or digest != provenance["features_sha256"]:
        raise ValueError("feature table, provenance and model results do not match")
    with (data_dir / "features.csv").open() as handle:
        features = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with (results_dir / "predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != len(features) or {row["sample_id"] for row in predictions} != set(features):
        raise ValueError("predictions do not match the input samples one-to-one")
    y = np.array([float(row["target"]) for row in predictions])
    wells = np.array([features[row["sample_id"]]["source_well"] for row in predictions])
    expected_y = np.array([float(features[row["sample_id"]]["reference_nuclear_fraction"]) for row in predictions])
    if not np.array_equal(y, expected_y):
        raise ValueError("saved targets do not match the input table")
    names = {"mean_baseline": "Training-fold mean", "ridge": "Standardized Ridge",
             "hist_gradient_boosting": "Histogram gradient boosting"}
    estimates = {name: np.array([float(row[name]) for row in predictions]) for name in names}
    well_order = ["training_low", "training_medium", "training_high"]
    diagnostics = {"dataset_sha256": digest, "secondary_subset": "reference_nuclear_fraction > 0",
                   "n_nonempty": int((y > 0).sum()), "models": {}, "wells": {}}
    for name, pred in estimates.items():
        diagnostics["models"][name] = {
            "nonempty_tiles": regression_metrics(y[y > 0], pred[y > 0]),
            "outside_target_range_count": int(((pred < 0) | (pred > 1)).sum()),
            "prediction_min": float(pred.min()), "prediction_max": float(pred.max())}
    for well in well_order:
        target = y[wells == well]
        diagnostics["wells"][well] = {"n_tiles": len(target), "n_empty": int((target == 0).sum()),
                                      "mean_target": float(target.mean()), "min_target": float(target.min()),
                                      "max_target": float(target.max())}

    lines = ["# NIST iPSC nuclear-area benchmark", "",
        "**192 phase-contrast tiles from three source wells, with each whole well held out in turn.**",
        "The outcome is fluorescence-derived nuclear mask area. This is an exploratory imaging measurement benchmark.", "",
        "## Primary results", "", "Errors are **percentage points of tile area**, not relative percentage errors.", "",
        "| Fixed model | MAE (pp) | RMSE (pp) | R² |", "|---|---:|---:|---:|"]
    for name in names:
        score = metrics["models"][name]
        lines.append(f"| {names[name]} | {100 * score['mae']:.2f} | {100 * score['rmse']:.2f} | {score['r2']:.3f} |")
    lines.extend(["", "Pooled and equal-well MAE agree because each well contributes 64 tiles.",
                  "These are all three prespecified baselines; no hyperparameter search was performed.", "",
                  "![Out-of-fold predictions colored by held-out source well](predictions.png)", "",
                  "Each dot is a tile predicted by a model that did not train on its source well.",
                  "The dashed line is equality. Tiles within a well are related measurements.", "",
                  "## Errors by held-out well", "",
                  "| Source well | Mean baseline MAE (pp) | Ridge MAE (pp) | Boosting MAE (pp) |",
                  "|---|---:|---:|---:|"])
    for well in well_order:
        errors = [100 * regression_metrics(y[wells == well], pred[wells == well])["mae"]
                  for pred in estimates.values()]
        lines.append(f"| {well} | " + " | ".join(f"{e:.2f}" for e in errors) + " |")
    lines.extend(["", "The high-density well is the hardest holdout for both learned models. Density and",
        "well identity cannot be separated in this design. Strong pooled performance does",
        "not establish robustness across laboratories, donors, acquisition days or density extremes.", "",
        "## Background diagnostic", "",
        "| Source well | Tiles | Empty nuclear masks | Mean nuclear area (%) | Observed range (%) |",
        "|---|---:|---:|---:|---|"])
    for well, info in diagnostics["wells"].items():
        lines.append(f"| {well} | {info['n_tiles']} | {info['n_empty']} | {100 * info['mean_target']:.2f} | "
                     f"{100 * info['min_target']:.2f}–{100 * info['max_target']:.2f} |")
    lines.extend(["", f"The prespecified secondary subset contains {diagnostics['n_nonempty']} tiles with nonzero nuclear area.",
                  "Models were fitted on all training tiles; this diagnostic does not change the primary analysis.", "",
                  "| Model | Nonempty-tile MAE (pp) | Predictions outside [0, 1] |", "|---|---:|---:|"])
    for name, info in diagnostics["models"].items():
        lines.append(f"| {names[name]} | {100 * info['nonempty_tiles']['mae']:.2f} | {info['outside_target_range_count']} |")
    lines.extend(["", "Predictions are not clipped, including negative predictions. Low area error is not",
        "an instance-segmentation score or a cell-count error.", "", "## What this supports", "",
        "The existing nine phase-image descriptors carry information about the provided nuclear-area",
        "reference within this study. The per-well failures and saved predictions are the starting point",
        "for a collaborator review. No senescence, differentiation, pluripotency, viability or clinical",
        "claim follows from this measurement. Reference masks were generated automatically from fluorescence.", "",
        "There are only **three source groups**. No confidence interval, significance test or external",
        "validation claim is provided. Further tuning on these folds would make them development data",
        "for that tuning; an independent dataset is needed for a subsequent generalization claim.", "",
        "## Reproducibility", "", f"Feature table SHA-256: `{digest}`.", "",
        "- [Analysis plan](../ANALYSIS_PLAN.md), fixed before fitting these models.",
        "- [Data card and rerun commands](../README.md).",
        "- [Source archive and tile provenance](../data/provenance.json).",
        "- [Exact configuration, versions and fold metrics](metrics.json).",
        "- [Every out-of-fold prediction](predictions.csv) and [secondary diagnostics](diagnostics.json).", "",
        "Data: [NIST mds2-2960](https://doi.org/10.18434/mds2-2960), accompanying",
        "[Asmar et al. (2024)](https://doi.org/10.1371/journal.pone.0298446).",
        "Derived analysis dated 2026-09-24; [attribution and source terms](../SOURCE_NOTICE.md).",
        "NIST did not produce or endorse these benchmark results.", ""])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.3), sharex=True, sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "^"]
    lo = min(0.0, *(float(pred.min() * 100) for pred in estimates.values())) - 1
    hi = max(float(y.max() * 100), *(float(pred.max() * 100) for pred in estimates.values())) + 1
    for ax, (name, pred) in zip(axes, estimates.items()):
        for well, color, marker in zip(well_order, colors, markers):
            mask = wells == well
            ax.scatter(y[mask] * 100, pred[mask] * 100, s=19, alpha=0.7, c=color,
                       marker=marker, edgecolors="none", label=well.removeprefix("training_") + " density well")
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Reference nuclear area (%)",
               title=f"{names[name]}\nMAE {100 * metrics['models'][name]['mae']:.2f} pp")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Predicted nuclear area (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Phase-contrast features → nuclear area: leave one source well out", fontsize=14, y=0.99)
    fig.text(0.5, 0.012, "NIST mds2-2960 • 192 tiles / 3 wells • Exploratory within-study analysis; no confidence intervals",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.20, 1, 0.94))
    fig.savefig(destinations[2], dpi=180, facecolor="white")
    plt.close(fig)
    destinations[1].write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    destinations[0].write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    summarize(args.data, args.results)
