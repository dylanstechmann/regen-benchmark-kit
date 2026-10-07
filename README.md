# Regen Benchmark Kit

This is a personal hobby and learning project, developed with substantial
assistance from AI coding tools.

**Turn an annotated feature table into a reproducible, group-aware baseline.**
This is the evaluation layer between a working demo and a research result:
keep related donors, plates or batches together; fit preprocessing inside each
training fold; compare fixed classification or regression baselines; export every
out-of-fold prediction and the exact input hash.
The loader parses the same CSV bytes it hashes, so a file changed during
loading cannot produce a report whose hash identifies different input data.

`regenbench leakage-check --out artifacts/leakage.json` scores a synthetic donor-tag fixture twice: a random row split, which can see the same donor in train and test, and the group holdout, which cannot. The gap is a leak detector. It is not a biological metric and it is not computed on the NIST table.

It accepts numerical features from microscopy, organoid measurements or
preprocessed expression summaries. It does not normalize raw RNA counts or
establish that a phenotype is senescence, pluripotency or rejuvenation.

## Real iPSC example

The [NIST phase-image benchmark](examples/nist_ipsc/results/STUDY_REPORT.md)
predicts fluorescence-derived nuclear area from nine fixed image descriptors.
It uses 192 tiles from three source wells, leaving an entire well out each time.
Ridge achieved **2.14 percentage-point MAE**, compared with **9.72** for the
training-fold mean. The high-density well had larger errors (Ridge: **4.17 pp**).
This is an exploratory measurement benchmark with three wells, not external
biological validation. Data, provenance, every prediction, the analysis plan,
and [rerun instructions](examples/nist_ipsc/README.md) are included.

## Run in five minutes

Python 3.10+, NumPy and scikit-learn; CPU only.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
regenbench run examples/synthetic_features.csv \
  --group-by donor_id,batch_id --folds 5 --seed 0 --out artifacts/demo
```

Open `artifacts/demo/REPORT.md` for the summary, `metrics.json` for full fold
metrics and software versions, and `predictions.csv` to inspect errors.
Output directories must be new: earlier results are never silently overwritten.
The checked-in [example report](examples/results/REPORT.md) is **synthetic**.

Generate another software fixture with:

```bash
regenbench demo --seed 42 --out artifacts/new_synthetic.csv
```

## Bring a collaborator's data

```csv
sample_id,label,donor_id,batch_id,f_area,f_texture
field001,compact,donor01,batch01,0.25,0.07
field002,irregular,donor01,batch01,0.17,0.12
```

This two-row illustration is too small to evaluate. Use enough independent
groups that every training and test fold contains every class.

| Column | Meaning |
|---|---|
| `sample_id` | Unique record; never used as a feature |
| `label` | Collaborator-supplied annotation |
| Selected `--group-by` columns | Nonempty donor, batch, plate or other experimental units |
| `f_*` | Finite numerical features; only these enter the model |
| `image_sha256` (optional) | Unique source-image checksum; duplicate images are rejected |

When provided, `image_sha256` must be a 64-character hexadecimal SHA-256.
Checks for duplicate source images ignore letter case, so the same digest
cannot be placed in separate groups by changing its capitalization. The tool
checks supplied digests; it does not read or verify the underlying images.

Use pseudonymous IDs. Keep private data outside Git. Record data license,
annotation process, preprocessing and cohort limitations in your own data card.

With `--group-by donor_id,batch_id`, rows sharing **either** unit are connected
transitively and kept together. Concatenating donor/plate IDs would not do this.
A study where every donor shares one batch becomes one component and cannot
support this joint holdout. Choose a different estimand or collect new batches;
do not manufacture independence by renaming IDs.

If you block only donors, the report explicitly counts unblocked batch overlap.
Both tasks also check plate, group, acquisition-day and source-well metadata
when present. Shared unblocked values produce explicit holdout warnings.
The tool cannot detect related samples when their metadata is missing or wrong.

## What is measured

### Classification (`regenbench run`)

- Stratified group cross-validation with a fixed seed and no hyperparameter search.
- Fold-local standardization and class-weighted logistic regression (`C=1`).
- Random forest classifier baseline (`n_estimators=100`, `max_depth=5`, balanced subsampling).
- Majority-class baseline trained independently in every fold.
- Accuracy, balanced accuracy, macro F1, class-ordered confusion matrices,
  class-ordered out-of-fold probabilities, multiclass Brier score and log loss.
- Top-label 10-bin reliability summaries and fixed 0.5/0.7/0.9
  confidence-versus-coverage summaries for review workflows.
- Equal-group-weighted accuracy with a 95% percentile bootstrap over whole groups.
- Equal-group-weighted Brier/log-loss bootstrap intervals when at least three
  independent groups are available.
- Paired equal-group bootstrap differences in Brier/log loss against the
  majority baseline, using the same resampled groups for each comparison.
- Per-fold feature importance tracking and CSV export (`feature_importances.csv`) for tree ensembles and standardized linear coefficients.

The bootstrap conditions on already fitted out-of-fold predictions. It does
not refit models or include uncertainty from model selection. Very few groups
give weak intervals; fewer than three yield no interval. Pooled row metrics
and equal-group metrics answer different questions when group sizes differ.
Probability metrics retain every out-of-fold probability in `predictions.csv`.
Reliability bins and coverage summaries are descriptive estimates on the same
development folds; they do not calibrate the model for a new study or replace
an external test set. A majority baseline naturally has low coverage at higher
confidence thresholds when its only class probability is below the threshold.
Paired probability-score differences are conditional on the fixed out-of-fold
predictions; negative values favor the candidate because lower scores are
better. Intervals are omitted with fewer than three groups. Classification
probability features are exercised on the synthetic software fixture and do
not establish performance on biological outcomes.

### Regression (`regenbench regress`)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 regenbench regress \
  examples/nist_ipsc/data/features.csv --target reference_nuclear_fraction \
  --group-by source_well --seed 0 --out artifacts/nist
```

The target defaults to a numeric `target` column; `--target` selects another
column. Only `f_*` columns enter predictors. Targets and grouping columns cannot
also be feature columns. Targets must be finite, and at least two source groups
are required. Leave-one-group-out is the default; `--folds N` uses GroupKFold.

- Training-fold mean, standardized Ridge (`alpha=1`), fixed histogram gradient boosting, and random forest regressor (`n_estimators=100`, `max_depth=5`).
- MAE, RMSE, R², signed error, per-group errors and equal-group MAE.
- Per-fold feature importance export (`feature_importances.csv`) and top feature summary in `REPORT.md`.
- No target-range clipping; negative predictions remain visible.
- Equal-group-weighted MAE and RMSE with a 95% percentile bootstrap over whole
  groups, reported per model as `group_error_intervals`. Each group's error is
  computed before resampling, so a group with nine rows and a group with one row
  weigh the same. Fewer than three groups reports no interval and records the
  reason instead.
- Paired equal-group bootstrap differences against the training-fold mean
  baseline (`paired_group_error_comparisons`), using the same resampled groups
  for every model. Negative values favor the candidate because both quantities
  are errors.
- Constant-target R² is reported as undefined.

The regression bootstrap conditions on the already fitted out-of-fold
predictions. It does not refit models, carries no uncertainty from model
selection, and is not a row bootstrap: resampling rows inside a well would
describe that well, not a new well. With only a few groups the interval is wide
and unstable, and the report says so; read it as a spread across the available
groups, not a population interval. The committed `examples/nist_ipsc/results/`
snapshot predates these fields — re-running the command below emits them, and the
three-well metrics are unchanged.

Regression uses the same connected grouping and input provenance as
classification. The report records settings, versions, folds and unblocked
metadata overlaps. Fixed models are baselines, not tuned recommendations.

`feature_importances.csv` labels each quantity in its `method` column and keeps
full numeric precision. Tree values are training impurity decreases; linear
values are absolute standardized coefficients (averaged across class rows).
These quantities are not comparable across model types and do not identify
biological mechanisms. Impurity measures can favor continuous or high-cardinality
predictors; correlated predictors can obscure coefficient interpretation.
See the [scikit-learn inspection example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html).

Repeated tuning against these folds turns them into development data. Reserve
an external study or untouched test cohort for a final generalization claim.
Do feature selection, imputation and any learned normalization inside training
folds too; this toolkit cannot undo leakage already baked into input features.

## Related projects

[brightfield-colony-qc](https://github.com/dylanstechmann/brightfield-colony-qc)
exports directly compatible features with `colonyqc export-features`.
[senescence-module-score](https://github.com/dylanstechmann/senescence-module-score)
can supply exploratory summaries, but its control selection uses the supplied
cohort: **do not** pre-score an entire supervised dataset and assume fold-local
preprocessing. Fit any cohort-dependent scoring on training data separately.

See the [project review and next research milestones](docs/PORTFOLIO_REVIEW.md)
and [data/model card](docs/DATA_MODEL_CARD.md).

## Method references

- [scikit-learn: grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
- [scikit-learn: preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

Original code: MIT. NIST-derived example data retain their
[source notice](examples/nist_ipsc/SOURCE_NOTICE.md). These benchmarks do not
establish biological identity or clinical validity.
