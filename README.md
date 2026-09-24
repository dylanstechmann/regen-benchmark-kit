# Regen Benchmark Kit

**Turn an annotated feature table into a reproducible, group-aware baseline.**
This is the evaluation layer between a working demo and a research result:
keep related donors, plates or batches together; fit preprocessing inside each
training fold; compare a logistic model to a majority baseline; export every
out-of-fold prediction and the exact input hash.

It accepts numerical features from microscopy, organoid measurements or
preprocessed expression summaries. It does not normalize raw RNA counts or
establish that a phenotype is senescence, pluripotency or rejuvenation.

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

Use pseudonymous IDs. Keep private data outside Git. Record data license,
annotation process, preprocessing and cohort limitations in your own data card.

With `--group-by donor_id,batch_id`, rows sharing **either** unit are connected
transitively and kept together. Concatenating donor/plate IDs would not do this.
A study where every donor shares one batch becomes one component and cannot
support this joint holdout. Choose a different estimand or collect new batches;
do not manufacture independence by renaming IDs.

If you block only donors, the report explicitly counts unblocked batch overlap.
The tool cannot detect related samples when their metadata is missing or wrong.

## What is measured

- Stratified group cross-validation with a fixed seed and no hyperparameter search.
- Fold-local standardization and class-weighted logistic regression (`C=1`).
- Majority-class baseline trained independently in every fold.
- Accuracy, balanced accuracy, macro F1 and class-ordered confusion matrices.
- Equal-group-weighted accuracy with a 95% percentile bootstrap over whole groups.

The bootstrap conditions on already fitted out-of-fold predictions. It does
not refit models or include uncertainty from model selection. Very few groups
give weak intervals; fewer than three yield no interval. Pooled row metrics
and equal-group metrics answer different questions when group sizes differ.

Repeated tuning against these folds turns them into development data. Reserve
an external study or untouched test cohort for a final generalization claim.
Do feature selection, imputation and any learned normalization inside training
folds too; this toolkit cannot undo leakage already baked into input features.

## Connect the portfolio

[brightfield-colony-qc](https://github.com/dylanstechmann/brightfield-colony-qc)
exports directly compatible features with `colonyqc export-features`.
[senescence-module-score](https://github.com/dylanstechmann/senescence-module-score)
can supply exploratory summaries, but its control selection uses the supplied
cohort: **do not** pre-score an entire supervised dataset and assume fold-local
preprocessing. Fit any cohort-dependent scoring on training data separately.

See the [portfolio audit and next research milestones](docs/PORTFOLIO_REVIEW.md)
and [data/model card](docs/DATA_MODEL_CARD.md).

## Method references

- [scikit-learn: grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
- [scikit-learn: preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

MIT license. These are software benchmarks, not biological validation.
