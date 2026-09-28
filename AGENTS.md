# Agent instructions — regen-benchmark-kit

Work only in this repository. This is grouped cross-validation for feature
tables. The only real table is the NIST three-well phase-image regression
(192 tiles). The synthetic classification CSV is a software fixture.

## Do not

- Add a biological claim that nuclear-area MAE means identity, potency, or senescence.
- Tune hyperparameters against the reported folds and then quote the same folds as confirmation.
- Concatenate donor and batch IDs into one group column. The connected-component rule is the point.
- Train on `brightfield-colony-qc` synthetic labels and call it iPSC QC.
- Invent a fourth well.

## First commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Re-run the NIST regression only if you changed regression code:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 regenbench regress \
  examples/nist_ipsc/data/features.csv --target reference_nuclear_fraction \
  --group-by source_well --seed 0 --out artifacts/nist-rerun
```

Use a new output directory. Do not overwrite `examples/nist_ipsc/results/` unless the numbers changed and you update `STUDY_REPORT.md` to match.

## Improve, in this order

1. If a metric, hash, or group rule disagrees with `README.md`, fix the disagreement and add a test.
2. `regenbench leakage-check` is the random-row versus group-holdout diagnostic. It runs only on a synthetic donor-tag fixture. Do not point it at the NIST table or quote it as a biological metric.
3. Do not add a new model class unless the NIST and synthetic baselines still run unchanged.

## Done when

Tests pass and any quoted NIST number matches a report you just generated.
