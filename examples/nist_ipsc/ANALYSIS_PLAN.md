# Analysis plan: phase-image prediction of nuclear coverage

Fixed during implementation on 2026-09-24, before fitting these baselines or
inspecting their scores. This is an exploratory development benchmark, not a
registered study. Source metadata and representative image/mask crops were
inspected to establish the meaning of the files.

1. Use the three checksum-pinned `training_low`, `training_medium` and
   `training_high` archives from NIST mds2-2960. Each represents a different
   source well and seeding-density condition. No original pretrained models.
2. Sample 64 full, nonoverlapping 512 × 512 pixel tiles per well, seed 0,
   without looking at image content or masks. Drop only incomplete edge tiles.
   Keep empty/background tiles. The 192 tiles are not independent replicates.
3. Predict the fraction of pixels occupied by the binary `img_segmented.tif`
   nuclear mask. `img_fg.tif` is not the reference target. These are automated
   fluorescence-derived annotations, not manually established ground truth.
4. Use the nine existing `colonyqc` phase-image features, unchanged. Scale
   uint8 intensities by 255. No feature selection, learned image normalization,
   donor/well metadata, coordinates or mask-derived quantities as predictors.
5. Leave one complete source well out in each of three folds. Fit a mean
   baseline, standardized Ridge (alpha 1), and histogram gradient boosting
   (100 iterations, learning rate 0.05, 15 leaves, minimum 20 samples per leaf,
   L2 1, no early stopping) on the other two wells. No hyperparameter tuning.
6. Report pooled and per-well MAE, RMSE, R² and signed error. Use nuclear-area
   **percentage points** in the study summary. Include all out-of-fold
   predictions, even outside the [0, 1] target range. Report equal-well MAE;
   it equals pooled MAE here because each well supplies the same tile count.
7. As a secondary diagnostic, report MAE on tiles with nonzero reference
   nuclear area (`target > 0`) and the number of empty tiles. This subset is
   only for interpretation: it does not change fitting or the primary metric.
8. Do not report confidence intervals with three source groups, choose an
   optimized model from these results, compare area errors to the paper's
   instance-segmentation F1, or claim donor/day/laboratory generalization.

Well identity and density are confounded. A successful result would establish
only an association within this small source study. A failure is also useful:
it identifies whether the existing synthetic-image descriptors transfer at all.

Sources: [NIST dataset](https://doi.org/10.18434/mds2-2960) and
[Asmar et al., PLOS ONE (2024)](https://doi.org/10.1371/journal.pone.0298446).
