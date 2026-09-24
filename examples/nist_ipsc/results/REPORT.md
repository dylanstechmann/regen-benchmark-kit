# Grouped regression benchmark

Input SHA-256: `e055bf31bc6d08594b9010625a779d027821c2333cb369e7247c011c81158645`

192 rows / 3 source groups.

Target: `reference_nuclear_fraction`. Errors are in the target's units.

| Model | MAE | RMSE | R² | Equal-group MAE |
|---|---:|---:|---:|---:|
| mean_baseline | 0.0972 | 0.1143 | -0.5974 | 0.0972 |
| ridge | 0.0214 | 0.0330 | 0.8668 | 0.0214 |
| hist_gradient_boosting | 0.0292 | 0.0528 | 0.6590 | 0.0292 |

## Holdout groups

| Held-out group | Model | MAE | R² |
|---|---|---:|---:|
| source_well=training_low | mean_baseline | 0.1042 | -9.9038 |
| source_well=training_low | ridge | 0.0093 | 0.8850 |
| source_well=training_low | hist_gradient_boosting | 0.0107 | 0.8366 |
| source_well=training_medium | mean_baseline | 0.0603 | -0.0873 |
| source_well=training_medium | ridge | 0.0133 | 0.9471 |
| source_well=training_medium | hist_gradient_boosting | 0.0063 | 0.9826 |
| source_well=training_high | mean_baseline | 0.1272 | -2.0820 |
| source_well=training_high | ridge | 0.0417 | 0.6020 |
| source_well=training_high | hist_gradient_boosting | 0.0708 | -0.1091 |

## Interpretation

- Out-of-fold development benchmark; no hyperparameter selection was performed.
- Models and scalers fit only on each training fold. Features must already be free of upstream leakage.
- Grouping metadata defines the holdout; it does not establish biological independence.
- No confidence interval is reported. Inspect the per-group results and the number of source groups.
- Predictions are not clipped to a target range; out-of-range predictions remain visible.
- Only 3 groups: results are exploratory and cannot establish broad generalization.
