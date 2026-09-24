# Grouped benchmark

Input SHA-256: `2a7b04d80008c1d2d8cd3e0aaaf1f0329eb029a333ba5242401a88c681476e93`

180 samples / 10 independent groups.

| Model | Accuracy | Balanced accuracy | Macro F1 | Equal-group accuracy |
|---|---:|---:|---:|---:|
| majority | 0.500 | 0.500 | 0.333 | 0.500 |
| logistic | 0.872 | 0.872 | 0.872 | 0.872 |

## Interpretation

- OOF evaluation is development evidence. Reserve an external dataset for a final claim.
- Group bootstrap intervals condition on fitted predictions and exclude model-selection uncertainty.
- Only selected grouping columns are blocked; choose the units appropriate to the intended generalization.
