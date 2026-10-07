# Regenerative computation project review

Review date: 2026-09-24. Scope: source and tests of the five recent Python
projects; README/tree review of the atlas and educational applications.
This is a software review, not a validation of biological claims in the cited
papers or the older atlases.

## Direction

These personal hobby and learning projects use AI coding assistance to explore
evaluation design, imaging features, experiment history, analysis provenance
and instrumentation measurements. The next development question is how the
existing methods behave on permitted, appropriately grouped real datasets and
recorded measurements. Keep software behavior, published source biology and
measured results clearly attributed.

## Existing work and changes

| Repository | What was present | This revision |
|---|---|---|
| [brightfield-colony-qc](https://github.com/dylanstechmann/brightfield-colony-qc) | Synthetic images, interpretable features, softmax baseline; 5 tests | Real image manifest import, optional PNG/TIFF, duplicate-image rejection, source hashes, benchmark-compatible feature table; 10 tests |
| [diffmedia-loop](https://github.com/dylanstechmann/diffmedia-loop) | GP/EI simulation against a synthetic response; 3 tests | Candidate-table planner, observed/pending exclusion, immutable proposal output, input hashes, finite checks, uncertainty with flat readouts; 8 tests |
| [senescence-module-score](https://github.com/dylanstechmann/senescence-module-score) | SenMayo-inspired control scoring and synthetic spike; 4 tests | Strict matrix/CSV validation, explicit control-gene mapping and seed, input digest, zero-variance effect-size handling; 8 tests |
| [cell-protocol-compiler](https://github.com/dylanstechmann/cell-protocol-compiler) | Abridged published checklists and validators; 5 tests | Finite values, unique parameter names, final-feed-to-endpoint gap, canonical protocol digest; 8 tests |
| [open-perfusion-rig](https://github.com/dylanstechmann/open-perfusion-rig) | Geometry, host simulator, Arduino sketch and carriage; 5 tests | Strict host commands, finite values, duration/rounding checks, completed-run flow reset, density-aware effective diameter and transport limitations; 9 tests |
| [geroscience-compound-atlas](https://github.com/dylanstechmann/geroscience-compound-atlas) | Compound evidence, scaffold benchmark, molecule generation, dashboard | No code changes in this pass; keep as a separate chemical-data exploration project |
| [anagen](https://github.com/dylanstechmann/anagen) | Hair/tooth research atlas | No changes; current biological and trial claims need their own source review |
| [ReaperDelay](https://github.com/dylanstechmann/ReaperDelay) | Educational browser game | No changes; distinct from the research methods projects |

## Two additions

1. **Regen Benchmark Kit:** generic numerical-feature evaluation with held-out
   groups, majority/logistic baselines, fold-local scaling, predictions, hashes
   and conditional group-bootstrap summaries. Eight tests.
2. **[Perfusion Calibration Lab](https://github.com/dylanstechmann/perfusion-calibration-lab):**
   balance-trace slope fitting, independent-repeat summaries and drift diagnostics.
   Seven tests. It analyzes measurements but does not control hardware.

All seven Python repos include installation metadata and a GitHub Actions test
matrix for Python 3.10 and 3.12. Local validation is recorded separately from
remote CI; a workflow file does not prove a successful CI run.

## Concrete next research milestones

### Follow-through: real microscopy measurement benchmark

The follow-up revision adds a checksum-pinned NIST iPSC importer to
`brightfield-colony-qc` (17 tests total) and grouped regression to this toolkit
(13 tests total). A [reproducible case study](../examples/nist_ipsc/README.md)
uses 192 tiles from three source wells. Fixed Ridge achieved 2.14 pp nuclear-area
MAE versus 9.72 pp for a training-fold mean, with worse performance on the
high-density holdout (4.17 pp). The analysis plan, source terms, numerical data,
all predictions and per-well results are committed. This is the first real
microscopy measurement evaluation in these two repositories; three wells from
one study do not satisfy the external-validation milestone below.

| Priority | Deliverable | Evidence needed |
|---|---|---|
| 1 | Collaborator imaging benchmark | A permitted, annotated dataset with donor/plate/batch identifiers; fixed preprocessing; untouched external test set |
| 2 | One complete prospective planning round | Reviewed candidate list, recorded proposals and pending IDs, returned assay measurements, predefined objective and budget |
| 3 | Pump measurement report | Independent physical repeat runs, fluid conditions, balance specification, timing method and uncertainty budget |
| 4 | Senescence scoring comparison | Public or collaborator cohort with permission; tissue/cell-type stratification; confounder checks and orthogonal assay comparison |
| 5 | Protocol review handoff | A domain expert checks every source, parameter and timing assumption; checksum ties comments to the reviewed version |

Useful next steps are domain review of the NIST case study and evaluation on
an untouched independent imaging dataset. The preliminary within-study
results motivate those checks; they do not establish a biological finding.

## Open limitations

- Image features depend on image scale and illumination. The NIST result is
  limited to nuclear area in three source wells; no biological QC classifier
  has been validated. Source hashes catch exact duplicates, not near duplicates.
- The planner maximizes one response with a fixed GP and common noise scale.
  The old broad factor box is a synthetic search space, not a jointly validated
  experimental region. The collaborator-supplied candidate list remains essential.
- Senescence controls are selected from the supplied cohort. Scores are not a
  biological-age measure, a validated classifier, or directly comparable across
  unrelated cohorts. They are not fold-independent supervised features by default.
- Protocol validation checks encoded consistency. It does not verify citations,
  establish biological correctness, or rewrite narrative steps when parameters
  are changed programmatically.
- The firmware is not queued, independently hardware-validated, or certified.
  It has no verified occlusion detection or closed-loop delivery measurement.
  Host-side validation does not harden commands sent directly to the board.

## Method references

The grouped evaluation and fold-local preprocessing design follows the
[scikit-learn grouped validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
Biological references remain attached to their original repositories; this
review does not claim to have revalidated their contents.
