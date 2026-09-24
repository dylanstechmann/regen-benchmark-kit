# Real iPSC phase-image benchmark

Can nine simple descriptors of a phase-contrast image predict its nuclear mask
area when the entire source well is withheld from training?

Read the [results and figure](results/STUDY_REPORT.md), the
[analysis plan](ANALYSIS_PLAN.md), and the [source notice](SOURCE_NOTICE.md).
This tests a continuous imaging measurement. It does not validate the four
synthetic morphology labels in `brightfield-colony-qc`.

## Rerun the benchmark without downloading images

From the root of `regen-benchmark-kit`:

```bash
python -m pip install -e '.[plots]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m regenbench.cli regress \
  examples/nist_ipsc/data/features.csv --target reference_nuclear_fraction \
  --group-by source_well --seed 0 --out artifacts/nist-rerun
python examples/nist_ipsc/summarize.py \
  --data examples/nist_ipsc/data --results artifacts/nist-rerun
```

The saved report records the actual software versions. Small numerical changes
are possible across scikit-learn versions; this release used NumPy 2.3.5,
scikit-learn 1.8.0, Pillow 12.3.0 and Python 3.12.14. The CSV hash verifies the
input independently of the fitting environment. For matching versions use
`python -m pip install -r examples/nist_ipsc/requirements-reproduction.txt`.

## Rebuild the feature table from original images

Install a sibling checkout of
[brightfield-colony-qc](https://github.com/dylanstechmann/brightfield-colony-qc)
(version 0.3.0,
[commit 1a5f110](https://github.com/dylanstechmann/brightfield-colony-qc/commit/1a5f11051dd7e049ce7d111ec10f0416f413443e)),
then run:

```bash
python -m pip install -e '../brightfield-colony-qc[images]'
OPENBLAS_NUM_THREADS=1 python -m colonyqc.nist_ipsc \
  --cache artifacts/nist-cache --download --out artifacts/nist-features \
  --tile-size 512 --tiles-per-well 64 --seed 0
```

`--download` fetches missing archives only. Omit it to use an existing cache.
All three archives must match pinned byte counts and SHA-256 checksums; an
existing corrupt cache is rejected. Files are extracted to temporary seekable
storage and checked for expected geometry, uint8 mode and a binary 0/1 mask.
Only phase and segmented-nucleus members are extracted. Raw TIFFs and ZIPs
stay outside version control.

Allow roughly 501 MB for the downloaded archives, another 250 MB of temporary
disk space, and about 1 GB of working RAM. No GPU is needed. Extraction takes
a few minutes on a CPU. The default export is 192 rows. Verify
`features_sha256` in the generated provenance against the checked-in
`data/provenance.json`, then evaluate the new CSV with the same regression command.
Every output directory must be new, to preserve prior runs.

## Data card

| Item | Specification |
|---|---|
| Source | NIST mds2-2960, catalog metadata 1.1.0; accessed 2026-09-24 |
| Source files | `training_low.zip`, `training_medium.zip`, `training_high.zip` |
| Original geometry | One 15104 × 15104 phase image and aligned nuclear mask per archive |
| Experimental units | Three different source wells at three seeding-density conditions, described in Asmar et al. |
| Sample selection | 64 tiles per well, 512 × 512 pixels; seed 0; sorted random selection from 29 × 29 full-tile grid |
| Border exclusion | Last 256 pixels on the right and bottom; all selected tiles are full size |
| Inputs | Nine fixed `colonyqc` descriptors from phase pixels divided by 255 |
| Target | Fraction of pixels with `img_segmented.tif > 0` |
| Unused file | `img_fg.tif`; this is a broader foreground mask, not the nuclear reference |
| Holdout | Leave one source well out; three folds, 128 training and 64 test tiles each |
| Provenance | Archive and extracted-member hashes, sampled coordinates, pixel hashes, feature-table hash and software versions |
| Data terms | NIST non-SRD terms; preserve the accompanying [attribution and notice](SOURCE_NOTICE.md) |

The source filenames contain `training` because they were used to train models
in the original paper. This analysis trains its own fixed baselines; it imports
neither the authors' weights nor their code. It is not a replication of their
U-Net segmentation study, and nuclear-area MAE is not comparable to their F1.

Reference masks come from automated processing of fluorescence images. They
can contain merged or missed objects and are not manual ground truth. The
target measures area, not instance count or total colony coverage. Grouped
splitting avoids mixing tiles from one source well across training and test,
but does not create new biological replicates or establish independence of
donors, acquisition days or laboratories. Density and source well are fully
confounded; there is one well per condition. No interval is estimated from
three groups. Further improvement requires untouched independent data.

## Source integrity note

All three ZIP files matched the sizes and SHA-256 values published in the
[NIST metadata record](https://data.nist.gov/od/id/mds2-2960). On 2026-09-24,
the auxiliary `Dataset_Information.xlsx` served by the repository did **not**
match that record (13,935 received bytes versus 14,175 listed bytes). Its
received SHA-256 was `8d58e0bc826b604e4cc1c61aa2bbef1f166135062a678c02772df3c810df7146`;
the catalog listed `ebe28aa1a52569beb2e571d14a55c61f4a5fa1193e4d3c917dc15093b703e891`.
That workbook is not an analysis input and is not redistributed. The well/density
interpretation is supported by the paper, not by the discrepant workbook.

Sources: [dataset DOI](https://doi.org/10.18434/mds2-2960),
[Asmar et al. (2024)](https://doi.org/10.1371/journal.pone.0298446).
