# cryoem-halfmap-qc

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20618527.svg)](https://doi.org/10.5281/zenodo.20618527)

Python tools for **local map reliability** in cryo-EM reconstructions: density statistics, half-map reproducibility, windowed local FSC (Å), a reproducibility score (H_repro), build/caution/omit zones, and optional deposited-model B-factor checks.

The goal is to test whether inexpensive map features track **half-map cross-correlation** and **local FSC** well enough to guide modeling. This is **not** a claim that density alone defines molecular flexibility.

All volumes use NumPy 3D arrays in `(Z, Y, X)` order (section, row, column), consistent with typical `mrcfile` layouts.

---

## Install

```bash
uv pip install -e .
# or:
pip install -e .
# from GitHub (no PyPI upload yet):
pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.2.0"
```

**Dependencies:** NumPy, SciPy, mrcfile, Matplotlib, gemmi (mmCIF/PDB for residue-level validation).

---

## Data layout

Cryo-EM maps are **not** stored in this repository (too large for git). After cloning, create local directories:

```text
data/emd_<ID>-<label>/     # deposited map + half-maps (.map or .mrc)
outputs/emd_<ID>/           # pipeline products (created by scripts)
pdb/                        # fitted models (mmCIF) — sample models included for the cohort
cohort/manifest.csv         # EMDB IDs, relative paths, contours, validation labels
```

Download deposited and half maps from [EMDB](https://www.ebi.ac.uk/emdb/) and fitted models from [PDBe](https://www.ebi.ac.uk/pdbe/). Use the depositor-recommended contour for each entry (listed in `cohort/manifest.csv`). See [docs/COHORT.md](docs/COHORT.md) for download status and pipeline progress.

---

## Quick start

**Single-map features** (writes a compressed `.npz` bundle):

```bash
python -m cryoem_mrc path/to/map.mrc --out map_features.npz --float32
```

**Typical deposited-map workflow** (replace paths and contour for your entry):

```bash
EMD=49450
CONTOUR=0.116
DATA=data/emd_${EMD}-mgtA_e2p+e1

python scripts/run_analysis.py \
  --features "${DATA}/emd_${EMD}_avg_features_t0116.npz" \
  --half1 "${DATA}/emd_${EMD}_half_map_1.map" \
  --half2 "${DATA}/emd_${EMD}_half_map_2.map" \
  --reference "${DATA}/emd_${EMD}.map" \
  --contour "${CONTOUR}" \
  --out-dir "outputs/emd_${EMD}/analysis"

python scripts/run_local_fsc.py \
  --half1 "${DATA}/emd_${EMD}_half_map_1.map" \
  --half2 "${DATA}/emd_${EMD}_half_map_2.map" \
  --reference "${DATA}/emd_${EMD}.map" \
  --contour "${CONTOUR}" \
  --out "outputs/emd_${EMD}/analysis_localres/emd_${EMD}_local_fsc.mrc"

python scripts/run_lh_map_reliability_export.py --emd-id "${EMD}"
python scripts/run_extended_feature_validation.py --emd-id "${EMD}"
```

**Cohort batch run** (all active entries in `cohort/manifest.csv` with local data):

```bash
python scripts/run_cohort_pipeline.py
```

---

## Scripts


| Script                                             | Purpose                                                  |
| -------------------------------------------------- | -------------------------------------------------------- |
| `scripts/run_analysis.py`                          | Feature vs windowed half-map correlation (+ local FSC) |
| `scripts/run_local_fsc.py`                         | Windowed local FSC → Å MRC                               |
| `scripts/run_lh_map_reliability_export.py`         | H_repro, reliability score, build zones, summary figures |
| `scripts/run_extended_feature_validation.py`       | Extended stats, Hessian, ridge CV vs CC                  |
| `scripts/run_residue_bfactor_validation.py`        | Cα B_iso vs reliability / build zones                    |
| `scripts/run_residue_bfactor_score_correlation.py` | B vs multiple map scores (sphere sampling)               |
| `scripts/run_residue_bfactor_conformation_pair.py` | ΔB vs Δreliability across two EMDB states                |
| `scripts/run_cohort_pipeline.py`                   | Batch processing from `cohort/manifest.csv`              |
| `scripts/run_cohort_summary_figures.py`            | Cohort-level summary tables and figures                  |


Archive / sensitivity scripts live under `scripts/archive/`.

---

## Python API (high level)

```python
import numpy as np
from cryoem_mrc import load_full_and_half_maps, run_pipeline, half_map_local_metrics
from cryoem_mrc.reliability import compute_reliability_maps, classify_build_zones

bundle = load_full_and_half_maps(
    "full.mrc", "half1.mrc", "half2.mrc", dtype=np.float32, resample_if_needed=True
)
metrics = half_map_local_metrics(bundle.half1, bundle.half2, window=5)
# metrics["windowed_halfmap_correlation"], etc.

features = run_pipeline("map.mrc", use_float32=True)
reliability = compute_reliability_maps(
    bundle.half1, bundle.half2,
    density_normalized=features["density_normalized"],
    window=5,
)
zones = classify_build_zones(reliability["reliability_score"])
```

**Package modules:** `io`, `map_grid`, `local_stats`, `multiscale`, `half_map_repro`, `local_fsc`, `mechanics`, `reliability`, `analysis`, `structure_validation`. Path helpers: `cryoem_mrc/repo_paths.py`.

---

## Methods summary

- **Windowed half-map correlation** is the fast internal reproducibility target for feature validation; **local FSC resolution (Å)** is the field-standard reference.
- **Local FSC** is computed in-repo (`cryoem_mrc.local_fsc`); external BlocRes / ResMap / MonoRes maps are not loaded.
- **H_repro** combines windowed half-map fluctuation (T) and density gradient smoothness (V); **reliability_score** is an in-mask percentile used for build/caution/omit terciles.
- **Local variance** is often the strongest single feature predictor of windowed half-map correlation; treat B-factor correlations as exploratory and report partial correlations when comparing scores.

Design choices and parameter defaults are recorded in [DECISIONS.md](DECISIONS.md).

**Thesis prose:** full narrative draft in [docs/THESIS_NARRATIVE.md](docs/THESIS_NARRATIVE.md). Writing guide and defense notes in [docs/THESIS_AND_PUBLICATION.md](docs/THESIS_AND_PUBLICATION.md).

---

## Tests

```bash
python -m unittest discover -s tests -v
```

---

## Citation

**Before the manuscript is published**, cite the software with the Zenodo concept DOI (resolves to the latest release; pin `v0.2.0` or a commit hash for exact reproducibility):

```bibtex
@software{mohanty2026cryoem_halfmap_qc,
  author = {Mohanty, Sarthak},
  title = {cryoem-halfmap-qc: local map reliability from cryo-EM density and half-maps},
  year = {2026},
  doi = {10.5281/zenodo.20618527},
  url = {https://doi.org/10.5281/zenodo.20618527},
  version = {0.2.0}
}
```

GitHub also reads [CITATION.cff](CITATION.cff) for the **Cite this repository** button.

**After publication**, cite the paper as the primary reference. Also cite this Zenodo archive when you need the exact pipeline version used in the work.

When the manuscript exists, add a `preferred-citation` block to `CITATION.cff` (template included there) and drop the BibTeX for the article into this section.

## License

MIT License. See [LICENSE](LICENSE).