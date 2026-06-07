# cryoem-map-reliability

Python tools for **local map reliability** in cryo-EM reconstructions: density statistics, half-map reproducibility, windowed local FSC (Å), a reproducibility score (H_repro), build/caution/omit zones, and optional deposited-model B-factor checks.

The goal is to test whether inexpensive map features track **half-map cross-correlation** and **local FSC** well enough to guide modeling. This is **not** a claim that density alone defines molecular flexibility.

All volumes use NumPy 3D arrays in `(Z, Y, X)` order (section, row, column), consistent with typical `mrcfile` layouts.

---

## Install

```bash
pip install -r requirements.txt
# or editable:
pip install -e .
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

Download deposited and half maps from [EMDB](https://www.ebi.ac.uk/emdb/) and fitted models from [PDBe](https://www.ebi.ac.uk/pdbe/). Use the depositor-recommended contour for each entry (listed in `cohort/manifest.csv`).

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
| `scripts/run_analysis.py`                          | Feature vs half-map CC (+ optional local FSC map)        |
| `scripts/run_local_fsc.py`                         | Windowed local FSC → Å MRC                               |
| `scripts/run_lh_map_reliability_export.py`         | H_repro, reliability score, build zones, summary figures |
| `scripts/run_extended_feature_validation.py`       | Extended stats, Hessian, ridge CV vs CC                  |
| `scripts/run_lh_vs_b_and_fsc.py`                   | Reliability metrics vs B-factors, CC, local FSC          |
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
# metrics["local_cross_correlation"], etc.

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

- **Half-map CC** (windowed cross-correlation) is the primary reproducibility target for feature validation.
- **Local FSC** is computed in-repo (`cryoem_mrc.local_fsc`); external BlocRes / ResMap / MonoRes maps are not loaded.
- **H_repro** combines windowed half-map fluctuation (T) and density gradient smoothness (V); **reliability_score** is an in-mask percentile used for build/caution/omit terciles.
- **Local variance** is often the strongest single feature predictor of half-map CC; treat B-factor correlations as exploratory and report partial correlations when comparing scores.

Design choices and parameter defaults are recorded in [DECISIONS.md](DECISIONS.md).

---

## Tests

```bash
python -m unittest discover -s tests -v
```

---

## Citation

**Before the manuscript is published**, cite the software (and pin a commit hash or release tag if reproducibility matters):

```bibtex
@software{mohanty2026cryoem_map_reliability,
  author = {Mohanty, Sarthak},
  title = {cryoem-map-reliability: local map reliability from cryo-EM density and half-maps},
  year = {2026},
  url = {https://github.com/sarthaktexas/cryoem-map-reliability},
  version = {0.1.0}
}
```

GitHub also reads [CITATION.cff](CITATION.cff) for the **Cite this repository** button.

**After publication**, cite the paper as the primary reference. Also cite this repo (or a Zenodo archive) when you need the exact pipeline version used in the work—for example after tagging a release or minting a DOI at acceptance.

When the manuscript exists, add a `preferred-citation` block to `CITATION.cff` (template included there) and drop the BibTeX for the article into this section.

## License

MIT License. See [LICENSE](LICENSE).