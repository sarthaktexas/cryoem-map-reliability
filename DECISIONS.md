# Project decision log

Running record of design decisions for the cryo-EM local-density thesis. Each
entry follows an ADR (Architectural Decision Record) format so they are
self-contained and can be presented or revisited individually. New entries are
appended at the top.

---

## Decision 008 — Drop BlocRes / MonoRes / ResMap; `local_fsc` only

*Date: 2026-05-21 · Status: ACCEPTED · Extends Decision 005*

### Context

Decision 005 removed ResMap and added home-rolled FSC, but
`local_resolution_io` still accepted BlocRes and MonoRes imports. The thesis
pipeline will use only in-repo windowed local FSC.

### Decision

- Delete `cryoem_mrc/resmap_local.py` (if still present).
- `LocalResolutionSource = Literal["local_fsc"]` only; loader rejects other
  `source=` values and requires `local_fsc` in the filename or MRC labels.
- Remove `--local-res-source` from `scripts/run_analysis.py`.
- Update README, cohort guide, and audit docs to describe `local_fsc` only.

### Consequences

- External Å maps from BlocRes / MonoRes / ResMap are not loaded by this package.
- Recompute local resolution with `scripts/run_local_fsc.py` for any entry.

---

## Decision 006 — Home-rolled windowed local FSC: module, loader, drivers (Tasks 1–4)

*Date: 2026-05-21 · Status: ACCEPTED · Implements Decision 005*

### Context

Decision 005 removed ResMap and committed to in-tree BlocRes-style windowed half-map
FSC. The estimator lives in `cryoem_mrc/local_fsc.py`, with drivers
`scripts/run_local_fsc.py` and `scripts/archive/run_local_fsc_sensitivity.py`, wired into
`local_resolution_io` and `scripts/run_analysis.py` with `--local-res`.

### Decision

- **Task 1:** New module `cryoem_mrc/local_fsc.py` with
  `compute_local_fsc_resolution`, shell-averaged FSC via `scipy.fft.rfftn`,
  trilinear upsampling (`scipy.ndimage.zoom`), optional `n_jobs` via
  `ProcessPoolExecutor`, and `save_local_fsc_resolution_mrc` (MRC label
  `local_fsc t=… P=… s=…`).
- **Task 2:** `LocalResolutionSource` is `"local_fsc"` only (Decision 008 removed
  BlocRes / MonoRes).
- **Task 3:** `scripts/run_local_fsc.py` — aligned halves, contour mask (Decision
  002), summary `.summary.txt`.
- **Task 4:** `scripts/run_analysis.py --local-res` → `correlations_localres.csv`,
  `localres_vs_cc.csv` (Spearman of `local_cross_correlation` and
  `local_reproducibility_snr` vs `local_resolution_A`), and top-K scatter figures.

### Consequences

- `tests/test_local_fsc.py` added (synthetic blob + clip bounds + MRC label).
- Full-box EMD-49450 runs are **user-side** (maps not in repo); see README and
  `scripts/run_local_fsc.py --help` for production commands.
- Methods section must state: no noise substitution; patch-center masking only;
  clip range `[2*voxel, P*voxel]`; map-reliability not molecular flexibility.

### Validation (in-repo)

```bash
python -m unittest tests.test_local_fsc tests.test_local_resolution_export
```

---

## Decision 007 — Default local FSC parameters (Task 5 sensitivity protocol)

*Date: 2026-05-21 · Status: ACCEPTED (locked defaults); EMD-49450 sensitivity panel computed*

### Context

Patch size and FSC threshold trade localization vs. stability. Task 5 requires a
2×3 panel (P ∈ {13, 17, 25}, t ∈ {0.143, 0.5}) and Spearman agreement with
windowed half-map CC inside the contour mask.

### Decision

- **Default for production runs:** `patch_size=17`, `fsc_threshold=0.143`,
  `stride=4`, `window=hann` (BlocRes-style compromise; gold-standard half-map
  threshold for main figures; t=0.5 for supplementary).
- **Sensitivity driver:** `scripts/archive/run_local_fsc_sensitivity.py` writes six MRCs
  under `outputs/sensitivity/local_fsc/`, `spearman_vs_local_cc.csv`, and figures
  `figures/midplane_panel_2x3.png`, `figures/spearman_vs_cc_bar.png`.
- **Expected sign:** Spearman(local_CC, local_FSC Å) **negative** inside mask
  (higher CC ↔ lower Å ↔ better resolution).

### Consequences

- Run on EMD-49450 when `emd_49450/` maps are present:

  ```bash
  python scripts/archive/run_local_fsc_sensitivity.py \
      --half1 data/emd_49450-mgtA_tetramer/emd_49450_half_map_1.map \
      --half2 data/emd_49450-mgtA_tetramer/emd_49450_half_map_2.map \
      --reference data/emd_49450-mgtA_tetramer/emd_49450.map \
      --contour 0.116 --stride 4 --out-dir outputs/sensitivity/local_fsc
  ```

- Lock thesis wording after reviewing midplane panel + bar chart (EMD-49450):
  for `t=0.143`, Spearman(local_CC, local_FSC Å) was `-0.8728` (P=13),
  `-0.9006` (P=17), `-0.9004` (P=25); default production setting kept as
  `P=17, t=0.143` (Decision 007).

---

## Decision 005 — Drop ResMap dependency; compute local resolution in-tree via windowed local FSC

*Date: 2026-05-21 · Status: ACCEPTED · Supersedes the ResMap side-quest in Decision 003*

### Context

ResMap (Kucukelbir 2014) was the original target for Å-valued local
resolution. The wrapper `cryoem_mrc/resmap_local.py` ran ResMap as a
subprocess on the aligned half-maps. Repeated install attempts on the
user's Apple Silicon Mac failed: ResMap is Python-2 era, depends on legacy
TkInter and old NumPy, and has no maintained macOS arm64 build path. The
"install ResMap in the background" plan in Decision 003 was not converging.

Three alternatives were considered:

- **(a) BlocRes (Bsoft).** Same FSC-windowed math, mature reference, but
  Bsoft's macOS arm64 install is also finicky and the binary becomes
  another brittle external dependency.
- **(b) Phenix `mtriage`.** Cleanest install path *if* Phenix is available
  on the user's machine, but adds a heavy external dependency for one
  output map.
- **(c) Home-rolled windowed local FSC in pure Python.** Implements the
  same Cardone et al. 2013 / BlocRes math (windowed FSC, threshold-based
  resolution assignment) in ~150 lines. No external binary. Every
  parameter (patch size, window, threshold, interpolation) is visible in
  the methods section.

### Decision

Option **(c)**. Remove the ResMap wrapper entirely
(`cryoem_mrc/resmap_local.py` deleted; `resmap` source dropped from
`LocalResolutionSource` literal; package init no longer exports
ResMap-related symbols). Implement a home-rolled windowed local FSC
estimator in `cryoem_mrc/local_fsc.py` (see Decision 006 for module and drivers).

External tools (BlocRes, MonoRes, ResMap, Phenix mtriage) are **not** supported
in this codebase (Decision 008). Å-valued maps come only from
`cryoem_mrc.local_fsc`.

### Consequences

- `cryoem_mrc.resmap_local` no longer exists. Code that imported it
  (`require_resmap`, `run_resmap_local_resolution`,
  `run_resmap_local_resolution_with_metadata`, `ResMapError`) was removed
  from `cryoem_mrc/__init__.py` and `cryoem_mrc/local_resolution_io.py`.
- `LocalResolutionSource = Literal["local_fsc"]` (was `["blocres", "resmap",
  "monores"]`, then briefly `["blocres", "monores", "local_fsc"]` per Decision 006).
- `build_dataset_from_pipeline` no longer takes `run_resmap` or
  `resmap_kwargs`; it requires `local_res_path` and an Å-valued MRC on
  disk (produced by `local_fsc` or any external tool the user runs).
- Tests in `tests/test_local_resolution_export.py` were rewritten to test
  `source="local_fsc"` inference and explicit loading.
- README §5 ("ResMap required") and §6 ("Local resolution import") are
  merged into a single section describing the two supported paths
  (external tool, or home-rolled `local_fsc`).
- The presentation framing is now: "we implement BlocRes-style local FSC
  ourselves so the method is fully described in the methods section,"
  which is actually a stronger thesis claim than "we ran ResMap."

### How to apply

When the user runs the in-repo cleanup:

```bash
cd /Users/sarthakmohanty/Developer/thesis
rm cryoem_mrc/resmap_local.py
rm -f cryoem_mrc/__pycache__/resmap_local.cpython-*.pyc
pytest tests/  # verify nothing else still imports the deleted module
```

---

## Decision 004 — Treat averaged-halves correlations as canonical; keep primary-map run as a sharpening-sensitivity reference

*Date: 2026-05-04 · Status: ACCEPTED*

### Context

Decision 001 set the canonical feature input to `0.5 * (half1 + half2)` to avoid
mixing depositor sharpening artifacts into feature-vs-half-map correlations.
As of this run, both analyses are available:

- **Preview (primary-map features):** `outputs/analysis/correlations.csv` *(removed in doc cleanup; deltas preserved below)*
- **Canonical (avg-of-halves features):**
  `outputs/emd_49450/analysis/correlations.csv`

Both use the same half-map target (`local_cross_correlation`) and contour mask
(0.116; 235,240 masked voxels), so row-wise deltas isolate feature-input
differences from map-processing state.

### Options considered

- **(a) Replace the preview outputs and stop discussing them.**
  Simplest, but loses evidence that depositor-side sharpening/filtering moves
  correlation magnitudes for some feature families.
- **(b) Keep both runs and document canonical-vs-preview deltas explicitly.**
  Preserves the methods rationale from Decision 001 while providing a concrete
  sensitivity check suitable for thesis discussion and committee questions.

### Decision

Path **(b)**. Use `outputs/emd_49450/analysis/` as the canonical results for
all headline claims. The preview run documented the sharpening-sensitivity
deltas below; the stale `outputs/analysis/` CSVs were removed after those
deltas were recorded here.

### Observed deltas (preview -> canonical)

Largest absolute deltas (Spearman/Pearson vs `local_cross_correlation`) are in
intensity-heavy features, consistent with map filtering/sharpening effects:

- `density_raw` Spearman: `0.649` -> `0.454` (`delta = -0.195`)
- `density_normalized` Spearman: `0.649` -> `0.459` (`delta = -0.190`)
- `rigidity` Pearson: `-0.587` -> `-0.740` (`delta = -0.152`)
- Coarser Gaussian terms (`gauss_s4_*`) also weaken by ~0.11-0.13.

Top-feature ranking remains broadly stable in composition:

- Top-10 Spearman overlap is **9/10** between preview and canonical.
- Top-5 order changes, but the same core family remains dominant
  (local variance + multiscale variance/gradient terms).
- `local_variance` remains a leading feature in both runs
  (`0.9378` preview, `0.9379` canonical).

### Consequences

- Canonical figures/tables for presentation and write-up should come from:
  `outputs/emd_49450/analysis/`.
- Preview-vs-canonical deltas in this entry remain the sharpening-sensitivity
  appendix panel; re-run preview analysis only if needed for replication.
- Task 4 is closed by the canonical run: all four half-map reliability MRCs are
  regenerated under `outputs/emd_49450/analysis/halfmap_metrics/`, and the
  earlier partial `outputs/halfmap_metrics/` directory was removed.

---

## Decision 003 — Build half-map analysis layer before computing local resolution

*Date: 2026-05-03 · Status: ACCEPTED · **Partially superseded by Decision 005 (2026-05-21)** — the "install ResMap in the background" side-quest is dropped; ResMap is no longer pursued.*

### Context

The handoff document lists two primary reliability signals to compare local
density statistics against:

1. **Half-map agreement** (cross-correlation, MSE, reproducibility-SNR computed
   directly from the two refined halves).
2. **Local resolution in Å** (BlocRes, ResMap, or MonoRes, computed from the
   half-maps via local Fourier shell correlation).

The pipeline already implements (1) end-to-end and produces sensible numbers
on the real EMD-49450 halves (`local_cross_correlation` ranges −0.72 to
+0.91, `local_reproducibility_snr` peaks at ~2.1). It also wraps the ResMap
binary for (2), but does not yet have an analysis layer that turns either
signal into the deliverables called for in handoff §4 and §7 (CSV tables,
correlation figures, `summary.txt`).

The next-step decision is whether to:

- (a) Install ResMap first, run it, and build the analysis machinery against
  half-map agreement and local resolution simultaneously; or
- (b) Build the analysis machinery against half-map agreement first, then add
  local resolution as a second column once a ResMap output is in hand.

### Options considered

**(a) ResMap-first.**
*Pros:* Produces an Å-valued map that is immediately recognizable to
structural-biology audiences; matches the handoff's stated comparison
ordering; gives one defensible reference signal that the field already
trusts.
*Cons:* ResMap installation is a known operational pain point. The binary is
from the Python-2 era (last released ~2015), depends on TKinter and older
NumPy, and frequently requires site-specific patching to run on modern
systems. Even when working, ResMap takes hours to run on a 430³ box. Making
the analysis path depend on a successful ResMap install risks burning
project time on tooling rather than science.

**(b) Half-map first.**
*Pros:* Half-map cross-correlation is already implemented, validated, and
runs in ~2 minutes on the full 430³ EMD-49450 box with no external tool
dependencies. Local FSC — what BlocRes / ResMap / MonoRes estimate — is
mathematically the Fourier-domain reformulation of half-map agreement; both
measure the same underlying reproducibility. Building the analysis around
half-map CC is therefore not "deferring" the central comparison, it is
running the central comparison through its primary signal.
*Cons:* ResMap output remains an outstanding deliverable for the thesis.

**(c) Implement local FSC in pure Python.**
Rejected. Duplicates well-established tools; introduces a method-development
dimension the thesis is not scoped for; harder to defend in a committee
setting than running a citable tool.

### Decision

Path **(b)**. Build `cryoem_mrc/analysis.py` against half-map metrics and the
existing `features.npz` first. Treat ResMap as a parallel side-quest: install
and run overnight in the background while iterating on the analysis layer.
Wire the analysis module so it accepts an optional local-resolution map as a
second input — adding the ResMap column later is a one-line change, not a
rewrite.

### Rationale (presentation-ready)

1. The handoff itself characterizes half-map correlation as the primary
   reliability signal and local resolution as an "imperfect reference."
   Sequencing the implementation to match this priority is consistent with
   the project's stated scientific framing.
2. Local FSC and spatial-domain cross-correlation are two views of the same
   reproducibility signal. The thesis claim — that local density statistics
   correlate with map reliability — is testable against either, and the one
   that does not depend on an external tool is the one that can be tested
   first.
3. Decoupling analysis development from a finicky external install protects
   the timeline. ResMap can fail to install for reasons that have nothing to
   do with the science, and a one-week delay there would propagate.
4. The analysis module is designed to take an optional local-resolution map
   as a second input. Once ResMap is running, dropping its output into the
   pipeline adds a column to the correlation table and a figure panel, with
   no architectural rewrites.

### Consequences

- **Need:** full-box `half_map_local_metrics` on EMD-49450 (~2 min runtime,
  produces four MRCs viewable in ChimeraX).
- **Need:** `cryoem_mrc/analysis.py` and `scripts/run_analysis.py` driver.
- **ResMap install becomes asynchronous.** Recommend setting it up on the
  user's Mac in the background while the analysis layer matures.
- **Thesis caveat to record:** explicitly state in the methods section that
  local-resolution comparison is added in a second pass, and that the
  analysis was designed to accept it without architectural change.

---

## Decision 002 — Use the 0.116 contour level as the analysis mask, with sensitivity check

*Date: 2026-05-03 · Status: ACCEPTED*

### Context

EMD-49450 has a deposited recommended contour level of **0.116** (the density
threshold at which the depositor's chosen ChimeraX isosurface separates
protein density from solvent). The current analysis pipeline runs over the
entire 430³ box (~80M voxels), of which most are solvent or empty. Per-voxel
correlations and feature distributions, computed without a mask, are
dominated by these solvent voxels and become difficult to interpret.

### Options considered

- **(a) No mask.** Correlations diluted by ~70M near-zero solvent voxels;
  conclusions become statistically uninformative even when point estimates
  look strong.
- **(b) Hard mask at 0.116.** Matches the depositor's recommended
  visualization threshold; matches the existing `--start-threshold 0.116`
  flag in the CLI; defensible as "deposited recommended contour from EMDB."
  Caveat: 0.116 is a visualization choice, not a biophysical truth.
- **(c) Percentile-based mask** (e.g. top 10% of voxels by density). Pros:
  data-driven, doesn't depend on depositor choices. Cons: harder to defend
  across different maps; threshold drifts with the map's dynamic range and
  becomes a tuning knob.
- **(d) Soft mask from RELION / cryoSPARC.** Most rigorous, but requires the
  depositor to have provided a mask, which is not always available.

### Decision

Option **(b)** with a sensitivity panel at 0.5×, 1×, and 1.5× the contour as
supporting evidence. Document 0.116 throughout as "deposited recommended
contour from EMDB."

### Consequences

- All analysis runs use mask = `density_raw >= 0.116` by default.
- `analysis.py` accepts `--contour` and (later) `--contour-sensitivity` for
  the multi-threshold comparison panel.
- Reduces the number of voxels entering correlations / scatter plots / CSV
  exports by roughly 5–15× compared to the unmasked box, making downstream
  computation faster and more interpretable.

---

## Decision 001 — Use `0.5 * (half1 + half2)` as the canonical "full map" for stats-vs-CC analysis

*Date: 2026-05-03 · Status: ACCEPTED*

### Context

EMD-49450's deposited primary map and its half-maps share the same grid
(430³, voxel 0.93 Å, origin 0) but have very different processing states.
Sanity check on the loaded data:

- `mean(|0.5*(half1+half2) − full|) ≈ 0.072` against `std(full) ≈ 0.019`
  (a ~4σ discrepancy in absolute terms).
- The halves' standard deviation is roughly **7×** the deposited primary's.

This is normal for EMDB depositions: primary maps are typically B-factor
sharpened, low-pass filtered, and / or masked, while half-maps are the raw
refined halves. Some depositions include an "additional / unsharpened" map
that is closer to the half-map processing state, but EMD-49450 does not.

### Options considered

- **(a) Use the deposited primary map for feature extraction.** Convenient,
  but feature distributions then reflect the depositor's sharpening choices,
  not the underlying density. Comparing those features against half-map CC
  mixes processing artifacts into the correlations.
- **(b) Compute features on `0.5 * (half1 + half2)`.** Same processing state
  as the halves; mathematically equivalent to an unfiltered "additional"
  map deposit when one is provided.
- **(c) Download a deposited additional / unsharpened map if one exists.**
  Signed by the depositor when available, but EMD-49450 does not provide
  one, and adding another input file to manage costs more than computing
  the average ourselves.

### Decision

Option **(b)** for the canonical analysis. Keep the deposited primary map as
a secondary visualization reference (it is what ChimeraX shows users by
default and what the depositor wants the community to look at).

### Consequences

- Features for the canonical CC-comparison analysis are computed on the
  averaged-halves volume. The existing `emd_49450_features.npz` becomes the
  "primary-map features" reference; a new `emd_49450_avg_features.npz`
  becomes the input for `analysis.py`.
- The thesis methods section explicitly documents that primary-map features
  and avg-of-halves features differ by depositor sharpening / filtering and
  that we use the latter for stats-vs-CC because both signals then share a
  common processing state.
- Re-running the pipeline on the averaged-halves volume requires the
  user-side command (the sandbox per-call time limit cannot complete a full
  430³ pipeline pass in one shot):

  ```bash
  # On the user's Mac, from the repo root with .venv active:
  python -c "
  import numpy as np, mrcfile
  from cryoem_mrc.map_grid import load_full_and_half_maps
  from cryoem_mrc.io import save_volume_like_reference
  b = load_full_and_half_maps(
      'emd_49450/emd_49450.map',
      'emd_49450/emd_49450_half_map_1.map',
      'emd_49450/emd_49450_half_map_2.map',
      reference='full', dtype=np.float32, resample_if_needed=False,
  )
  avg = (0.5 * (b.half1.data + b.half2.data)).astype(np.float32)
  save_volume_like_reference('emd_49450/emd_49450.map', avg, 'emd_49450/emd_49450_avg.map')
  "
  python -m cryoem_mrc emd_49450/emd_49450_avg.map \
      --start-threshold 0.116 --float32 \
      --out emd_49450_avg_features_t0116.npz \
      --rigidity-mrc-out emd_49450_avg_rigidity_t0116.mrc
  ```

---
