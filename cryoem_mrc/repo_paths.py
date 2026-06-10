"""Repository layout conventions: ``data/``, ``outputs/emd_<ID>/``, ``cohort/manifest.csv``."""

from __future__ import annotations

import shutil
from pathlib import Path

DATA_ROOT = Path("data")
OUTPUTS_ROOT = Path("outputs")
DOCS_FIGURES_ROOT = Path("docs/figures")

# ``outputs/cohort_summary/<src>`` → ``docs/figures/<dest>`` for THESIS_NARRATIVE.md.
THESIS_NARRATIVE_COHORT_FIGURES: dict[str, str] = {
    "cohort_metrics_heatmap.png": "fig_3_2_cohort_metrics_heatmap.png",
    "cohort_variance_vs_reliability_cc.png": "fig_3_2_cohort_variance_vs_reliability_cc.png",
    "bfactor_horse_race.png": "fig_3_3_bfactor_horse_race.png",
    "cohort_reliability_by_class.png": "fig_3_4_reliability_by_class.png",
    "cohort_cross_metric_median.png": "fig_3_4_cross_metric_median.png",
    "cohort_cross_metric_locres_pairs.png": "fig_3_4_cross_metric_locres_pairs.png",
    "qscore_vs_V_cohort.png": "fig_3_4_qscore_vs_V_cohort.png",
    "fscq_vs_V_cohort.png": "fig_3_4_fscq_vs_V_cohort.png",
    "tv_line_scatter.png": "fig_3_4_tv_line_scatter.png",
    "v_incremental_prediction.png": "fig_3_4_v_incremental_prediction.png",
    "placement_decoupling_cohort.png": "fig_3_6_placement_decoupling_cohort.png",
    "clpb_wt2a_placement_supplement.png": "fig_s4_clpb_wt2a_placement_supplement.png",
    "guinier_sharpen_benchmark.png": "fig_3_8_guinier_sharpen_benchmark.png",
}

THESIS_APPENDIX_B_FIGURES: dict[str, str] = {
    "fig_b1_local_fsc_production_slice.png": "fig_b1_local_fsc_production_slice.png",
    "fig_b2_local_fsc_sensitivity_bar.png": "fig_b2_local_fsc_sensitivity_bar.png",
    "fig_b3_contour_sensitivity.png": "fig_b3_contour_sensitivity.png",
}


def sync_thesis_doc_figure(src: Path, dest_name: str) -> Path:
    """Copy a generated figure into ``docs/figures/`` for thesis markdown embedding."""
    dest = DOCS_FIGURES_ROOT / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    pdf_src = src.with_suffix(".pdf")
    if pdf_src.is_file():
        shutil.copy2(pdf_src, dest.with_suffix(".pdf"))
    return dest


def sync_thesis_appendix_b_figure(source: Path, thesis_name: str) -> Path:
    """Copy a generated sensitivity PNG into ``docs/figures/`` (appendix B)."""
    return sync_thesis_doc_figure(source, THESIS_APPENDIX_B_FIGURES[thesis_name])


def sync_thesis_narrative_cohort_figures(
    cohort_dir: Path | None = None,
) -> list[Path]:
    """Mirror cohort-summary PNGs into ``docs/figures/`` for self-contained thesis prose."""
    cohort_dir = cohort_dir or (OUTPUTS_ROOT / "cohort_summary")
    synced: list[Path] = []
    for src_name, dest_name in THESIS_NARRATIVE_COHORT_FIGURES.items():
        src = cohort_dir / src_name
        if src.is_file():
            synced.append(sync_thesis_doc_figure(src, dest_name))
    return synced


PDB_ROOT = Path("pdb")
COHORT_MANIFEST = Path("cohort/manifest.csv")

# Canonical anchor map for thesis validation panels (Decision 004).
ANCHOR_EMDB_ID = "49450"

# Subset of b_factor manifest rows worth B-factor validation figures in the thesis.
BFACTOR_VALIDATION_EMDB_IDS: tuple[str, ...] = ("49450", "44471", "28498")


def emd_output_dir(emdb_id: str | int) -> Path:
    return OUTPUTS_ROOT / f"emd_{str(emdb_id).strip()}"


def analysis_dir(emdb_id: str | int) -> Path:
    return emd_output_dir(emdb_id) / "analysis"


def analysis_localres_dir(emdb_id: str | int) -> Path:
    return emd_output_dir(emdb_id) / "analysis_localres"


def lh_map_reliability_dir(emdb_id: str | int) -> Path:
    """Per-map LH reliability bundle: ``outputs/emd_<ID>/lh_map_reliability/``."""
    return emd_output_dir(emdb_id) / "lh_map_reliability"


def thesis_overview_dir(emdb_id: str | int = "49450") -> Path:
    return emd_output_dir(emdb_id) / "thesis_overview"


def sync_thesis_appendix_b1_from_anchor(
    emdb_id: str | int = ANCHOR_EMDB_ID,
) -> Path | None:
    """Sync anchor local-FSC slice for appendix B.1."""
    src = thesis_overview_dir(emdb_id) / "local_resolution_slice.png"
    if not src.is_file():
        return None
    return sync_thesis_doc_figure(src, "fig_b1_local_fsc_production_slice.png")


def locres_blocres_mrc(emdb_id: str | int) -> Path:
    return emd_output_dir(emdb_id) / "locres_blocres.mrc"


def halfmap_metrics_npz(emdb_id: str | int) -> Path:
    return analysis_dir(emdb_id) / "halfmap_metrics.npz"


def sensitivity_local_fsc_dir() -> Path:
    return OUTPUTS_ROOT / "sensitivity" / "local_fsc"


def archive_rigidity_vs_mechanics_dir() -> Path:
    return OUTPUTS_ROOT / "archive" / "rigidity_vs_mechanics"


def bfactor_conformation_pairs_dir() -> Path:
    return OUTPUTS_ROOT / "bfactor_conformation_pairs"


def find_features_npz(data_dir: Path, emdb_id: str | int, contour: float) -> Path | None:
    """
    Locate averaged-map feature NPZ for one EMDB entry.

    Prefer the depositor-contour tag, then ``t0000`` (unsharpened avg maps whose
    in-mask max sits below the depositor contour — see cohort pipeline), then any match.
    """
    emd = f"emd_{str(emdb_id).strip()}"
    tag = f"t{int(round(float(contour) * 1000)):04d}"
    candidates = [
        data_dir / f"{emd}_avg_features_{tag}.npz",
        data_dir / f"{emd}_avg_features_t{int(round(contour * 1000)):04d}.npz",
        data_dir / f"{emd}_avg_features_t0000.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_dir.glob(f"{emd}_avg_features*.npz"))
    return matches[0] if matches else None
