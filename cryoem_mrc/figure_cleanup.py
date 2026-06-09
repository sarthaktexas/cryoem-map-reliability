"""Remove retired per-map figure exports superseded by panels or cohort summaries."""

from __future__ import annotations

from pathlib import Path

ANALYSIS_SCATTER_GLOB = "*_vs_*.png"
ANALYSIS_SCATTER_GLOB_PDF = "*_vs_*.pdf"

LH_RETIRED_STEMS = (
    "spearman_predictors",
    "partial_incremental",
    "reliability_vs_cc_binned",
    "bfactor_vs_reliability",
    "bfactor_by_build_zone",
)


def prune_analysis_scatter_figures(fig_dir: Path) -> list[Path]:
    """Delete per-feature scatter PNG/PDF; keep histograms and validation panels."""
    removed: list[Path] = []
    if not fig_dir.is_dir():
        return removed
    keep = frozenset(
        {
            "halfmap_metric_histograms.png",
            "halfmap_metric_histograms.pdf",
            "analysis_validation_panel.png",
            "analysis_validation_panel.pdf",
        }
    )
    for path in sorted(fig_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in keep:
            continue
        if path.suffix in {".png", ".pdf"} and "_vs_" in path.name:
            path.unlink()
            removed.append(path)
    return removed


def prune_lh_retired_figures(fig_dir: Path) -> list[Path]:
    """Delete orphaned LH diagnostics replaced by panels or cohort_summary."""
    removed: list[Path] = []
    if not fig_dir.is_dir():
        return removed
    keep = frozenset(
        {
            "model_building_row.png",
            "model_building_row.pdf",
            "bfactor_validation_panel.png",
            "bfactor_validation_panel.pdf",
        }
    )
    for stem in LH_RETIRED_STEMS:
        for ext in (".png", ".pdf"):
            path = fig_dir / f"{stem}{ext}"
            if path.is_file() and path.name not in keep:
                path.unlink()
                removed.append(path)
    return removed


def prune_retired_figures_under_outputs(outputs_root: Path) -> dict[str, list[str]]:
    """Prune analysis and lh_map_reliability figure dirs for every EMD entry."""
    summary: dict[str, list[str]] = {"analysis": [], "lh_map_reliability": []}
    for emd_dir in sorted(outputs_root.glob("emd_*")):
        for sub, fn in (
            ("analysis/figures", prune_analysis_scatter_figures),
            ("lh_map_reliability/figures", prune_lh_retired_figures),
        ):
            fig_dir = emd_dir / sub
            for path in fn(fig_dir):
                summary[sub.split("/")[0]].append(str(path))
    return summary
