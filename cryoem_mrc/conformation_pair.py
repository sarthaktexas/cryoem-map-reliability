"""Conformation-pair helpers: coverage and visualization alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .structure_validation import ResidueValidationRow

COVERAGE_FLAG_THRESHOLD_PCT = 20.0
_COHORT_DIR = Path(__file__).resolve().parent.parent / "cohort"
TRPV1_DOMAIN_REGIONS_PATH = _COHORT_DIR / "trpv1_domain_regions.json"
MGTA_DOMAIN_REGIONS_PATH = _COHORT_DIR / "mgta_domain_regions.json"
TRPV1_EMDB_IDS = frozenset({"23129", "23130"})
MGTA_EMDB_IDS = frozenset({"49450", "48534", "48923", "48602"})


@dataclass
class ConformationPairCoverage:
    """How completely matched in-mask Cα cover each deposited model."""

    emdb_a: str
    emdb_b: str
    n_ca_total_a: int
    n_ca_total_b: int
    n_matched: int
    n_matched_in_mask_both: int
    n_matched_in_mask_a: int
    n_matched_in_mask_b: int
    frac_analysis_of_a: float
    frac_analysis_of_b: float
    missing_pct_a: float
    missing_pct_b: float
    coverage_flag: bool
    notes: str = ""


def compute_conformation_pair_coverage(
    pairs: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    emdb_a: str,
    emdb_b: str,
    n_ca_total_a: int,
    n_ca_total_b: int,
    threshold_pct: float = COVERAGE_FLAG_THRESHOLD_PCT,
) -> ConformationPairCoverage:
    """Compare analysis residue count to full deposited Cα count per model."""
    n_matched = len(pairs)
    n_mask_a = sum(1 for a, _ in pairs if a.in_contour_mask)
    n_mask_b = sum(1 for _, b in pairs if b.in_contour_mask)
    n_both = sum(1 for a, b in pairs if a.in_contour_mask and b.in_contour_mask)

    def _frac(n: int, total: int) -> float:
        return float(n / total) if total > 0 else float("nan")

    frac_a = _frac(n_both, n_ca_total_a)
    frac_b = _frac(n_both, n_ca_total_b)
    miss_a = 100.0 * (1.0 - frac_a) if np.isfinite(frac_a) else float("nan")
    miss_b = 100.0 * (1.0 - frac_b) if np.isfinite(frac_b) else float("nan")
    flagged = (np.isfinite(miss_a) and miss_a > threshold_pct) or (
        np.isfinite(miss_b) and miss_b > threshold_pct
    )

    notes = ""
    if flagged:
        notes = (
            f"> {threshold_pct:.0f}% of deposited Cα are outside the analysis set "
            f"(unmatched and/or below contour). Interpret coupling maps with this gap in mind."
        )

    return ConformationPairCoverage(
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        n_ca_total_a=n_ca_total_a,
        n_ca_total_b=n_ca_total_b,
        n_matched=n_matched,
        n_matched_in_mask_both=n_both,
        n_matched_in_mask_a=n_mask_a,
        n_matched_in_mask_b=n_mask_b,
        frac_analysis_of_a=frac_a,
        frac_analysis_of_b=frac_b,
        missing_pct_a=miss_a,
        missing_pct_b=miss_b,
        coverage_flag=flagged,
        notes=notes,
    )


def interior_residue_indices(n: int, half_window: int) -> np.ndarray:
    """Residue indices with full ±half_window coupling windows (no edge NaNs)."""
    if n <= 2 * half_window:
        return np.arange(n, dtype=int)
    return np.arange(half_window, n - half_window, dtype=int)


def kabsch_align_coords(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rigid-body alignment of ``mobile`` onto ``target`` (N×3 Cα coordinates).

    Used for **matplotlib 3D overlay only** — Δ statistics remain per-map without superposition.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.shape[0] < 3:
        return mobile.copy(), np.eye(3, dtype=np.float64)
    mob_ctr = mobile.mean(axis=0)
    tgt_ctr = target.mean(axis=0)
    rot, _ = Rotation.align_vectors(mobile - mob_ctr, target - tgt_ctr)
    aligned = rot.apply(mobile - mob_ctr) + tgt_ctr
    return aligned, rot.as_matrix()


@dataclass(frozen=True)
class DomainRegion:
    """Sequence-based fold band for heatmap annotation (auth seq_id)."""

    name: str
    seq_start: int
    seq_end: int
    color: str


def is_trpv1_conformation_pair(emdb_a: str, emdb_b: str) -> bool:
    return {str(emdb_a).strip(), str(emdb_b).strip()} <= TRPV1_EMDB_IDS


def is_mgta_conformation_pair(emdb_a: str, emdb_b: str) -> bool:
    return {str(emdb_a).strip(), str(emdb_b).strip()} <= MGTA_EMDB_IDS


def load_domain_regions_from_json(path: Path) -> list[DomainRegion]:
    raw = json.loads(path.read_text())
    return [
        DomainRegion(
            name=str(r["name"]),
            seq_start=int(r["seq_start"]),
            seq_end=int(r["seq_end"]),
            color=str(r["color"]),
        )
        for r in raw["regions"]
    ]


def load_trpv1_domain_regions() -> list[DomainRegion]:
    """Rat TRPV1 domain bands for EMD-23129/23130 (PDB 7L2I/7L2J auth numbering)."""
    return load_domain_regions_from_json(TRPV1_DOMAIN_REGIONS_PATH)


def load_mgta_domain_regions() -> list[DomainRegion]:
    """L. lactis MgtA domain bands for EMD-49450/48923/48534 (PDB 9NHZ/9N5J/9MQM)."""
    return load_domain_regions_from_json(MGTA_DOMAIN_REGIONS_PATH)


def _all_domain_region_paths() -> list[Path]:
    return sorted(_COHORT_DIR.glob("*_domain_regions.json"))


def domain_colors_from_regions(regions: Sequence[DomainRegion]) -> dict[str, str]:
    """Domain name → hex color (single source for figure panels)."""
    return {reg.name: reg.color for reg in regions}


def _merged_domain_colors() -> dict[str, str]:
    colors: dict[str, str] = {}
    for path in _all_domain_region_paths():
        colors.update(domain_colors_from_regions(load_domain_regions_from_json(path)))
    return colors


DOMAIN_COLORS: dict[str, str] = _merged_domain_colors()
UNASSIGNED_DOMAIN_COLOR = "#aaaaaa"


def reload_domain_colors() -> dict[str, str]:
    """Refresh DOMAIN_COLORS after JSON edits (tests / long-running sessions)."""
    global DOMAIN_COLORS
    DOMAIN_COLORS = _merged_domain_colors()
    return DOMAIN_COLORS


def get_domain_regions_for_pair(emdb_a: str, emdb_b: str) -> list[DomainRegion]:
    """Return domain region definitions when annotated for this pair, else []."""
    if is_trpv1_conformation_pair(emdb_a, emdb_b):
        return load_trpv1_domain_regions()
    if is_mgta_conformation_pair(emdb_a, emdb_b):
        return load_mgta_domain_regions()
    return []


def get_domain_assignments(
    chain_residue_list: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    regions: Sequence[DomainRegion] | None = None,
) -> dict[str, list[int]]:
    """Map domain name → chain-order residue indices (auth seq_id bands)."""
    if not regions:
        return {}
    assignments: dict[str, list[int]] = {reg.name: [] for reg in regions}
    for i, (row, _) in enumerate(chain_residue_list):
        seq_num = int(row.seq_num)
        for reg in regions:
            if reg.seq_start <= seq_num <= reg.seq_end:
                assignments[reg.name].append(i)
                break
    return assignments


def compute_domain_mean_coupling(
    corr: np.ndarray,
    assignments: dict[str, list[int]],
    *,
    domain_order: Sequence[str] | None = None,
    metric: str = "mean_abs",
    abs_threshold: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """Summarize residue×residue coupling within each domain block.

    ``metric``:
        ``signed_mean`` — mean Pearson *r* (mixed signs often cancel → ~0 blocks).
        ``mean_abs`` — mean |*r*| (coupling magnitude regardless of sign).
        ``frac_strong`` — fraction of pairs with |*r*| > ``abs_threshold``.
    """
    if domain_order is None:
        names = [name for name, idx in assignments.items() if idx]
    else:
        names = [name for name in domain_order if assignments.get(name)]
    n_dom = len(names)
    mat = np.full((n_dom, n_dom), np.nan, dtype=np.float64)
    for i, di in enumerate(names):
        rows = np.asarray(assignments[di], dtype=int)
        for j, dj in enumerate(names):
            cols = np.asarray(assignments[dj], dtype=int)
            if rows.size and cols.size:
                block = corr[np.ix_(rows, cols)]
                if metric == "signed_mean":
                    mat[i, j] = float(np.nanmean(block))
                elif metric == "mean_abs":
                    mat[i, j] = float(np.nanmean(np.abs(block)))
                elif metric == "frac_strong":
                    mat[i, j] = float(np.nanmean(np.abs(block) > abs_threshold))
                else:
                    raise ValueError(
                        f"metric must be signed_mean, mean_abs, or frac_strong; got {metric!r}"
                    )
    return mat, names


def domain_residue_color(
    seq_num: int,
    regions: Sequence[DomainRegion],
) -> str | None:
    """Return domain color for auth seq_id, or None when outside annotated bands."""
    for reg in regions:
        if reg.seq_start <= int(seq_num) <= reg.seq_end:
            return DOMAIN_COLORS.get(reg.name, reg.color)
    return None


def domain_index_spans(
    use: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    regions: Sequence[DomainRegion],
) -> list[tuple[float, float, str, str]]:
    """Map auth seq_id regions to contiguous chain-order index intervals.

    Each chain copy yields its own span (homotetramer → up to four ANK/TM/C-term blocks).
    """
    spans: list[tuple[float, float, str, str]] = []
    n = len(use)
    for reg in regions:
        i = 0
        while i < n:
            seq_num = int(use[i][0].seq_num)
            if reg.seq_start <= seq_num <= reg.seq_end:
                j = i + 1
                while j < n:
                    sn = int(use[j][0].seq_num)
                    if not (reg.seq_start <= sn <= reg.seq_end):
                        break
                    j += 1
                spans.append(
                    (float(i) - 0.5, float(j) - 0.5, reg.name, DOMAIN_COLORS.get(reg.name, reg.color))
                )
                i = j
            else:
                i += 1
    return spans


__all__ = [
    "COVERAGE_FLAG_THRESHOLD_PCT",
    "ConformationPairCoverage",
    "DomainRegion",
    "DOMAIN_COLORS",
    "UNASSIGNED_DOMAIN_COLOR",
    "compute_conformation_pair_coverage",
    "compute_domain_mean_coupling",
    "domain_index_spans",
    "domain_residue_color",
    "get_domain_assignments",
    "get_domain_regions_for_pair",
    "interior_residue_indices",
    "is_mgta_conformation_pair",
    "is_trpv1_conformation_pair",
    "kabsch_align_coords",
    "load_domain_regions_from_json",
    "load_mgta_domain_regions",
    "reload_domain_colors",
    "load_trpv1_domain_regions",
]
