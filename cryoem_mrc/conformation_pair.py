"""Conformation-pair helpers: coverage, superposition, ChimeraX export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .structure_validation import CaResidue, ResidueValidationRow, iter_ca_residues

COVERAGE_FLAG_THRESHOLD_PCT = 20.0


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

    Used for **visualization only** — Δ statistics remain per-map without superposition.
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


def residue_key_to_coupling_map(
    use: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    values: np.ndarray,
) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    for (a, _), val in zip(use, values):
        if np.isfinite(val):
            out[a.residue_key] = float(val)
    return out


def write_coupling_colored_pdb(
    structure_path: str | Path,
    out_path: str | Path,
    residue_values: Mapping[tuple[str, int, str], float],
    *,
    bfactor_scale: float = 100.0,
) -> Path:
    """Write PDB with Cα B_iso = coupling × scale (ChimeraX ``color byattribute bfaa``)."""
    import gemmi

    structure_path = Path(structure_path)
    out_path = Path(out_path)
    st = gemmi.read_structure(str(structure_path))
    st.remove_alternative_conformations()
    for model in st:
        for chain in model:
            for residue in chain:
                key = (chain.name, int(residue.seqid.num), str(residue.seqid.icode).strip())
                val = residue_values.get(key)
                if val is None:
                    continue
                ca = residue.find_atom("CA", "\0")
                if ca is not None:
                    ca.b_iso = float(val) * bfactor_scale
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_path))
    return out_path


def write_aligned_ca_pdb(
    residues: Sequence[CaResidue],
    coords: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Minimal Cα-only PDB from aligned coordinates (for ChimeraX overlay)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, (res, xyz) in enumerate(zip(residues, coords), start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  {res.res_name:>3s} {res.chain:>1s}"
            f"{res.seq_num:4d}{res.seq_icode:>1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00           C  "
        )
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_chimerax_coupling_script(
    *,
    colored_pdb_a: Path,
    aligned_pdb_b: Path | None,
    session_path: Path,
    emdb_a: str,
    emdb_b: str,
    bfactor_scale: float = 100.0,
) -> Path:
    """
    Write a ChimeraX command script (``.cxc``) to color state A by coupling and optionally
    overlay aligned state B Cα trace.
    """
    session_path = Path(session_path)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    hi = float(bfactor_scale)
    lo = 0.0
    lines = [
        f"# ChimeraX — conformation pair EMD-{emdb_a} vs EMD-{emdb_b}",
        f"# Coupling stored in Cα B-factor (×{bfactor_scale:.0f}); open in ChimeraX:",
        f"#   chimerax {session_path.name}",
        "",
        f'open "{colored_pdb_a.resolve()}" name stateA',
        "show #stateA @ca",
        f"color byattribute #stateA bfaa palette yellow:red range {lo},{hi}",
        "size #stateA @ca diameter 1.2",
    ]
    if aligned_pdb_b is not None and aligned_pdb_b.is_file():
        lines.extend(
            [
                f'open "{aligned_pdb_b.resolve()}" name stateB',
                "show #stateB @ca",
                "color #stateB @ca lightgray",
                "size #stateB @ca diameter 0.9",
                "transparency #stateB @ca 60",
            ]
        )
    lines.extend(
        [
            "view orient",
            f'save "{session_path.resolve()}"',
            "",
        ]
    )
    session_path.write_text("\n".join(lines))
    return session_path


__all__ = [
    "COVERAGE_FLAG_THRESHOLD_PCT",
    "ConformationPairCoverage",
    "compute_conformation_pair_coverage",
    "interior_residue_indices",
    "kabsch_align_coords",
    "residue_key_to_coupling_map",
    "write_coupling_colored_pdb",
    "write_aligned_ca_pdb",
    "write_chimerax_coupling_script",
]
