"""Cohort placement-decoupling diagnostics: reliability rank vs half-map CC at deposited Cα."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from .analysis import build_contour_mask
from .half_map_repro import (
    WINDOWED_HALFMAP_CORRELATION_KEY,
    half_map_local_metrics,
    load_windowed_halfmap_correlation,
)
from .map_grid import load_full_and_half_maps
from .reliability import attach_reliability_to_features, percentile_rank_in_mask
from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, halfmap_metrics_npz, lh_map_reliability_dir
from .structure_validation import (
    ResidueValidationRow,
    build_residue_validation_table,
    compute_model_placement_audit_stats,
    iter_ca_residues,
    load_cohort_manifest_row,
    read_residue_validation_csv,
)


@dataclass(frozen=True)
class PlacementDecouplingRow:
    emdb_id: str
    display_name: str
    global_resolution_a: float
    n_in_mask: int
    frac_in_contour_mask: float
    rho_rel_vs_cc: float
    rho_t_rank_vs_cc: float
    rho_h_raw_vs_cc: float
    median_cc_omit: float
    median_cc_build: float
    zone_cc_inverted: bool
    frac_omit_zone: float
    frac_cc_below_0_5: float
    tercile_absolute_gap: float
    permutation_p: float
    decoupled: bool
    notes: str = ""


def _spearman(x: np.ndarray, y: np.ndarray, *, min_pairs: int = 30) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < min_pairs:
        return float("nan")
    rho, _ = stats.spearmanr(x[m], y[m])
    return float(rho)


def _in_mask_arrays(rows: Sequence[ResidueValidationRow]) -> tuple[np.ndarray, ...]:
    cc: list[float] = []
    rel: list[float] = []
    zones: list[int] = []
    for r in rows:
        if not r.in_contour_mask:
            continue
        if not (np.isfinite(r.windowed_halfmap_correlation) and np.isfinite(r.reliability_score)):
            continue
        cc.append(float(r.windowed_halfmap_correlation))
        rel.append(float(r.reliability_score))
        zones.append(int(r.build_zone))
    return (
        np.asarray(cc, dtype=np.float64),
        np.asarray(rel, dtype=np.float64),
        np.asarray(zones, dtype=np.int32),
    )


def median_cc_by_zone(cc: np.ndarray, zones: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for z in (0, 1, 2):
        m = zones == z
        if m.any():
            out[z] = float(np.median(cc[m]))
    return out


def permutation_p_value(rel: np.ndarray, cc: np.ndarray, *, n_perm: int = 999) -> float:
    """Two-sided sign test vs within-map rank shuffle of reliability."""
    m = np.isfinite(rel) & np.isfinite(cc)
    rel_f = rel[m]
    cc_f = cc[m]
    if rel_f.size < 30:
        return float("nan")
    obs = _spearman(rel_f, cc_f, min_pairs=30)
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        shuf = rng.permutation(rel_f)
        rho = _spearman(shuf, cc_f, min_pairs=30)
        if np.isfinite(rho) and abs(rho) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def analyze_residue_rows(
    rows: Sequence[ResidueValidationRow],
    *,
    emdb_id: str,
    display_name: str = "",
    global_resolution_a: float = float("nan"),
    cc_threshold: float = 0.5,
    n_perm: int = 999,
) -> PlacementDecouplingRow | None:
    audit = compute_model_placement_audit_stats(
        rows,
        emdb_id=emdb_id,
        display_name=display_name,
        global_resolution_a=global_resolution_a,
        cc_threshold=cc_threshold,
    )
    cc, rel, zones = _in_mask_arrays(rows)
    if cc.size < 30:
        return None

    med = median_cc_by_zone(cc, zones)
    med_omit = med.get(0, float("nan"))
    med_build = med.get(2, float("nan"))
    inverted = bool(np.isfinite(med_omit) and np.isfinite(med_build) and med_omit > med_build)

    rho = float(audit.spearman_reliability_vs_cc)
    return PlacementDecouplingRow(
        emdb_id=emdb_id,
        display_name=display_name,
        global_resolution_a=global_resolution_a,
        n_in_mask=int(cc.size),
        frac_in_contour_mask=float(audit.frac_in_contour_mask),
        rho_rel_vs_cc=rho,
        rho_t_rank_vs_cc=float("nan"),
        rho_h_raw_vs_cc=float("nan"),
        median_cc_omit=med_omit,
        median_cc_build=med_build,
        zone_cc_inverted=inverted,
        frac_omit_zone=float(audit.frac_in_omit_zone),
        frac_cc_below_0_5=float(audit.frac_cc_below_0_50),
        tercile_absolute_gap=abs(float(audit.frac_in_omit_zone) - float(audit.frac_cc_below_0_50)),
        permutation_p=permutation_p_value(rel, cc, n_perm=n_perm),
        decoupled=bool(np.isfinite(rho) and rho < 0),
        notes=audit.notes,
    )


def _zscore_global(volume: np.ndarray) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float64)
    mu = float(v.mean())
    sig = float(v.std())
    return ((v - mu) / (sig + 1e-6)).astype(np.float32)


def recompute_rho_at_ca(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    rho_source: str = "avg_half",
    contour_scale: float = 1.0,
    cc_window: int = 5,
    lh_window: int = 5,
) -> dict[str, float]:
    """
    Re-sample reliability variants at deposited Cα.

    ``rho_source``: ``avg_half`` (canonical), ``primary`` (sharpened deposit), or ``t_only``.
    """
    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = Path(row["reference_mrc"])
    pdb_path = Path(row["flexibility_path_or_pdb"])
    contour = float(row["contour"]) * float(contour_scale)

    bundle = load_full_and_half_maps(
        ref_path,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        reference="full",
        dtype=np.float32,
        resample_if_needed=True,
    )
    ref_grid = bundle.full
    ref_vol = np.asarray(ref_grid.data, dtype=np.float32)
    mask = build_contour_mask(ref_vol, contour)
    if rho_source == "primary":
        rho = _zscore_global(ref_vol)
    else:
        rho = _zscore_global(0.5 * (bundle.half1.data + bundle.half2.data))

    if cc_window != 5:
        cc_vol = half_map_local_metrics(
            bundle.half1.data, bundle.half2.data, window=cc_window
        )[WINDOWED_HALFMAP_CORRELATION_KEY]
    else:
        hm = halfmap_metrics_npz(emd_id)
        if not hm.is_file():
            cc_vol = half_map_local_metrics(
                bundle.half1.data, bundle.half2.data, window=cc_window
            )[WINDOWED_HALFMAP_CORRELATION_KEY]
        else:
            with np.load(hm, allow_pickle=False) as d:
                cc_vol = load_windowed_halfmap_correlation(d)

    if rho_source == "t_only":
        delta = bundle.half1.data - bundle.half2.data
        work: dict[str, np.ndarray] = {"density_normalized": rho}
        attach_reliability_to_features(
            work, bundle.half1.data, bundle.half2.data, window=lh_window, mask=mask, compute_zones=False
        )
        rel_vol = percentile_rank_in_mask(work["reliability_fluctuation"], mask)
        h_vol = work["reliability_fluctuation"]
    else:
        work = {"density_normalized": rho}
        attach_reliability_to_features(
            work, bundle.half1.data, bundle.half2.data, window=lh_window, mask=mask, compute_zones=True
        )
        rel_vol = work["reliability_score"]
        h_vol = work["reliability_H_repro"]

    residues = iter_ca_residues(pdb_path)
    rows = build_residue_validation_table(
        residues,
        grid=ref_grid,
        reference_density=ref_vol,
        contour=contour,
        reliability_score=rel_vol,
        reliability_H_repro=h_vol,
        build_zone=work.get("build_zone", np.zeros_like(rel_vol, dtype=np.uint8)),
        windowed_halfmap_correlation=cc_vol,
    )
    cc, rel, _ = _in_mask_arrays(rows)
    h_vals = np.array(
        [
            r.reliability_H_repro
            for r in rows
            if r.in_contour_mask and np.isfinite(r.windowed_halfmap_correlation) and np.isfinite(r.reliability_H_repro)
        ],
        dtype=np.float64,
    )
    return {
        "rho_rel_vs_cc": _spearman(rel, cc),
        "rho_h_raw_vs_cc": _spearman(h_vals, cc) if h_vals.size == cc.size else float("nan"),
        "n_in_mask": float(cc.size),
        "contour": contour,
    }


def load_decoupling_cohort(
    *,
    manifest: Path = COHORT_MANIFEST,
    min_in_mask: int = 30,
    min_frac_mask: float = 0.3,
) -> list[PlacementDecouplingRow]:
    out: list[PlacementDecouplingRow] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            emd_id = str(row["emdb_id"]).strip()
            pdb = Path(row.get("flexibility_path_or_pdb", ""))
            if pdb.suffix.lower() not in {".cif", ".pdb"} or not pdb.is_file():
                continue
            rv = lh_map_reliability_dir(emd_id) / "residue_validation.csv"
            if not rv.is_file():
                continue
            residues = read_residue_validation_csv(rv)
            gr = row.get("global_resolution_a", "").strip()
            try:
                global_res = float(gr) if gr else float("nan")
            except ValueError:
                global_res = float("nan")
            rec = analyze_residue_rows(
                residues,
                emdb_id=emd_id,
                display_name=str(row.get("display_name", "")).strip(),
                global_resolution_a=global_res,
            )
            if rec is None:
                continue
            if rec.frac_in_contour_mask < min_frac_mask:
                rec = PlacementDecouplingRow(
                    **{**rec.__dict__, "notes": (rec.notes + " low mask coverage").strip()}
                )
            if rec.n_in_mask >= min_in_mask:
                out.append(rec)
    return out


def write_decoupling_csv(path: Path, rows: Sequence[PlacementDecouplingRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "emdb_id",
        "display_name",
        "global_resolution_a",
        "n_in_mask",
        "frac_in_contour_mask",
        "rho_rel_vs_cc",
        "rho_t_rank_vs_cc",
        "rho_h_raw_vs_cc",
        "median_cc_omit",
        "median_cc_build",
        "zone_cc_inverted",
        "frac_omit_zone",
        "frac_cc_below_0_5",
        "tercile_absolute_gap",
        "permutation_p",
        "decoupled",
        "notes",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "display_name": r.display_name,
                    "global_resolution_a": (
                        f"{r.global_resolution_a:.2f}" if np.isfinite(r.global_resolution_a) else ""
                    ),
                    "n_in_mask": r.n_in_mask,
                    "frac_in_contour_mask": f"{r.frac_in_contour_mask:.4f}",
                    "rho_rel_vs_cc": f"{r.rho_rel_vs_cc:+.4f}",
                    "rho_t_rank_vs_cc": (
                        f"{r.rho_t_rank_vs_cc:+.4f}" if np.isfinite(r.rho_t_rank_vs_cc) else ""
                    ),
                    "rho_h_raw_vs_cc": (
                        f"{r.rho_h_raw_vs_cc:+.4f}" if np.isfinite(r.rho_h_raw_vs_cc) else ""
                    ),
                    "median_cc_omit": f"{r.median_cc_omit:.4f}",
                    "median_cc_build": f"{r.median_cc_build:.4f}",
                    "zone_cc_inverted": int(r.zone_cc_inverted),
                    "frac_omit_zone": f"{r.frac_omit_zone:.4f}",
                    "frac_cc_below_0_5": f"{r.frac_cc_below_0_5:.4f}",
                    "tercile_absolute_gap": f"{r.tercile_absolute_gap:.4f}",
                    "permutation_p": f"{r.permutation_p:.6f}",
                    "decoupled": int(r.decoupled),
                    "notes": r.notes,
                }
            )
    return path
