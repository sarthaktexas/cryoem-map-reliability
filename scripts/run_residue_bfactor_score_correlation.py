"""Residue-level aggregation of map score volumes vs deposited B-factors.

Builds a full grid: score_map × sphere_radius × aggregation (mean/median),
with mask QC, B-factor preflight, and collinearity vs local_variance.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_residue_bfactor_score_correlation.py --emd-id 49450
    python scripts/run_residue_bfactor_score_correlation.py --emd-id 49450 --with-median
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.hessian import density_hessian_scalar_maps
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.mechanics import fluctuation_constraint_decomposition
from cryoem_mrc.repo_paths import COHORT_MANIFEST, halfmap_metrics_npz
from cryoem_mrc.structure_validation import (
    BfactorDistributionSummary,
    build_ca_sphere_index_caches,
    compute_bfactor_score_correlation_rows,
    iter_ca_residues,
    load_cohort_manifest_row,
    mask_fraction_in_sphere_caches,
    rank_normalize_b_per_chain,
    summarize_b_iso_distribution,
    write_bfactor_score_correlation_csv,
)

DEFAULT_RADII = (1.5, 2.0, 2.5)
DEFAULT_SCORE_ORDER = (
    "local_variance",
    "local_cross_correlation",
    "hessian_frobenius",
    "hessian_trace",
    "lagrangian",
    "hamiltonian",
)


def _mrc_origin_preflight(reference: Path) -> dict:
    """Report MRC origin vs nstart* voxel_size (diagnostic for alignment)."""
    import mrcfile

    with mrcfile.open(reference, permissive=True) as mrc:
        h = mrc.header
        vs = mrc.voxel_size
        origin = (float(h.origin.x), float(h.origin.y), float(h.origin.z))
        nstart = (float(h.nxstart), float(h.nystart), float(h.nzstart))
        nstart_a = (
            nstart[0] * float(vs.x),
            nstart[1] * float(vs.y),
            nstart[2] * float(vs.z),
        )
    o_norm = max(abs(x) for x in origin)
    n_norm = max(abs(x) for x in nstart_a)
    diff = max(abs(origin[i] - nstart_a[i]) for i in range(3))
    notes: list[str] = []
    if o_norm > 1e-3 and n_norm > 1e-3 and diff > 0.5:
        notes.append(
            f"origin field ({origin}) and nstart*vs ({nstart_a}) differ by up to {diff:.2f} Å; "
            "load_map_grid uses their sum."
        )
    return {
        "origin_xyz": origin,
        "nstart_voxels": nstart,
        "nstart_offset_angstrom": nstart_a,
        "notes": notes,
    }


def _load_score_maps(
    *,
    data_dir: Path,
    emd: str,
    features_npz: Path,
    halfmap_npz: Path,
    lh_window: int,
    skip_hessian: bool,
    chunk_z: int,
) -> dict[str, np.ndarray]:
    with np.load(features_npz, allow_pickle=False) as z:
        rho = np.asarray(z["density_normalized"], dtype=np.float32)
        score_maps: dict[str, np.ndarray] = {
            "local_variance": np.asarray(z["local_variance"], dtype=np.float32),
        }

    with np.load(halfmap_npz, allow_pickle=False) as z:
        score_maps["local_cross_correlation"] = np.asarray(
            z["local_cross_correlation"], dtype=np.float32
        )

    bundle = load_full_and_half_maps(
        data_dir / f"{emd}.map",
        data_dir / f"{emd}_half_map_1.map",
        data_dir / f"{emd}_half_map_2.map",
    )
    delta = bundle.half1.data.astype(np.float32) - bundle.half2.data.astype(np.float32)
    del bundle
    lh = fluctuation_constraint_decomposition(rho, delta, window=lh_window)
    score_maps["lagrangian"] = lh["L_balance"].astype(np.float32)
    score_maps["hamiltonian"] = lh["H_sum"].astype(np.float32)
    del lh, delta
    gc.collect()

    if not skip_hessian:
        hess = density_hessian_scalar_maps(rho, chunk_z=chunk_z)
        score_maps["hessian_frobenius"] = hess["hessian_frobenius"].astype(np.float32)
        score_maps["hessian_trace"] = hess["hessian_trace"].astype(np.float32)
        del hess
        gc.collect()

    return score_maps


def run_one(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reference: Path | None = None,
    pdb: Path | None = None,
    features_npz: Path | None = None,
    halfmap_npz: Path | None = None,
    contour: float | None = None,
    radii: tuple[float, ...] = DEFAULT_RADII,
    with_median: bool = False,
    min_mask_fraction: float = 0.5,
    rank_b_per_chain: bool = False,
    lh_window: int = 5,
    skip_hessian: bool = False,
    chunk_z: int = 32,
    out_dir: Path | None = None,
) -> dict:
    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    data_dir = ref_path.parent
    emd = f"emd_{emd_id}"

    if features_npz is None:
        feat_glob = list(data_dir.glob(f"{emd}_avg_features*.npz"))
        if not feat_glob:
            raise FileNotFoundError(f"No features npz in {data_dir}")
        features_npz = feat_glob[0]
    if halfmap_npz is None:
        halfmap_npz = halfmap_metrics_npz(emd_id)
        if not halfmap_npz.exists():
            raise FileNotFoundError(f"Missing halfmap metrics: {halfmap_npz}")

    for label, p in (("reference", ref_path), ("pdb", pdb_path), ("features", features_npz)):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    out = out_dir or Path("outputs") / f"emd_{emd_id}" / "bfactor_score_correlation"
    out.mkdir(parents=True, exist_ok=True)

    residues = iter_ca_residues(pdb_path)
    b_raw = np.array([r.b_iso for r in residues], dtype=np.float64)
    b_dist = summarize_b_iso_distribution(b_raw)
    b_iso = rank_normalize_b_per_chain(residues, b_raw) if rank_b_per_chain else b_raw

    grid = load_map_grid(ref_path, dtype=np.float32)
    ref = np.asarray(grid.data, dtype=np.float32)
    mask = build_contour_mask(ref, contour_val).astype(np.float32)

    score_maps = _load_score_maps(
        data_dir=data_dir,
        emd=emd,
        features_npz=features_npz,
        halfmap_npz=halfmap_npz,
        lh_window=lh_window,
        skip_hessian=skip_hessian,
        chunk_z=chunk_z,
    )
    for name in DEFAULT_SCORE_ORDER:
        if name not in score_maps:
            raise KeyError(f"Expected score map {name!r} missing (skip_hessian={skip_hessian})")

    aggregations: tuple[str, ...] = ("mean", "median") if with_median else ("mean",)
    sphere_caches = {
        float(r): build_ca_sphere_index_caches(residues, grid, float(r)) for r in radii
    }

    mask_fracs = {r: mask_fraction_in_sphere_caches(mask, sphere_caches[float(r)]) for r in radii}
    residue_mask = mask_fracs[float(radii[0])] >= float(min_mask_fraction)
    mask_policy = f"sphere_mask_fraction>={min_mask_fraction}"

    corr_rows = compute_bfactor_score_correlation_rows(
        score_maps,
        residues,
        b_iso,
        sphere_caches_by_radius=sphere_caches,
        radii_a=radii,
        aggregations=aggregations,  # type: ignore[arg-type]
        atom_mode="CA",
        residue_mask=residue_mask,
        mask_policy=mask_policy,
    )
    # Stable row order for thesis tables
    order = {n: i for i, n in enumerate(DEFAULT_SCORE_ORDER)}
    corr_rows.sort(
        key=lambda r: (
            r.radius_a,
            r.aggregation,
            order.get(r.score_name, 99),
            r.score_name,
        )
    )
    csv_path = write_bfactor_score_correlation_csv(out / "bfactor_score_correlation.csv", corr_rows)

    dropped_pct = 100.0 * (1.0 - residue_mask.sum() / max(len(residues), 1))
    preflight = {
        "emdb_id": emd_id,
        "pdb": str(pdb_path),
        "reference_mrc": str(ref_path),
        "contour": contour_val,
        "grid_origin_zyx": list(grid.origin_zyx),
        "grid_voxel_size_zyx": list(grid.voxel_size_zyx),
        "mrc_origin": _mrc_origin_preflight(ref_path),
        "b_iso_distribution": {
            "n": b_dist.n,
            "mean": b_dist.mean,
            "std": b_dist.std,
            "min": b_dist.min,
            "max": b_dist.max,
            "median": b_dist.median,
            "notes": b_dist.notes,
        },
        "rank_b_per_chain": rank_b_per_chain,
        "n_ca_residues": len(residues),
        "n_residues_in_mask": int(residue_mask.sum()),
        "pct_dropped_by_mask": dropped_pct,
        "radii_a": list(radii),
        "aggregations": list(aggregations),
        "mask_policy": mask_policy,
        "median_mask_fraction_by_radius": {
            str(r): float(np.median(mask_fracs[float(r)])) for r in radii
        },
    }
    (out / "bfactor_score_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")

    print(f"[bfactor_scores] EMD-{emd_id}  Cα={len(residues)}  in_mask={int(residue_mask.sum())}", flush=True)
    print(
        f"  B_iso: n={b_dist.n}  mean={b_dist.mean:.1f}  std={b_dist.std:.1f}  "
        f"range=[{b_dist.min:.1f}, {b_dist.max:.1f}]",
        flush=True,
    )
    if b_dist.notes:
        print(f"  WARNING: {b_dist.notes}", flush=True)
    if dropped_pct > 10.0:
        print(
            f"  WARNING: {dropped_pct:.1f}% residues below mask fraction {min_mask_fraction} "
            "(check origin alignment)",
            flush=True,
        )
    for note in preflight["mrc_origin"].get("notes", []):
        print(f"  MRC: {note}", flush=True)

    print("[bfactor_scores] Spearman ρ(score, B_iso)  [in-mask, mean agg]:", flush=True)
    for r in corr_rows:
        if r.aggregation != "mean" or abs(r.radius_a - 2.0) > 1e-6:
            continue
        print(
            f"  {r.score_name:28s}  rho={r.spearman_rho:+.4f}  "
            f"vs_var={r.spearman_vs_local_variance:+.4f}  n={r.n_used}",
            flush=True,
        )
    print(f"[bfactor_scores] wrote {csv_path}", flush=True)
    return {"preflight": preflight, "rows": corr_rows, "out_dir": str(out)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", required=True, help="EMDB ID (e.g. 49450)")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--pdb", type=Path, default=None)
    p.add_argument("--features-npz", type=Path, default=None)
    p.add_argument("--halfmap-npz", type=Path, default=None)
    p.add_argument("--contour", type=float, default=None)
    p.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    p.add_argument("--with-median", action="store_true", help="Also correlate median sphere aggregation")
    p.add_argument("--min-mask-fraction", type=float, default=0.5,
                   help="Min fraction of sphere voxels inside contour mask")
    p.add_argument("--rank-b-per-chain", action="store_true", help="Rank-normalize B_iso within each chain")
    p.add_argument("--lh-window", type=int, default=5, help="Box window for LH maps")
    p.add_argument("--skip-hessian", action="store_true", help="Skip Hessian (faster smoke test)")
    p.add_argument("--chunk-z", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_one(
            args.emd_id.strip(),
            manifest=args.manifest,
            reference=args.reference,
            pdb=args.pdb,
            features_npz=args.features_npz,
            halfmap_npz=args.halfmap_npz,
            contour=args.contour,
            radii=tuple(args.radii),
            with_median=args.with_median,
            min_mask_fraction=args.min_mask_fraction,
            rank_b_per_chain=args.rank_b_per_chain,
            lh_window=args.lh_window,
            skip_hessian=args.skip_hessian,
            chunk_z=args.chunk_z,
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[bfactor_scores] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
