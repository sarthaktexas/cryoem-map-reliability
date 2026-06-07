"""Export tidy CSVs for Prism / Illustrator figure production.

Writes under ``outputs/figure_data/``:

- ``emd_<ID>_residue_tv_b.csv`` — Cα sphere samples: T, V, L, H, B_iso, CC, variance
- ``emd_<ID>_voxel_tv_cc.csv`` — masked voxel subsample for scatter plots
- ``emd_<ID>_binned_v_vs_cc.csv`` — quantile-binned mean V vs CC (line/error bars)
- ``emd_<ID>_binned_b_vs_v.csv`` — quantile-binned mean B_iso vs V at residues
- ``conformation_pair_emd_<A>_vs_<B>.csv`` — matched ΔB, ΔT, ΔV, ΔL, Δreliability

Example::

    source .venv/bin/activate
    python scripts/export_figure_data.py
    python scripts/export_figure_data.py --emd-id 49450 --voxel-samples 8000
    python scripts/export_figure_data.py --pair 23129 23130 --pair 49450 48923
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from cryoem_mrc.analysis import binned_feature_by_target, build_contour_mask
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.repo_paths import (
    COHORT_MANIFEST,
    OUTPUTS_ROOT,
    halfmap_metrics_npz,
    lh_map_reliability_dir,
)
from cryoem_mrc.structure_validation import (
    iter_ca_residues,
    load_cohort_manifest_row,
    physical_xyz_to_voxel_indices,
    sample_volume_at_ca,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = OUTPUTS_ROOT / "figure_data"


def _contour_tag(contour: float) -> str:
    return f"t{int(round(float(contour) * 1000)):04d}"


def _find_features_npz(data_dir: Path, emd: str, contour: float) -> Path | None:
    tag = _contour_tag(contour)
    candidates = [
        data_dir / f"{emd}_avg_features_{tag}.npz",
        data_dir / f"{emd}_avg_features_t{int(round(contour * 1000)):04d}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_dir.glob(f"{emd}_avg_features*.npz"))
    return matches[0] if matches else None


def _load_volumes_for_emd(emd_id: str, *, manifest: Path) -> dict:
    """Load LH maps, CC, variance, mask, and paths for one EMDB entry."""

    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = Path(row["reference_mrc"])
    if not ref_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing reference {ref_path}")

    contour = float(row["contour"])
    data_dir = ref_path.parent
    emd = f"emd_{emd_id}"
    rel_npz = lh_map_reliability_dir(emd_id) / "reliability.npz"
    if not rel_npz.is_file():
        raise FileNotFoundError(
            f"EMD-{emd_id}: missing {rel_npz} — run scripts/run_lh_map_reliability_export.py first"
        )

    grid = load_map_grid(ref_path, dtype=np.float32)
    ref = np.asarray(grid.data, dtype=np.float32)
    mask = build_contour_mask(ref, contour)

    with np.load(rel_npz, allow_pickle=False) as d:
        t = np.asarray(d["reliability_fluctuation"], dtype=np.float32)
        v = np.asarray(d["reliability_smoothness"], dtype=np.float32)
        h = np.asarray(d["reliability_H_repro"], dtype=np.float32)
        score = np.asarray(d["reliability_score"], dtype=np.float32)

    l_balance = (t - v).astype(np.float32)

    cc_path = halfmap_metrics_npz(emd_id)
    if not cc_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing {cc_path} — run scripts/run_analysis.py first")
    with np.load(cc_path, allow_pickle=False) as hm:
        cc = np.asarray(hm["local_cross_correlation"], dtype=np.float32)

    local_var = np.full(ref.shape, np.nan, dtype=np.float32)
    feat_path = _find_features_npz(data_dir, emd, contour)
    if feat_path is not None:
        with np.load(feat_path, allow_pickle=False) as feat:
            local_var = np.asarray(feat["local_variance"], dtype=np.float32)

    pdb = row.get("flexibility_path_or_pdb", "").strip()
    pdb_path = Path(pdb) if pdb else None

    return {
        "emd_id": emd_id,
        "contour": contour,
        "ref_path": ref_path,
        "pdb_path": pdb_path,
        "grid": grid,
        "mask": mask,
        "lh_T": t,
        "lh_V": v,
        "lh_L": l_balance,
        "lh_H": h,
        "reliability_score": score,
        "halfmap_CC": cc,
        "local_variance": local_var,
    }


def _residue_in_mask(residue, grid, mask_vol: np.ndarray) -> bool:
    iz, iy, ix = physical_xyz_to_voxel_indices(residue.x, residue.y, residue.z, grid)
    return _voxel_in_mask(mask_vol, iz, iy, ix)


def export_residue_tv_b(
    ctx: dict,
    *,
    out_path: Path,
    sphere_radius_a: float = 2.0,
) -> int:
    """Residue-level T/V/B table for Prism scatter (Fig 3A / dual validation)."""
    pdb_path = ctx.get("pdb_path")
    if pdb_path is None or not Path(pdb_path).is_file():
        print(f"[export] skip residue CSV: no PDB for EMD-{ctx['emd_id']}", file=sys.stderr)
        return 0

    residues = iter_ca_residues(pdb_path)
    grid = ctx["grid"]
    volumes = {
        "lh_T": ctx["lh_T"],
        "lh_V": ctx["lh_V"],
        "lh_L": ctx["lh_L"],
        "lh_H": ctx["lh_H"],
        "reliability_score": ctx["reliability_score"],
        "halfmap_CC": ctx["halfmap_CC"],
        "local_variance": ctx["local_variance"],
    }
    sampled = {
        name: sample_volume_at_ca(vol, grid, residues, sphere_radius_a=sphere_radius_a)
        for name, vol in volumes.items()
    }
    b = np.array([r.b_iso for r in residues], dtype=np.float64)

    fieldnames = [
        "chain",
        "seq_num",
        "seq_icode",
        "res_name",
        "x",
        "y",
        "z",
        "b_iso",
        "lh_T",
        "lh_V",
        "lh_L",
        "lh_H",
        "reliability_score",
        "halfmap_CC",
        "local_variance",
        "in_contour_mask",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        ref = np.asarray(grid.data, dtype=np.float32)
        mask_vol = build_contour_mask(ref, float(ctx["contour"]))
        for i, res in enumerate(residues):
            in_mask = _residue_in_mask(res, grid, mask_vol)
            if in_mask:
                n_in += 1
            row = {
                "chain": res.chain,
                "seq_num": res.seq_num,
                "seq_icode": res.seq_icode,
                "res_name": res.res_name,
                "x": f"{res.x:.3f}",
                "y": f"{res.y:.3f}",
                "z": f"{res.z:.3f}",
                "b_iso": f"{res.b_iso:.2f}",
                "in_contour_mask": int(in_mask),
            }
            for key in (
                "lh_T",
                "lh_V",
                "lh_L",
                "lh_H",
                "reliability_score",
                "halfmap_CC",
                "local_variance",
            ):
                val = float(sampled[key][i])
                row[key] = f"{val:.6f}" if np.isfinite(val) else ""
            w.writerow(row)

    rho_b_v = _spearman(b, sampled["lh_V"])
    rho_b_t = _spearman(b, sampled["lh_T"])
    print(
        f"[export] residue {out_path.name}: n={len(residues)} in_mask={n_in} "
        f"rho(B,V)={rho_b_v:+.3f} rho(B,T)={rho_b_t:+.3f}",
        flush=True,
    )
    return len(residues)


def export_voxel_subsample(
    ctx: dict,
    *,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> int:
    """Masked voxel subsample: T, V, L vs CC for Prism (Fig 2B)."""
    mask = np.asarray(ctx["mask"], dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        print(f"[export] skip voxel CSV: empty mask EMD-{ctx['emd_id']}", file=sys.stderr)
        return 0
    if idx.size > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, n_samples, replace=False)

    fields = ["lh_T", "lh_V", "lh_L", "lh_H", "halfmap_CC", "local_variance", "reliability_score"]
    flat = {name: np.asarray(ctx[name], dtype=np.float64).ravel()[idx] for name in fields}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(idx.size):
            w.writerow(
                {
                    name: f"{flat[name][i]:.6f}" if np.isfinite(flat[name][i]) else ""
                    for name in fields
                }
            )

    rho_t_cc = _spearman(flat["lh_T"], flat["halfmap_CC"])
    rho_v_cc = _spearman(flat["lh_V"], flat["halfmap_CC"])
    print(
        f"[export] voxel {out_path.name}: n={idx.size} rho(T,CC)={rho_t_cc:+.3f} rho(V,CC)={rho_v_cc:+.3f}",
        flush=True,
    )
    return int(idx.size)


def export_binned_v_vs_cc(ctx: dict, *, out_path: Path, n_bins: int = 12) -> None:
    """Quantile-binned mean lh_V per half-map CC bin."""
    binned = binned_feature_by_target(
        ctx["lh_V"],
        ctx["halfmap_CC"],
        ctx["mask"],
        feature_name="lh_V",
        target_name="halfmap_CC",
        n_bins=n_bins,
    )
    _write_binned_csv(out_path, binned, x_name="halfmap_CC", y_name="lh_V")


def export_binned_b_vs_v(
    ctx: dict,
    *,
    out_path: Path,
    sphere_radius_a: float = 2.0,
    n_bins: int = 12,
) -> None:
    """Quantile-binned mean B_iso per lh_V bin (residue level)."""
    pdb_path = ctx.get("pdb_path")
    if pdb_path is None or not Path(pdb_path).is_file():
        return
    residues = iter_ca_residues(pdb_path)
    grid = ctx["grid"]
    v_s = sample_volume_at_ca(ctx["lh_V"], grid, residues, sphere_radius_a=sphere_radius_a)
    b = np.array([r.b_iso for r in residues], dtype=np.float64)
    mask_vol = build_contour_mask(np.asarray(grid.data, dtype=np.float32), float(ctx["contour"]))
    in_mask = np.array(
        [_residue_in_mask(r, grid, mask_vol) and np.isfinite(v_s[i]) and np.isfinite(b[i])
         for i, r in enumerate(residues)],
        dtype=bool,
    )

    m = int(in_mask.sum())
    if m < 30:
        return
    edges = np.quantile(v_s[in_mask], np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.clip(np.digitize(v_s[in_mask], edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["bin", "lh_V_center", "mean_b_iso", "std_b_iso", "count"],
        )
        w.writeheader()
        for b_idx in range(n_bins):
            sel = b[in_mask][bin_idx == b_idx]
            w.writerow(
                {
                    "bin": b_idx,
                    "lh_V_center": f"{centers[b_idx]:.6f}",
                    "mean_b_iso": f"{float(sel.mean()):.2f}" if sel.size else "",
                    "std_b_iso": f"{float(sel.std()):.2f}" if sel.size > 1 else "",
                    "count": int(sel.size),
                }
            )
    print(f"[export] binned {out_path.name}: n={m}", flush=True)


def _write_binned_csv(path: Path, binned, *, x_name: str, y_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                f"{x_name}_center",
                f"mean_{y_name}",
                f"std_{y_name}",
                "count",
            ],
        )
        w.writeheader()
        for i in range(len(binned.bin_centers)):
            w.writerow(
                {
                    "bin": i,
                    f"{x_name}_center": f"{binned.bin_centers[i]:.6f}",
                    f"mean_{y_name}": (
                        f"{binned.mean_feature[i]:.6f}" if np.isfinite(binned.mean_feature[i]) else ""
                    ),
                    f"std_{y_name}": (
                        f"{binned.std_feature[i]:.6f}" if np.isfinite(binned.std_feature[i]) else ""
                    ),
                    "count": int(binned.count[i]),
                }
            )
    print(f"[export] binned {path.name}", flush=True)


def export_conformation_pair(
    emd_a: str,
    emd_b: str,
    *,
    manifest: Path,
    out_path: Path,
    sphere_radius_a: float = 2.0,
) -> int:
    """Matched Cα ΔB, ΔT, ΔV, ΔL, Δreliability for Prism (Fig 4)."""
    ctx_a = _load_volumes_for_emd(emd_a, manifest=manifest)
    ctx_b = _load_volumes_for_emd(emd_b, manifest=manifest)
    pdb_a = ctx_a.get("pdb_path")
    pdb_b = ctx_b.get("pdb_path")
    if pdb_a is None or pdb_b is None or not pdb_a.is_file() or not pdb_b.is_file():
        raise FileNotFoundError(f"conformation pair {emd_a}/{emd_b}: missing PDB")

    res_a = iter_ca_residues(pdb_a)
    res_b = iter_ca_residues(pdb_b)
    index_b = {r.residue_key: (i, r) for i, r in enumerate(res_b)}
    grid_a = ctx_a["grid"]
    grid_b = ctx_b["grid"]
    mask_a = build_contour_mask(np.asarray(grid_a.data, dtype=np.float32), float(ctx_a["contour"]))
    mask_b = build_contour_mask(np.asarray(grid_b.data, dtype=np.float32), float(ctx_b["contour"]))
    samples_a = _batch_residue_samples(ctx_a, res_a, sphere_radius_a)
    samples_b = _batch_residue_samples(ctx_b, res_b, sphere_radius_a)

    fieldnames = [
        "chain",
        "seq_num",
        "seq_icode",
        "res_name",
        "b_iso_a",
        "b_iso_b",
        "delta_b_iso",
        "lh_T_a",
        "lh_T_b",
        "delta_lh_T",
        "lh_V_a",
        "lh_V_b",
        "delta_lh_V",
        "lh_L_a",
        "lh_L_b",
        "delta_lh_L",
        "reliability_a",
        "reliability_b",
        "delta_reliability",
        "lh_H_a",
        "lh_H_b",
        "delta_lh_H",
        "in_mask_a",
        "in_mask_b",
        "in_mask_both",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    rows_for_rho: list[dict[str, float]] = []

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ia, ra in enumerate(res_a):
            hit = index_b.get(ra.residue_key)
            if hit is None:
                continue
            ib, rb = hit
            sa = {k: float(samples_a[k][ia]) for k in samples_a}
            sb = {k: float(samples_b[k][ib]) for k in samples_b}
            in_a = _residue_in_mask(ra, grid_a, mask_a)
            in_b = _residue_in_mask(rb, grid_b, mask_b)
            in_both = in_a and in_b
            db = rb.b_iso - ra.b_iso
            w.writerow(
                {
                    "chain": ra.chain,
                    "seq_num": ra.seq_num,
                    "seq_icode": ra.seq_icode,
                    "res_name": ra.res_name,
                    "b_iso_a": f"{ra.b_iso:.2f}",
                    "b_iso_b": f"{rb.b_iso:.2f}",
                    "delta_b_iso": f"{db:.2f}",
                    "lh_T_a": _fmt(sa["lh_T"]),
                    "lh_T_b": _fmt(sb["lh_T"]),
                    "delta_lh_T": _fmt(sb["lh_T"] - sa["lh_T"]),
                    "lh_V_a": _fmt(sa["lh_V"]),
                    "lh_V_b": _fmt(sb["lh_V"]),
                    "delta_lh_V": _fmt(sb["lh_V"] - sa["lh_V"]),
                    "lh_L_a": _fmt(sa["lh_L"]),
                    "lh_L_b": _fmt(sb["lh_L"]),
                    "delta_lh_L": _fmt(sb["lh_L"] - sa["lh_L"]),
                    "reliability_a": _fmt(sa["reliability_score"]),
                    "reliability_b": _fmt(sb["reliability_score"]),
                    "delta_reliability": _fmt(sb["reliability_score"] - sa["reliability_score"]),
                    "lh_H_a": _fmt(sa["lh_H"]),
                    "lh_H_b": _fmt(sb["lh_H"]),
                    "delta_lh_H": _fmt(sb["lh_H"] - sa["lh_H"]),
                    "in_mask_a": int(in_a),
                    "in_mask_b": int(in_b),
                    "in_mask_both": int(in_both),
                }
            )
            n_written += 1
            if in_both and np.isfinite(db):
                rows_for_rho.append(
                    {
                        "delta_b": db,
                        "delta_v": sb["lh_V"] - sa["lh_V"],
                        "delta_l": sb["lh_L"] - sa["lh_L"],
                        "delta_rel": sb["reliability_score"] - sa["reliability_score"],
                        "delta_h": sb["lh_H"] - sa["lh_H"],
                    }
                )

    if rows_for_rho:
        db = np.array([r["delta_b"] for r in rows_for_rho])
        for label, key in (
            ("ΔV", "delta_v"),
            ("ΔL", "delta_l"),
            ("Δrel", "delta_rel"),
            ("ΔH", "delta_h"),
        ):
            y = np.array([r[key] for r in rows_for_rho])
            m = np.isfinite(db) & np.isfinite(y)
            if m.sum() >= 10:
                print(
                    f"[export] pair {emd_a}/{emd_b} rho(ΔB,{label})="
                    f"{_spearman(db[m], y[m]):+.3f} (n={int(m.sum())})",
                    flush=True,
                )
    print(f"[export] pair {out_path.name}: rows={n_written}", flush=True)
    return n_written


def _batch_residue_samples(
    ctx: dict, residues: list, radius_a: float
) -> dict[str, np.ndarray]:
    grid = ctx["grid"]
    keys = ("lh_T", "lh_V", "lh_L", "lh_H", "reliability_score")
    return {
        name: sample_volume_at_ca(ctx[name], grid, residues, sphere_radius_a=radius_a)
        for name in keys
    }


def _voxel_in_mask(mask: np.ndarray, iz: int, iy: int, ix: int) -> bool:
    return (
        0 <= iz < mask.shape[0]
        and 0 <= iy < mask.shape[1]
        and 0 <= ix < mask.shape[2]
        and bool(mask[iz, iy, ix])
    )


def _fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else ""


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[m], y[m]).statistic)


def _write_readme(out_dir: Path, files: list[str]) -> None:
    text = """# Figure data exports (Prism / Illustrator)

Generated by ``scripts/export_figure_data.py``.

| File | Use in Prism |
|------|----------------|
| ``emd_*_residue_tv_b.csv`` | Scatter: B_iso vs lh_V (filter in_contour_mask=1) |
| ``emd_*_voxel_tv_cc.csv`` | Scatter: lh_T vs halfmap_CC; lh_V vs halfmap_CC |
| ``emd_*_binned_v_vs_cc.csv`` | Line + error: mean lh_V vs CC quantile bins |
| ``emd_*_binned_b_vs_v.csv`` | Line + error: mean B_iso vs lh_V quantile bins |
| ``conformation_pair_*.csv`` | Scatter: delta_b_iso vs delta_lh_V (filter in_mask_both=1) |

Also available without re-export: ``outputs/cohort_summary/cohort_metrics.csv``
"""
    (out_dir / "README.md").write_text(text)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", action="append", default=["49450"], help="Map(s) for residue/voxel exports")
    p.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        default=None,
        help="Conformation pair(s) to export (default: 23129/23130 and 49450/48923)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--voxel-samples", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--skip-binned", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for emd_id in args.emd_id:
        eid = str(emd_id).strip()
        try:
            ctx = _load_volumes_for_emd(eid, manifest=args.manifest)
        except FileNotFoundError as exc:
            print(f"[export] ERROR EMD-{eid}: {exc}", file=sys.stderr)
            return 2

        residue_path = out_dir / f"emd_{eid}_residue_tv_b.csv"
        export_residue_tv_b(ctx, out_path=residue_path, sphere_radius_a=args.sphere_radius_a)
        written.append(residue_path.name)

        voxel_path = out_dir / f"emd_{eid}_voxel_tv_cc.csv"
        export_voxel_subsample(
            ctx, out_path=voxel_path, n_samples=args.voxel_samples, seed=args.seed
        )
        written.append(voxel_path.name)

        if not args.skip_binned:
            binned_cc = out_dir / f"emd_{eid}_binned_v_vs_cc.csv"
            export_binned_v_vs_cc(ctx, out_path=binned_cc, n_bins=args.n_bins)
            written.append(binned_cc.name)
            binned_b = out_dir / f"emd_{eid}_binned_b_vs_v.csv"
            export_binned_b_vs_v(
                ctx, out_path=binned_b, sphere_radius_a=args.sphere_radius_a, n_bins=args.n_bins
            )
            written.append(binned_b.name)

    pair_list = args.pair if args.pair else [["23129", "23130"], ["49450", "48923"]]
    seen_pairs: set[tuple[str, str]] = set()
    for emd_a, emd_b in pair_list:
        key = (str(emd_a).strip(), str(emd_b).strip())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pair_path = out_dir / f"conformation_pair_emd_{key[0]}_vs_{key[1]}.csv"
        try:
            export_conformation_pair(
                key[0],
                key[1],
                manifest=args.manifest,
                out_path=pair_path,
                sphere_radius_a=args.sphere_radius_a,
            )
            written.append(pair_path.name)
        except FileNotFoundError as exc:
            print(f"[export] WARN pair {emd_a}/{emd_b}: {exc}", file=sys.stderr)

    _write_readme(out_dir, written)
    print(f"[export] done -> {out_dir} ({len(written)} files)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
