"""Extended local statistics, Hessian, LH regimes, light ML vs half-map CC.

Compares voxel-level predictors and residue-level sampling (nearest vs sphere).

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_extended_feature_validation.py --emd-id 49450
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.feature_ml import ridge_cv_spearman
from cryoem_mrc.hessian import density_hessian_scalar_maps
from cryoem_mrc.io import load_mrc
from cryoem_mrc.local_stats import (
    gradient_magnitude,
    local_kurtosis_excess,
    local_laplacian,
    local_mean_and_variance,
    local_skewness,
    structure_tensor_fractional_anisotropy,
)
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.mechanics import classify_tv_regime, fluctuation_constraint_decomposition
from cryoem_mrc.repo_paths import halfmap_metrics_npz
from cryoem_mrc.structure_validation import iter_ca_residues, sample_volume_at_ca

MAP_CONFIG = {
    "49450": ("data/emd_49450-mgtA_e2p+e1", 0.116, "emd_49450_avg_features_t0116.npz", "pdb/9nhz.cif"),
    "11638": ("data/emd_11638-atomic_apoferritin", 0.116, "emd_11638_avg_features_t0116.npz", "pdb/7a4m.cif"),
    "23129": ("data/emd_23129-trpv1_ph6a", 0.009, "emd_23129_avg_features_t0009.npz", "pdb/7l2i.cif"),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", default="49450")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=400_000)
    p.add_argument("--chunk-z", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-hessian", action="store_true", help="Skip Hessian (faster smoke test)")
    p.add_argument("--with-entropy", action="store_true", help="Include local entropy (slow, high RAM)")
    p.add_argument("--with-structure-fa", action="store_true", help="Include structure-tensor FA (high RAM)")
    return p.parse_args(argv)


def _extended_stats(
    rho: np.ndarray,
    window: int,
    *,
    with_entropy: bool,
    with_structure_fa: bool,
) -> dict[str, np.ndarray]:
    """Lightweight extended stats (entropy off by default — full-box is RAM-heavy)."""
    w = int(window)
    mean, var = local_mean_and_variance(rho, size=w)
    out: dict[str, np.ndarray] = {
        f"local_mean_w{w}": mean.astype(np.float32),
        f"local_variance_w{w}": var.astype(np.float32),
        f"local_skewness_w{w}": local_skewness(rho, size=w).astype(np.float32),
        f"local_kurtosis_excess_w{w}": local_kurtosis_excess(rho, size=w).astype(np.float32),
        "gradient_magnitude": gradient_magnitude(rho).astype(np.float32),
        "local_laplacian": local_laplacian(rho).astype(np.float32),
    }
    if with_structure_fa:
        out[f"structure_tensor_fa_w{w}"] = structure_tensor_fractional_anisotropy(
            rho, smooth_size=w
        ).astype(np.float32)
    if with_entropy:
        from cryoem_mrc.local_stats import local_entropy

        out[f"local_entropy_w{w}"] = local_entropy(rho, size=w).astype(np.float32)
    return out


def _spearman(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    a = x.ravel()[idx].astype(np.float64)
    b = y.ravel()[idx].astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m]).statistic)


def _subsample_indices(mask: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= n:
        return idx
    return np.random.default_rng(seed).choice(idx, size=n, replace=False)


BASE_NPZ_KEYS = (
    "density_normalized",
    "local_variance",
    "local_mean",
    "gradient_magnitude",
    "gauss_s0",
    "gauss_s0_local_variance",
    "gauss_s0_gradient_magnitude",
    "gauss_s1_local_variance",
    "rigidity",
)


def _load_base_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k], dtype=np.float32) for k in BASE_NPZ_KEYS if k in z.files}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.emd_id not in MAP_CONFIG:
        print(f"Unknown emd-id {args.emd_id}; choose from {list(MAP_CONFIG)}", file=sys.stderr)
        return 2

    data_dir, contour, feat_name, pdb_rel = MAP_CONFIG[args.emd_id]
    d = Path(data_dir)
    emd = f"emd_{args.emd_id}"
    out_dir = args.out_dir or Path("outputs") / f"emd_{args.emd_id}" / "extended_feature_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ext_val] EMD-{args.emd_id} window={args.window}", flush=True)
    base = _load_base_features(d / feat_name)
    rho = np.asarray(base["density_normalized"], dtype=np.float32)

    ref = load_mrc(d / f"{emd}.map", dtype=np.float32)
    mask = build_contour_mask(ref, contour)
    idx = _subsample_indices(mask, args.max_samples)

    hm_path = halfmap_metrics_npz(args.emd_id)
    with np.load(hm_path, allow_pickle=False) as z:
        cc = np.asarray(z["local_cross_correlation"], dtype=np.float32)
        mse = np.asarray(z["local_mean_squared_difference"], dtype=np.float32)
        var_diff = np.asarray(z["local_variance_difference"], dtype=np.float32)
        snr = np.asarray(z["local_reproducibility_snr"], dtype=np.float32)

    print("[ext_val] extended local statistics...", flush=True)
    ext = _extended_stats(
        rho, args.window, with_entropy=args.with_entropy, with_structure_fa=args.with_structure_fa
    )

    print("[ext_val] LH decomposition...", flush=True)
    bundle = load_full_and_half_maps(
        d / f"{emd}.map", d / f"{emd}_half_map_1.map", d / f"{emd}_half_map_2.map"
    )
    delta = bundle.half1.data.astype(np.float32) - bundle.half2.data.astype(np.float32)
    del bundle
    lh = fluctuation_constraint_decomposition(rho, delta, window=args.window)
    tv_regime = classify_tv_regime(lh["fluctuation_T"], lh["constraint_V"], mask)

    hess: dict[str, np.ndarray] = {}
    if not args.skip_hessian:
        print("[ext_val] Hessian scalars...", flush=True)
        hess = density_hessian_scalar_maps(rho, chunk_z=args.chunk_z)
        gc.collect()

    predictors: dict[str, np.ndarray] = {}
    for k, v in base.items():
        if v.shape == rho.shape and np.issubdtype(v.dtype, np.floating):
            predictors[k] = v.astype(np.float32)
    predictors.update(ext)
    for k, v in lh.items():
        predictors[f"lh_{k}"] = v.astype(np.float32)
    predictors["tv_regime"] = tv_regime.astype(np.float32)
    predictors.update(hess)
    predictors["halfmap_mse"] = mse
    predictors["halfmap_var_diff"] = var_diff
    predictors["halfmap_snr"] = snr

    rows: list[dict] = []
    for name, arr in predictors.items():
        rows.append({"feature": name, "spearman_vs_cc": _spearman(arr, cc, idx), "level": "voxel"})
    rows.sort(key=lambda r: abs(r["spearman_vs_cc"]), reverse=True)

    csv_path = out_dir / "voxel_spearman_vs_cc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "spearman_vs_cc", "level"])
        w.writeheader()
        w.writerows(rows)

    ml_features = [
        "local_variance",
        "gauss_s0_local_variance",
        "gradient_magnitude",
        "local_laplacian",
        "hessian_trace",
        "hessian_frobenius",
        "hessian_anisotropy",
        f"structure_tensor_fa_w{args.window}",
        f"local_skewness_w{args.window}",
        f"local_kurtosis_excess_w{args.window}",
        "lh_fluctuation_T",
        "lh_constraint_V",
        "lh_L_balance",
        "halfmap_mse",
        "halfmap_var_diff",
    ]
    avail = [f for f in ml_features if f in predictors]
    if len(avail) >= 2:
        x_cols = np.column_stack([predictors[k].ravel()[idx] for k in avail])
        y_col = cc.ravel()[idx]
        ml_full = ridge_cv_spearman(x_cols, y_col)
        ml_var_only = ridge_cv_spearman(
            predictors["local_variance"].ravel()[idx, None], y_col
        )
        ml_gauss = (
            ridge_cv_spearman(predictors["gauss_s0_local_variance"].ravel()[idx, None], y_col)
            if "gauss_s0_local_variance" in predictors
            else {"spearman_rho": float("nan")}
        )
    else:
        ml_full = ml_var_only = ml_gauss = {"spearman_rho": float("nan")}

    residue_rows: list[dict] = []
    pdb_path = Path(pdb_rel)
    if pdb_path.exists():
        print(f"[ext_val] residue sampling from {pdb_path.name}...", flush=True)
        grid = load_map_grid(d / f"{emd}.map")
        residues = iter_ca_residues(pdb_path)
        b = np.array([r.b_iso for r in residues], dtype=np.float64)
        test_feats = ["local_variance", "gauss_s0_local_variance", "lh_L_balance", "hessian_trace"]
        test_feats = [f for f in test_feats if f in predictors]
        for feat in test_feats:
            vol = predictors[feat]
            for label, kwargs in (
                ("nearest_voxel", {"window_radius": 0, "sphere_radius_a": None}),
                ("sphere_2A", {"window_radius": 0, "sphere_radius_a": 2.0}),
                ("sphere_3A", {"window_radius": 0, "sphere_radius_a": 3.0}),
                ("window_5", {"window_radius": 2, "sphere_radius_a": None}),
            ):
                s = sample_volume_at_ca(vol, grid, residues, **kwargs)
                m = np.isfinite(s) & np.isfinite(b)
                rho_b = float(stats.spearmanr(s[m], b[m]).statistic) if m.sum() >= 30 else float("nan")
                residue_rows.append(
                    {"feature": feat, "sampling": label, "spearman_vs_b_iso": rho_b, "n": int(m.sum())}
                )

    summary = {
        "emd_id": args.emd_id,
        "contour": contour,
        "window": args.window,
        "n_mask_voxels": int(mask.sum()),
        "top_voxel_features": rows[:12],
        "ridge_cv_spearman": {
            "all_features": ml_full,
            "local_variance_only": ml_var_only,
            "gauss_s0_local_variance_only": ml_gauss,
            "feature_names": avail,
        },
        "residue_bfactor_spearman": residue_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[ext_val] top voxel predictors vs CC:", flush=True)
    for r in rows[:8]:
        print(f"  {r['feature']:32s}  rho={r['spearman_vs_cc']:+.4f}", flush=True)
    print(
        f"[ext_val] ridge CV Spearman: all={ml_full.get('spearman_rho', float('nan')):+.4f}  "
        f"var_only={ml_var_only.get('spearman_rho', float('nan')):+.4f}",
        flush=True,
    )
    print(f"[ext_val] wrote {csv_path} and {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
