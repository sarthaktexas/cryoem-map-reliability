"""Compare legacy rigidity vs exploratory L/H scores against half-map CC.

Production export uses H_repro only; see docs/LH_MAP_RELIABILITY.md.

Decision 001: features on 0.5*(h1+h2) -> density_normalized in avg features NPZ.
Decision 002: mask from deposited reference at contour (default 0.116), NOT avg map.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/archive/run_rigidity_mechanics_comparison.py \\
        --data-dir data/emd_49450-mgtA_tetramer \\
        --reference data/emd_49450-mgtA_tetramer/emd_49450.map \\
        --features data/emd_49450-mgtA_tetramer/emd_49450_avg_features_t0116.npz \\
        --halfmap-npz outputs/emd_49450/analysis/halfmap_metrics.npz \\
        --contour 0.116
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask, compute_feature_target_correlations, write_correlation_csv
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.mechanics import compute_mechanics_headlines
from cryoem_mrc.repo_paths import DATA_ROOT, archive_rigidity_vs_mechanics_dir, halfmap_metrics_npz
from cryoem_mrc.rigidity import compute_rigidity_map_from_npz

_EXPECTED_MASK_VOXELS_EMD_49450 = (150_000, 350_000)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT / "emd_49450-mgtA_tetramer")
    p.add_argument("--emd-id", type=str, default="49450")
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--half1", type=Path, default=None)
    p.add_argument("--half2", type=Path, default=None)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--halfmap-npz", type=Path, default=halfmap_metrics_npz("49450"))
    p.add_argument("--contour", type=float, default=0.116)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=2_000_000)
    p.add_argument("--out-dir", type=Path, default=archive_rigidity_vs_mechanics_dir())
    return p.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    d = args.data_dir
    emd = f"emd_{args.emd_id}"
    return {
        "reference": args.reference or d / f"{emd}.map",
        "half1": args.half1 or d / f"{emd}_half_map_1.map",
        "half2": args.half2 or d / f"{emd}_half_map_2.map",
        "features": args.features or d / f"{emd}_avg_features_t0116.npz",
    }


def _load_npz_keys(npz_path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in keys:
            if key not in data.files:
                raise KeyError(f"{npz_path.name} missing {key!r}")
            out[key] = np.asarray(data[key], dtype=np.float32)
    return out


def _validate_inputs(
    *,
    emd_id: str,
    contour: float,
    n_mask: int,
    reference: Path,
    features: Path,
    half1: Path,
    half2: Path,
    volume_shape: tuple[int, ...],
) -> None:
    print("[rigidity_vs_mech] --- pipeline provenance ---", flush=True)
    print(f"[rigidity_vs_mech] Decision 002 mask: deposited reference {reference.name}", flush=True)
    print(f"[rigidity_vs_mech]   contour rho >= {contour} on reference (NOT on avg map)", flush=True)
    print(f"[rigidity_vs_mech] Decision 001 features: {features.name}", flush=True)
    print("[rigidity_vs_mech]   rho = density_normalized from avg-half NPZ", flush=True)
    print(f"[rigidity_vs_mech]   delta_rho = {half1.name} - {half2.name}", flush=True)
    print(
        f"[rigidity_vs_mech] masked voxels: {n_mask:,} ({100.0 * n_mask / np.prod(volume_shape):.2f}%)",
        flush=True,
    )
    if emd_id == "49450" and not (_EXPECTED_MASK_VOXELS_EMD_49450[0] <= n_mask <= _EXPECTED_MASK_VOXELS_EMD_49450[1]):
        print(
            f"[rigidity_vs_mech] WARNING: EMD-49450 usually has ~235k masked voxels; got {n_mask:,}. "
            "Check --reference and --contour.",
            file=sys.stderr,
        )
    print("[rigidity_vs_mech] ---------------------------", flush=True)


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if finite.sum() < 10:
        return float("nan"), float("nan")
    xr = stats.rankdata(x[finite])
    yr = stats.rankdata(y[finite])
    zr = stats.rankdata(control[finite])
    if xr.std() == 0 or yr.std() == 0 or zr.std() == 0:
        return float("nan"), float("nan")
    r_xy = float(np.corrcoef(xr, yr)[0, 1])
    r_xz = float(np.corrcoef(xr, zr)[0, 1])
    r_yz = float(np.corrcoef(yr, zr)[0, 1])
    denom = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if denom <= 0.0:
        return float("nan"), float("nan")
    r_partial = (r_xy - r_xz * r_yz) / np.sqrt(denom)
    n = int(finite.sum())
    if abs(r_partial) >= 1.0 - 1e-15:
        return r_partial, 0.0
    t_stat = r_partial * np.sqrt((n - 3) / (1.0 - r_partial * r_partial))
    return float(r_partial), float(2.0 * stats.t.sf(abs(t_stat), df=n - 3))


def _subsample_mask_indices(mask: np.ndarray, max_samples: int, seed: int = 0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= max_samples:
        return idx
    return np.random.default_rng(seed).choice(idx, size=max_samples, replace=False)


def _plot_comparison_bar(rows: list[dict], out_path: Path) -> None:
    order = [
        "local_variance",
        "rigidity",
        "H_repro",
        "rigidity_like_H_repro",
        "hamiltonian",
        "rigidity_like_H",
        "el_residual_norm",
        "rigidity_like_el",
        "lagrangian_density",
    ]
    by_name = {r["feature"]: r for r in rows if r["method"] == "spearman"}
    labels, vals, colors = [], [], []
    for name in order:
        if name not in by_name:
            continue
        labels.append(name)
        vals.append(abs(float(by_name[name]["correlation"])))
        colors.append("#d62728" if name == "rigidity" else "#1f77b4" if name != "local_variance" else "#7f7f7f")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(np.arange(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("|Spearman r| vs local_cross_correlation")
    ax.set_title("Rigidity heuristic vs mechanics (EMD-49450, mask rho>=0.116)")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _resolve_paths(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, p in paths.items():
        if not p.exists():
            print(f"[rigidity_vs_mech] ERROR: missing {label}: {p}", file=sys.stderr)
            return 2
    if not args.halfmap_npz.exists():
        print(f"[rigidity_vs_mech] ERROR: missing halfmap npz: {args.halfmap_npz}", file=sys.stderr)
        return 2

    print(f"[rigidity_vs_mech] loading rho + local_variance from {paths['features'].name}", flush=True)
    base_feats = _load_npz_keys(paths["features"], ("density_normalized", "local_variance"))
    rho = base_feats["density_normalized"]

    print(f"[rigidity_vs_mech] mask from deposited reference {paths['reference'].name}", flush=True)
    reference = load_mrc(paths["reference"], dtype=np.float32)
    if reference.shape != rho.shape:
        print(f"[rigidity_vs_mech] ERROR: reference {reference.shape} != features {rho.shape}", file=sys.stderr)
        return 2
    mask = build_contour_mask(reference, args.contour)
    del reference
    gc.collect()
    n_mask = int(mask.sum())

    _validate_inputs(
        emd_id=args.emd_id,
        contour=args.contour,
        n_mask=n_mask,
        reference=paths["reference"],
        features=paths["features"],
        half1=paths["half1"],
        half2=paths["half2"],
        volume_shape=rho.shape,
    )

    print("[rigidity_vs_mech] computing rigidity (before mechanics — lower peak RAM)", flush=True)
    rigidity = compute_rigidity_map_from_npz(paths["features"], mask=mask)
    gc.collect()

    print("[rigidity_vs_mech] loading halves for delta_rho", flush=True)
    bundle = load_full_and_half_maps(
        paths["reference"], paths["half1"], paths["half2"],
        reference="full", dtype=np.float32, resample_if_needed=True,
    )
    delta_rho = (bundle.half1.data - bundle.half2.data).astype(np.float32, copy=False)
    del bundle
    gc.collect()

    print("[rigidity_vs_mech] computing mechanics headlines", flush=True)
    mech = compute_mechanics_headlines(
        rho, delta_rho,
        alpha=args.alpha, beta=args.beta, window=args.window,
        sigma=args.sigma, kappa=args.kappa,
    )
    del delta_rho, rho
    gc.collect()

    compare_feats = {"rigidity": rigidity, "local_variance": base_feats["local_variance"], **mech}
    del base_feats, mech, rigidity
    gc.collect()

    print("[rigidity_vs_mech] loading half-map CC target", flush=True)
    with np.load(args.halfmap_npz, allow_pickle=False) as hm:
        target = np.asarray(hm["local_cross_correlation"], dtype=np.float32)
    if target.shape != mask.shape:
        print(f"[rigidity_vs_mech] ERROR: CC shape {target.shape} != mask {mask.shape}", file=sys.stderr)
        return 2

    print("[rigidity_vs_mech] correlating vs local_cross_correlation", flush=True)
    result = compute_feature_target_correlations(
        compare_feats, target, mask,
        target_name="local_cross_correlation",
        methods=("pearson", "spearman"),
        max_samples=args.max_samples,
    )
    result.contour = float(args.contour)
    result.extra_metadata = {
        "contour": str(args.contour),
        "mask_source": str(paths["reference"]),
        "features_npz": str(paths["features"]),
        "rho_field": "density_normalized (avg halves)",
        "halfmap_npz": str(args.halfmap_npz),
        "alpha": str(args.alpha), "beta": str(args.beta),
        "sigma": str(args.sigma), "kappa": str(args.kappa),
        "window": str(args.window),
    }
    write_correlation_csv(result, out_dir / "correlations.csv")

    idx = _subsample_mask_indices(mask, args.max_samples or 2_000_000)
    y = target.ravel()[idx]
    control = compare_feats["local_variance"].ravel()[idx]
    headline = [
        "rigidity", "H_repro", "rigidity_like_H_repro", "hamiltonian",
        "rigidity_like_H", "el_residual_norm", "rigidity_like_el",
        "lagrangian_density", "rigidity_like_L",
    ]
    partial_rows = []
    for name in headline:
        if name not in compare_feats:
            continue
        r_partial, p_partial = _partial_spearman(compare_feats[name].ravel()[idx], y, control)
        partial_rows.append({
            "feature": name, "control": "local_variance",
            "partial_spearman": r_partial, "p_value": p_partial, "n_samples": int(idx.size),
        })
    with (out_dir / "partial_correlations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "control", "partial_spearman", "p_value", "n_samples"])
        w.writeheader()
        w.writerows(partial_rows)

    spearman = sorted(
        [c for c in result.correlations if c.method == "spearman"],
        key=lambda c: abs(c.correlation), reverse=True,
    )
    lines = [
        "Rigidity vs mechanics comparison — summary",
        "=" * 50,
        f"Mask: {paths['reference'].name} at rho >= {args.contour}",
        f"Features: {paths['features'].name} (density_normalized = avg halves)",
        f"Masked voxels: {n_mask:,}",
        "",
        f"{'feature':<28} {'rho':>8}  {'|rho|':>8}",
        "-" * 50,
    ]
    for c in spearman:
        lines.append(f"{c.feature_name:<28} {c.correlation:>+8.4f}  {abs(c.correlation):>8.4f}")
    lines.extend(["", "Partial Spearman | controlling local_variance:", "-" * 50])
    for row in partial_rows:
        lines.append(f"{row['feature']:<28} {row['partial_spearman']:>+10.4f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    meta = {
        "emd_id": args.emd_id, "contour": args.contour, "n_masked": n_mask,
        "mask_source": str(paths["reference"]),
        "features_npz": str(paths["features"]),
        "spearman": {c.feature_name: c.correlation for c in spearman},
        "partial_spearman": {r["feature"]: r["partial_spearman"] for r in partial_rows},
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_comparison_bar(
        [{"feature": c.feature_name, "method": c.method, "correlation": c.correlation} for c in result.correlations],
        fig_dir / "spearman_comparison_bar.png",
    )
    print(f"[rigidity_vs_mech] wrote {out_dir}", flush=True)
    print("[rigidity_vs_mech] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
