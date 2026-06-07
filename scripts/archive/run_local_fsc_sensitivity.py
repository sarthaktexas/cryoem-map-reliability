"""Sensitivity panel: patch_size x fsc_threshold local FSC runs + CC agreement figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.half_map_repro import half_map_local_metrics
from cryoem_mrc.local_fsc import compute_local_fsc_resolution, save_local_fsc_resolution_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.repo_paths import sensitivity_local_fsc_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--half1", required=True, type=Path)
    p.add_argument("--half2", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--contour", type=float, default=0.116)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--window", type=int, default=5, help="Half-map CC window")
    p.add_argument("--out-dir", type=Path, default=sensitivity_local_fsc_dir())
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_full_and_half_maps(
        args.reference, args.half1, args.half2,
        reference="full", dtype=np.float32, resample_if_needed=True,
    )
    ref_mg = load_map_grid(args.reference, normalize=None)
    vs = ref_mg.voxel_size_zyx
    voxel_size_a = float(vs[0])
    mask = build_contour_mask(bundle.full.data, args.contour)
    print(f"[sensitivity] mask: {int(mask.sum()):,}/{mask.size:,} voxels")

    print("[sensitivity] computing half-map CC for comparison")
    metrics = half_map_local_metrics(
        bundle.half1.data, bundle.half2.data, window=args.window,
    )
    cc = metrics["local_cross_correlation"]

    patch_sizes = (13, 17, 25)
    thresholds = (0.143, 0.5)
    records: list[dict[str, float | int | str]] = []
    maps: dict[tuple[int, float], np.ndarray] = {}

    for p in patch_sizes:
        for t in thresholds:
            tag = f"local_fsc_t{t:g}_P{p}_s{args.stride}".replace(".", "")
            out_mrc = out_dir / f"{tag}.mrc"
            print(f"[sensitivity] P={p} t={t} -> {out_mrc.name}")
            res = compute_local_fsc_resolution(
                bundle.half1.data,
                bundle.half2.data,
                voxel_size_a,
                patch_size=p,
                stride=args.stride,
                fsc_threshold=t,
                mask=mask,
                n_jobs=min(args.n_jobs, 4),
                require_mask=True,
            )
            save_local_fsc_resolution_mrc(
                res, args.reference, out_mrc,
                fsc_threshold=t, patch_size=p, stride=args.stride,
            )
            maps[(p, t)] = res
            x = cc[mask].ravel()
            y = res[mask].ravel()
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if x.size >= 3 and x.std() > 0 and y.std() > 0:
                rho = float(stats.spearmanr(x, y).statistic)
                pval = float(stats.spearmanr(x, y).pvalue)
            else:
                rho, pval = float("nan"), float("nan")
            records.append({
                "patch_size": p,
                "fsc_threshold": t,
                "spearman_vs_local_cc": rho,
                "p_value": pval,
                "n_samples": int(x.size),
                "mrc_path": str(out_mrc),
            })

    csv_path = out_dir / "spearman_vs_local_cc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[sensitivity] wrote {csv_path}")

    # 2x3 midplane panel (rows: threshold, cols: patch_size)
    nz, ny, nx = cc.shape
    z_mid = nz // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=True)
    vmin = 2.0 * voxel_size_a
    vmax = max(patch_sizes) * voxel_size_a
    for i, t in enumerate(thresholds):
        for j, p in enumerate(patch_sizes):
            ax = axes[i, j]
            sl = maps[(p, t)][z_mid, :, :]
            im = ax.imshow(sl, cmap="viridis_r", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"P={p}, t={t}")
            ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="resolution (Å)")
    fig.suptitle(f"Local FSC midplane Z={z_mid} (contour {args.contour})")
    fig.tight_layout()
    panel_path = fig_dir / "midplane_panel_2x3.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[sensitivity] wrote {panel_path}")

    labels = [f"P={r['patch_size']} t={r['fsc_threshold']}" for r in records]
    rhos = [r["spearman_vs_local_cc"] for r in records]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["steelblue" if r["fsc_threshold"] == 0.143 else "coral" for r in records]
    ax2.bar(range(len(labels)), rhos, color=colors)
    ax2.axhline(0.0, color="k", linewidth=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Spearman(local_CC, local_FSC Å)")
    ax2.set_title("Agreement with windowed half-map CC proxy")
    fig2.tight_layout()
    bar_path = fig_dir / "spearman_vs_cc_bar.png"
    fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[sensitivity] wrote {bar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
