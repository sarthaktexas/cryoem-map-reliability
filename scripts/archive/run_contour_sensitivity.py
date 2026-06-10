"""Contour sensitivity on EMD-49450: 0.5×, 1×, 1.5× depositor contour.

Uses precomputed ``halfmap_metrics.npz`` + feature NPZ + reference map for mask
(fast path). Writes CSV + bar figure for thesis Methods appendix.
"""

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

from style.nature import apply, label_panel, savefig as save_nature

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.half_map_repro import load_windowed_halfmap_correlation
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, sync_thesis_appendix_b_figure, analysis_dir, find_features_npz
from cryoem_mrc.structure_validation import load_cohort_manifest_row

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emd-id", default="49450")
    p.add_argument("--base-contour", type=float, default=None)
    p.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_ROOT / "sensitivity" / "contour",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    emd = str(args.emd_id).strip()
    row = load_cohort_manifest_row(REPO_ROOT / COHORT_MANIFEST, emd)
    base_contour = float(args.base_contour if args.base_contour is not None else row["contour"])
    ref = (REPO_ROOT / row["reference_mrc"]).resolve()

    metrics_npz = analysis_dir(emd) / "halfmap_metrics.npz"
    if not metrics_npz.is_file():
        print(f"[contour_sens] missing {metrics_npz}", file=sys.stderr)
        return 2
    features_npz = find_features_npz(ref.parent, emd, base_contour)
    if features_npz is None or not features_npz.is_file():
        print(f"[contour_sens] missing features NPZ for EMD-{emd}", file=sys.stderr)
        return 2

    ref_mg = load_map_grid(ref, normalize=None)
    cc = load_windowed_halfmap_correlation(np.load(metrics_npz))
    feat = np.load(features_npz)
    var_w = np.asarray(feat["local_variance"], dtype=np.float64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str]] = []
    for mult in args.multipliers:
        contour = float(base_contour) * float(mult)
        mask = build_contour_mask(ref_mg.data, contour)
        n = int(mask.sum())
        x = var_w[mask].ravel()
        y = cc[mask].ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size >= 3 and x.std() > 0 and y.std() > 0:
            rho = float(stats.spearmanr(x, y).statistic)
        else:
            rho = float("nan")
        records.append({
            "emdb_id": emd,
            "contour_multiplier": mult,
            "contour": contour,
            "n_mask_voxels": n,
            "spearman_var_vs_cc": rho,
        })
        print(f"[contour_sens] ×{mult:g} contour={contour:.4f} n={n:,} ρ={rho:.4f}")

    csv_path = args.out_dir / f"emd_{emd}_contour_sensitivity.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    labels = [f"×{r['contour_multiplier']:g}" for r in records]
    rhos = [r["spearman_var_vs_cc"] for r in records]
    ns = [r["n_mask_voxels"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax = axes[0]
    apply(ax)
    ax.bar(labels, rhos, color="#4C72B0")
    ax.set_ylabel("Spearman ρ(var, CC)")
    ax.set_xlabel("Contour multiplier")
    ax.set_title(f"EMD-{emd}: feature–CC stability")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(rhos):
        if np.isfinite(v):
            ax.text(i, float(v) + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    label_panel(ax, "a")

    ax2 = axes[1]
    apply(ax2)
    ax2.bar(labels, [n / 1e3 for n in ns], color="#55A868")
    ax2.set_ylabel("In-mask voxels (×10³)")
    ax2.set_xlabel("Contour multiplier")
    ax2.set_title("Mask volume")
    label_panel(ax2, "b")

    fig.suptitle(f"Contour sensitivity — EMD-{emd} (base {base_contour})", fontsize=10)
    fig.tight_layout()
    fig_path = fig_dir / f"emd_{emd}_contour_sensitivity.png"
    save_nature(fig, fig_path)
    plt.close(fig)
    thesis_path = sync_thesis_appendix_b_figure(fig_path, "fig_b3_contour_sensitivity.png")
    print(f"[contour_sens] wrote {csv_path} and {fig_path}")
    print(f"[contour_sens] synced → {thesis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
