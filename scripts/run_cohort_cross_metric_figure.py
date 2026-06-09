"""Cohort figure from exported ``cross_metric_correlations.csv`` tables.

Reads ``outputs/emd_<ID>/metric_comparison/cross_metric_correlations.csv`` (no map
reload) and writes:

- ``cohort_cross_metric_median.png`` — median Spearman ρ across maps (metric×metric)
- ``cohort_cross_metric_locres_pairs.png`` — per-map ρ vs BlocRes for key pairs

Example::

    source .venv/bin/activate
    python scripts/run_cohort_cross_metric_figure.py
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
import pandas as pd

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.metric_comparison import METRIC_COLUMNS
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, emd_output_dir

METRIC_LABELS = {
    "v_metric": "V",
    "reliability_score": "Reliability",
    "reliability_H_repro": "H_repro",
    "b_factor": "B_iso",
    "local_cross_correlation": "Half-map CC",
    "local_variance": "Local variance",
    "local_resolution": "BlocRes locres",
}

LOCres_PAIR_KEYS = (
    ("v_metric", "local_resolution"),
    ("b_factor", "local_resolution"),
    ("local_cross_correlation", "local_resolution"),
    ("local_variance", "local_resolution"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _eligible_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            corr = emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv"
            if corr.is_file():
                ids.append(eid)
    return ids


def _read_corr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _collect_median_matrix(ids: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = list(METRIC_COLUMNS)
    stacks: dict[tuple[str, str], list[float]] = {}
    for eid in ids:
        corr = _read_corr(emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv")
        for i, ci in enumerate(cols):
            if ci not in corr.index:
                continue
            for j, cj in enumerate(cols):
                if j < i or cj not in corr.columns:
                    continue
                val = float(corr.loc[ci, cj])
                if np.isfinite(val):
                    stacks.setdefault((ci, cj), []).append(val)

    mat = np.full((len(cols), len(cols)), np.nan, dtype=np.float64)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j < i:
                mat[i, j] = mat[j, i]
                continue
            vals = stacks.get((ci, cj), [])
            mat[i, j] = float(np.median(vals)) if vals else float("nan")
    return mat, cols


def _collect_locres_pairs(ids: list[str]) -> list[dict]:
    recs: list[dict] = []
    for eid in ids:
        corr = _read_corr(emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv")
        rec = {"emdb_id": eid}
        for a, b in LOCres_PAIR_KEYS:
            if a in corr.index and b in corr.columns:
                rec[f"{a}|{b}"] = float(corr.loc[a, b])
            else:
                rec[f"{a}|{b}"] = float("nan")
        recs.append(rec)
    return recs


def _build_median_figure(mat: np.ndarray, cols: list[str], out_dir: Path, dpi: int) -> Path:
    labels = [METRIC_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    apply(ax)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color="0.15")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median Spearman ρ (in-mask Cα)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Cross-metric coupling — cohort median")
    fig.tight_layout()
    out = out_dir / "cohort_cross_metric_median"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _build_locres_pairs_figure(recs: list[dict], out_dir: Path, dpi: int) -> Path:
    pair_labels = [
        ("v_metric|local_resolution", "V vs locres"),
        ("b_factor|local_resolution", "B vs locres"),
        ("local_cross_correlation|local_resolution", "CC vs locres"),
        ("local_variance|local_resolution", "Var vs locres"),
    ]
    usable = [
        r
        for r in recs
        if np.isfinite(float(r.get("v_metric|local_resolution", float("nan"))))
    ]
    usable.sort(key=lambda d: float(d["v_metric|local_resolution"]))
    if len(usable) < 3:
        raise ValueError("Need at least three maps with finite ρ(V, locres)")

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.22 * len(usable) + 1.5)))
    apply(ax)
    ypos = np.arange(len(usable))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(pair_labels))
    colors = PALETTES["categorical"][: len(pair_labels)]

    for off, (key, label), color in zip(offsets, pair_labels, colors):
        vals = np.array([float(r.get(key, float("nan"))) for r in usable], dtype=np.float64)
        ax.barh(ypos + off, vals, height=width, color=color, label=label, edgecolor="0.2", linewidth=0.3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"EMD-{r['emdb_id']}" for r in usable], fontsize=6)
    ax.axvline(0.0, color="0.35", linewidth=0.6)
    ax.set_xlabel("Spearman ρ vs BlocRes local resolution")
    ax.set_title(f"Per-map locres coupling (n = {len(usable)})")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    out = out_dir / "cohort_cross_metric_locres_pairs"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = _eligible_ids(args.manifest)
    if len(ids) < 3:
        print("[cross_metric_fig] fewer than three maps with cross_metric_correlations.csv", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mat, cols = _collect_median_matrix(ids)
    fig1 = _build_median_figure(mat, cols, args.out_dir, args.dpi)
    print(f"[cross_metric_fig] median heatmap → {fig1}", flush=True)

    recs = _collect_locres_pairs(ids)
    fig2 = _build_locres_pairs_figure(recs, args.out_dir, args.dpi)
    print(f"[cross_metric_fig] locres pairs → {fig2}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
