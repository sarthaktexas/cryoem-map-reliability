"""Resolution sensitivity of cohort ρ(Q-score, V): bin sweeps and cutoff analysis.

Reads ``outputs/cohort_summary/qscore_correlations.csv`` and ``cohort/manifest.csv``.
Writes CSV tables and a three-panel figure under ``outputs/cohort_summary/``.

Example::

    source .venv/bin/activate
    python scripts/run_qscore_resolution_sensitivity.py
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
from scipy import stats as scipy_stats

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.cohort_labels import cohort_figure_label, load_display_name_map
from cryoem_mrc.cohort_resolution import (
    COHORT_RESOLUTION_BINS,
    cutoff_median_table,
    summarize_resolution_bins,
    sweep_resolution_bins,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, sync_thesis_doc_figure

# RNA-only EMD-33736: zero protein Cα; see docs/THESIS_AND_PUBLICATION.md §3.4.
QSCORE_PANEL_EXCLUDE = frozenset({"33736"})

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--csv", type=Path, default=OUT_DIR / "qscore_correlations.csv")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _load_pairs(manifest: Path, csv_path: Path) -> tuple[list[dict], list[tuple[float, float]]]:
    res_by_id: dict[str, float] = {}
    name_by_id = load_display_name_map(manifest)
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            try:
                res_by_id[eid] = float(row["global_resolution_a"])
            except (KeyError, ValueError):
                pass

    recs: list[dict] = []
    pairs: list[tuple[float, float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            if eid in QSCORE_PANEL_EXCLUDE:
                continue
            raw = row.get("spearman_q_vs_V", "")
            if raw in ("", "nan"):
                continue
            rho = float(raw)
            if not np.isfinite(rho):
                continue
            res = res_by_id.get(eid, float("nan"))
            name = name_by_id.get(eid, eid)
            recs.append(
                {
                    "emdb_id": eid,
                    "display_name": name,
                    "global_resolution_a": res,
                    "spearman_q_vs_V": rho,
                    "n_in_mask": int(row.get("n_in_mask", 0)),
                }
            )
            if np.isfinite(res):
                pairs.append((res, rho))
    return recs, pairs


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _build_figure(
    recs: list[dict],
    standard_bins: list[dict],
    fine_bins: list[dict],
    cutoffs: list[dict],
    out_dir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    # Panel a: standard cohort bins
    ax = axes[0]
    apply(ax)
    if standard_bins:
        labels = [str(r["bin_label"]) for r in standard_bins]
        meds = [float(r["median_rho"]) for r in standard_bins]
        ns = [int(r["n"]) for r in standard_bins]
        x = np.arange(len(labels))
        ax.bar(x, meds, color=PALETTES["categorical"][0], edgecolor="0.2", linewidth=0.5)
        ax.axhline(0.0, color="0.35", linewidth=0.6)
        for i, (m, n) in enumerate(zip(meds, ns)):
            ax.text(i, m + 0.03 * np.sign(m if m else 1), f"n={n}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("Median Spearman ρ(Q, V)")
        ax.set_title("Standard resolution bins")
    label_panel(ax, "a")

    # Panel b: 0.5 Å sweep
    ax = axes[1]
    apply(ax)
    if fine_bins:
        centers = [(float(r["lo_a"]) + float(r["hi_a"])) / 2 for r in fine_bins]
        meds = [float(r["median_rho"]) for r in fine_bins]
        ax.plot(centers, meds, "o-", color=PALETTES["categorical"][1], linewidth=1.2, markersize=5)
        ax.axhline(0.0, color="0.35", linewidth=0.6)
        ax.axvline(4.0, color="0.55", linewidth=0.8, linestyle="--", label="4 Å cutoff")
        ax.set_xlabel("Global resolution (Å, bin center)")
        ax.set_ylabel("Median ρ(Q, V)")
        ax.set_title("0.5 Å bin sweep (2–6 Å)")
        ax.legend(frameon=False, fontsize=6)
    label_panel(ax, "b")

    # Panel c: cutoff sensitivity
    ax = axes[2]
    apply(ax)
    if cutoffs:
        xs = [float(r["cutoff_a"]) for r in cutoffs]
        med_le = [float(r["median_rho_le_cutoff"]) for r in cutoffs]
        med_gt = [float(r["median_rho_gt_cutoff"]) for r in cutoffs]
        ax.plot(xs, med_le, "o-", color=PALETTES["categorical"][0], linewidth=1.2, label="res ≤ cutoff")
        ax.plot(xs, med_gt, "s--", color=PALETTES["categorical"][3], linewidth=1.0, label="res > cutoff")
        ax.axhline(0.0, color="0.35", linewidth=0.6)
        ax.axvline(4.0, color="0.55", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.set_xlabel("Resolution ceiling (Å)")
        ax.set_ylabel("Median ρ(Q, V)")
        ax.set_title("Where does signal drop?")
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    label_panel(ax, "c")

    overall = float(np.median([r["spearman_q_vs_V"] for r in recs]))
    res = np.array([r["global_resolution_a"] for r in recs if np.isfinite(r["global_resolution_a"])])
    rhos = np.array(
        [
            r["spearman_q_vs_V"]
            for r in recs
            if np.isfinite(r["global_resolution_a"]) and np.isfinite(r["spearman_q_vs_V"])
        ]
    )
    rho_rr = scipy_stats.spearmanr(res, rhos).statistic if res.size >= 3 else float("nan")
    fig.suptitle(
        f"ρ(Q, V) resolution sensitivity — n={len(recs)}, cohort median={overall:+.2f}, "
        f"ρ vs res={rho_rr:+.2f}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "qscore_resolution_sensitivity"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _build_per_map_figure(recs: list[dict], manifest: Path, out_dir: Path, dpi: int) -> Path:
    """Per-structure ranking with protein names (replaces EMD IDs on y-axis)."""
    names = load_display_name_map(manifest)
    sorted_recs = sorted(recs, key=lambda r: r["spearman_q_vs_V"])
    rhos = np.array([r["spearman_q_vs_V"] for r in sorted_recs])
    res = np.array([r["global_resolution_a"] for r in sorted_recs])
    labels = [cohort_figure_label(r["emdb_id"], names=names) for r in sorted_recs]
    median_rho = float(np.median(rhos))

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(11.0, max(6.5, 0.22 * len(sorted_recs) + 1.5)))

    apply(ax_bar)
    res_finite = res[np.isfinite(res)]
    vmin = float(res_finite.min()) if res_finite.size else 0.0
    vmax = float(res_finite.max()) if res_finite.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(norm(v)) if np.isfinite(v) else "0.6" for v in res]
    ypos = np.arange(len(sorted_recs))
    ax_bar.barh(ypos, rhos, color=colors, edgecolor="0.2", linewidth=0.4)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=5)
    ax_bar.axvline(0.0, color="0.3", linewidth=0.6)
    ax_bar.axvline(median_rho, color=PALETTES["categorical"][1], linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Spearman ρ(Q-score, V), in-mask Cα")
    ax_bar.set_title(f"Per-structure Q-score vs V (median ρ={median_rho:+.2f})")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, fraction=0.046, pad=0.02)
    cbar.set_label("Global resolution (Å)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    label_panel(ax_bar, "a")

    apply(ax_sc)
    m = np.isfinite(res)
    ax_sc.scatter(res[m], rhos[m], s=24, c=PALETTES["categorical"][0], edgecolors="0.2", linewidths=0.4)
    if m.sum() >= 3:
        rho_rr = scipy_stats.spearmanr(res[m], rhos[m]).statistic
        coef = np.polyfit(res[m], rhos[m], 1)
        xline = np.linspace(res[m].min(), res[m].max(), 50)
        ax_sc.plot(xline, np.polyval(coef, xline), color=PALETTES["categorical"][1], linewidth=0.9)
        ax_sc.set_title(f"ρ(Q,V) vs resolution (Spearman={rho_rr:+.2f})")
    else:
        ax_sc.set_title("ρ(Q,V) vs resolution")
    ax_sc.axhline(0.0, color="0.3", linewidth=0.6)
    ax_sc.axvline(4.0, color="0.55", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_sc.set_xlabel("Global resolution (Å)")
    ax_sc.set_ylabel("Spearman ρ(Q-score, V)")
    label_panel(ax_sc, "b")

    fig.suptitle("Q-score vs constraint V — cohort summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / "qscore_vs_V_cohort"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    sync_thesis_doc_figure(out.with_suffix(".png"), "fig_3_4_qscore_vs_V_cohort.png")
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.csv.is_file():
        print(f"[resolution_sensitivity] missing {args.csv}", file=sys.stderr)
        return 2

    recs, pairs = _load_pairs(args.manifest, args.csv)
    if len(recs) < 3:
        print("[resolution_sensitivity] fewer than three finite ρ rows", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    standard = summarize_resolution_bins(pairs, bins=COHORT_RESOLUTION_BINS)
    fine = sweep_resolution_bins(pairs, width=0.5, lo=2.0, hi=6.0)
    cutoffs = cutoff_median_table(pairs)

    _write_csv(
        args.out_dir / "qscore_resolution_standard_bins.csv",
        standard,
        ["bin_label", "bin_key", "lo_a", "hi_a", "n", "median_rho", "mean_rho", "min_rho", "max_rho"],
    )
    _write_csv(
        args.out_dir / "qscore_resolution_fine_bins.csv",
        fine,
        ["bin_label", "lo_a", "hi_a", "n", "median_rho", "mean_rho"],
    )
    _write_csv(
        args.out_dir / "qscore_resolution_cutoffs.csv",
        cutoffs,
        ["cutoff_a", "n_le_cutoff", "median_rho_le_cutoff", "n_gt_cutoff", "median_rho_gt_cutoff"],
    )
    per_map_fields = ["emdb_id", "display_name", "global_resolution_a", "spearman_q_vs_V", "n_in_mask"]
    _write_csv(args.out_dir / "qscore_resolution_per_map.csv", recs, per_map_fields)

    fig1 = _build_figure(recs, standard, fine, cutoffs, args.out_dir, args.dpi)
    fig2 = _build_per_map_figure(recs, args.manifest, args.out_dir, args.dpi)
    sync_thesis_doc_figure(fig1, "fig_3_4_qscore_resolution_sensitivity.png")

    print(f"[resolution_sensitivity] standard bins → {args.out_dir / 'qscore_resolution_standard_bins.csv'}", flush=True)
    print(f"[resolution_sensitivity] figure → {fig1}", flush=True)
    print(f"[resolution_sensitivity] cohort ranking (protein names) → {fig2}", flush=True)

    for row in standard:
        print(
            f"  {row['bin_label']:10s}  n={int(row['n']):2d}  median={float(row['median_rho']):+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
