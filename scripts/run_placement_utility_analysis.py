"""Operational placement-utility analyses for Structure-paper validation.

Tier 1: low-Q enrichment, head-to-head predictors, calibration, mis-ranking.
Tier 2: per-map rank recovery ρ(Q, proxy).

Writes CSVs and figures under ``outputs/cohort_summary/`` and
``PLACEMENT_UTILITY.md``.

Example::

    source .venv/bin/activate
    python scripts/run_placement_utility_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import PALETTES, apply, savefig as save_nature

from cryoem_mrc.placement_utility import (
    PREDICTOR_LABELS,
    run_placement_utility_analysis,
    write_placement_utility_csvs,
    write_placement_utility_markdown,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, sync_thesis_doc_figure

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--q-threshold", type=float, default=0.5)
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def _plot_low_q_enrichment(summary, out_dir: Path, dpi: int) -> Path:
    rows = summary.enrichment_rows
    if not rows:
        raise ValueError("no enrichment rows")

    labels = [
        "Omit zone",
        "Reliability\n< 0.33",
        "CC < 0.5",
        "BlocRes worse\nthan median",
        "Variance above\nmedian",
    ]
    keys = [
        "frac_low_q_in_omit_zone",
        "frac_low_q_reliability_below",
        "frac_low_q_cc_below",
        "frac_low_q_locres_worse_than_median",
        "frac_low_q_variance_above_median",
    ]
    medians = []
    for k in keys:
        vals = [getattr(r, k) for r in rows if np.isfinite(getattr(r, k))]
        medians.append(float(np.median(vals)) if vals else float("nan"))
    baseline = float(np.median([r.omit_zone_baseline for r in rows]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    apply(ax)
    x = np.arange(len(labels))
    colors = PALETTES["categorical"][: len(labels)]
    ax.bar(x, medians, color=colors, edgecolor="0.15", linewidth=0.6)
    ax.axhline(baseline, color="0.45", linestyle="--", linewidth=1.0, label="Omit-zone baseline (all Cα)")
    ax.axhline(1 / 3, color="0.65", linestyle=":", linewidth=0.8, label="Random tercile (0.33)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"Fraction of low-Q residues flagged\n(Q < {summary.q_threshold:.1f})")
    ax.set_title("Cohort median: pre-model readouts vs low Q-score")
    ax.set_ylim(0, min(1.05, max(medians + [baseline, 1 / 3]) + 0.08))
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    out = out_dir / "placement_low_q_enrichment"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _plot_head_to_head(summary, out_dir: Path, dpi: int) -> Path:
    rows = summary.predictor_rows
    if not rows:
        raise ValueError("no predictor rows")

    labels = [PREDICTOR_LABELS[r.predictor] for r in rows]
    frac = [r.pooled_frac_low_q_flagged for r in rows]
    ba = [r.pooled_balanced_accuracy for r in rows]
    auc = [r.median_map_auc for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax in axes:
        apply(ax)
    x = np.arange(len(labels))
    colors = PALETTES["categorical"][: len(labels)]

    axes[0].barh(x, frac, color=colors, edgecolor="0.15", linewidth=0.5)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("Pooled frac. low-Q flagged")
    axes[0].set_title("Enrichment")

    axes[1].barh(x, ba, color=colors, edgecolor="0.15", linewidth=0.5)
    axes[1].set_xlabel("Pooled balanced accuracy")
    axes[1].set_title("Classification (Q threshold)")
    axes[1].set_yticks([])

    axes[2].barh(x, auc, color=colors, edgecolor="0.15", linewidth=0.5)
    axes[2].set_xlabel("Median per-map AUC")
    axes[2].set_title("Rank recovery")
    axes[2].set_yticks([])

    fig.suptitle("Head-to-head pre-model readouts (Q-score ground truth)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = out_dir / "placement_predictor_head_to_head"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _plot_rank_recovery(summary, out_dir: Path, dpi: int) -> Path:
    rows = summary.rank_recovery_rows
    if not rows:
        raise ValueError("no rank recovery rows")

    proxies = [
        ("reliability", [r.spearman_q_vs_reliability for r in rows]),
        ("windowed CC", [r.spearman_q_vs_cc for r in rows]),
        ("BlocRes", [r.spearman_q_vs_locres for r in rows]),
        ("variance", [r.spearman_q_vs_variance for r in rows]),
        ("constraint V", [r.spearman_q_vs_v for r in rows]),
    ]
    labels = [p[0] for p in proxies]
    meds = []
    for _, vals in proxies:
        v = [x for x in vals if np.isfinite(x)]
        meds.append(float(np.median(v)) if v else float("nan"))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    apply(ax)
    x = np.arange(len(labels))
    colors = PALETTES["categorical"][: len(labels)]
    ax.bar(x, meds, color=colors, edgecolor="0.15", linewidth=0.6)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median per-map Spearman ρ(Q, proxy)")
    ax.set_title("Rank recovery: which pre-model readout tracks Q?")
    out = out_dir / "placement_rank_recovery"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _plot_calibration(summary, out_dir: Path, dpi: int) -> Path:
    bins = summary.calibration_bins
    if not bins:
        raise ValueError("no calibration bins")

    centers = [(b.reliability_bin_lo + b.reliability_bin_hi) / 2 for b in bins]
    mean_q = [b.mean_q for b in bins]
    counts = [b.n_residues for b in bins]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    apply(ax)
    ax.plot(centers, mean_q, "o-", color=PALETTES["categorical"][0], linewidth=1.4, markersize=5)
    ax.set_xlabel("Reliability score (decile center)")
    ax.set_ylabel("Mean Q-score")
    ax.set_title("Calibration: higher reliability -> higher Q (cohort-pooled)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(mean_q) * 1.05 + 0.02)
    for c, mq in zip(centers, mean_q):
        ax.annotate(f"n={counts[centers.index(c)]:,}", (c, mq), fontsize=6, ha="center", va="bottom")
    out = out_dir / "placement_q_calibration"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _plot_misranking(summary, out_dir: Path, dpi: int) -> Path:
    rows = summary.misranking_rows
    if not rows:
        raise ValueError("no misranking rows")

    def med(attr: str) -> float:
        vals = [getattr(r, attr) for r in rows if np.isfinite(getattr(r, attr))]
        return float(np.median(vals)) if vals else float("nan")

    labels = [
        "Sharp BlocRes\n(bottom-Q tercile)",
        "Omit zone\n(bottom-Q tercile)",
        "CC ≥ 0.7\n(bottom-Q tercile)",
    ]
    vals = [
        med("frac_sharp_locres_low_q_tercile"),
        med("frac_omit_zone_low_q_tercile"),
        med("frac_cc_above_0_7_low_q_tercile"),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    apply(ax)
    x = np.arange(len(labels))
    colors = ["#6ea8ff", "#7ee2c7", "#ffd166"]
    ax.bar(x, vals, color=colors, edgecolor="0.15", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction among bottom-Q tercile")
    ax.set_title("Mis-ranking: resolvability looks fine, Q does not")
    ax.set_ylim(0, min(1.05, max(vals) + 0.1))
    out = out_dir / "placement_misranking"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_placement_utility_analysis(
        manifest=args.manifest,
        q_threshold=args.q_threshold,
        sphere_radius_a=args.sphere_radius_a,
    )

    if not summary.enrichment_rows:
        print("[placement_utility] no maps with Q-score validation", file=sys.stderr)
        return 2

    paths = write_placement_utility_csvs(summary, args.out_dir)
    md_path = write_placement_utility_markdown(summary, args.out_dir / "PLACEMENT_UTILITY.md")

    meta = {
        "q_threshold": args.q_threshold,
        "n_maps": len(summary.enrichment_rows),
        "resolution_bins": summary.resolution_bins,
        "csv_paths": {k: str(v) for k, v in paths.items()},
    }
    json_path = args.out_dir / "placement_utility.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[placement_utility] {len(summary.enrichment_rows)} maps analyzed", flush=True)
    for k, p in paths.items():
        print(f"  {k}: {p}", flush=True)
    print(f"  markdown: {md_path}", flush=True)

    if not args.no_figures:
        fig_paths = [
            _plot_low_q_enrichment(summary, args.out_dir, args.dpi),
            _plot_head_to_head(summary, args.out_dir, args.dpi),
            _plot_rank_recovery(summary, args.out_dir, args.dpi),
            _plot_calibration(summary, args.out_dir, args.dpi),
            _plot_misranking(summary, args.out_dir, args.dpi),
        ]
        for fp in fig_paths:
            print(f"  figure: {fp}", flush=True)
        sync_names = [
            ("placement_low_q_enrichment.png", "fig_placement_low_q_enrichment.png"),
            ("placement_predictor_head_to_head.png", "fig_placement_predictor_head_to_head.png"),
            ("placement_rank_recovery.png", "fig_placement_rank_recovery.png"),
            ("placement_q_calibration.png", "fig_placement_q_calibration.png"),
            ("placement_misranking.png", "fig_placement_misranking.png"),
        ]
        for src_name, dest_name in sync_names:
            sync_thesis_doc_figure(args.out_dir / src_name, dest_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
