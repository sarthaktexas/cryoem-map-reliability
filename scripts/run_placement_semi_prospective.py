"""Semi-prospective placement validation and main-paper ROC figures.

Leave-one-map-out (LOMO) held-out metrics and pooled low-Q ROC curves for
Structure-paper / thesis figures.

Example::

    source .venv/bin/activate
    python scripts/run_placement_semi_prospective.py
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
    MAIN_ROC_PREDICTORS,
    PREDICTOR_LABELS,
    load_per_map_frames_for_lomo,
    pooled_roc_curve,
    run_lomo_placement_validation,
    write_lomo_placement_csvs,
    write_lomo_placement_markdown,
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


def _plot_lomo_violin(lomo_summary, out_dir: Path, dpi: int) -> Path:
    folds = lomo_summary.fold_rows
    predictors = [p for p in MAIN_ROC_PREDICTORS if p in PREDICTOR_LABELS]
    labels = [PREDICTOR_LABELS[p] for p in predictors]

    auc_data: list[list[float]] = []
    rho_data: list[list[float]] = []
    for pid in predictors:
        auc_vals = [
            r.auc for r in folds if r.predictor == pid and np.isfinite(r.auc)
        ]
        rho_vals = [
            r.spearman_q_vs_score
            for r in folds
            if r.predictor == pid and np.isfinite(r.spearman_q_vs_score)
        ]
        auc_data.append(auc_vals)
        rho_data.append(rho_vals)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax in axes:
        apply(ax)

    colors = PALETTES["categorical"][: len(predictors)]
    parts0 = axes[0].violinplot(auc_data, positions=np.arange(len(predictors)), showmeans=True, showextrema=False)
    for i, body in enumerate(parts0["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.75)
    axes[0].set_xticks(np.arange(len(predictors)))
    axes[0].set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    axes[0].set_ylabel("Held-out AUC (per map)")
    axes[0].set_title("LOMO rank classification")
    axes[0].set_ylim(0, 1.02)

    parts1 = axes[1].violinplot(rho_data, positions=np.arange(len(predictors)), showmeans=True, showextrema=False)
    for i, body in enumerate(parts1["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.75)
    axes[1].set_xticks(np.arange(len(predictors)))
    axes[1].set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    axes[1].set_ylabel("Held-out Spearman rho(Q, score)")
    axes[1].set_title("LOMO rank recovery")
    axes[1].axhline(0, color="0.5", linewidth=0.8)

    fig.suptitle(
        f"Semi-prospective validation (leave-one-map-out, Q < {lomo_summary.q_threshold:.1f})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "placement_lomo_held_out"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _plot_low_q_roc(per_map_frames, q_threshold: float, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    apply(ax)
    colors = PALETTES["categorical"]

    for i, pid in enumerate(MAIN_ROC_PREDICTORS):
        curve = pooled_roc_curve(per_map_frames, pid, q_threshold=q_threshold)
        if not curve.fpr:
            continue
        label = f"{PREDICTOR_LABELS[pid]} (AUC={curve.auc:.2f})"
        ax.plot(curve.fpr, curve.tpr, color=colors[i % len(colors)], linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (Q < threshold)")
    ax.set_title(f"Pooled low-Q ROC (cohort, Q < {q_threshold:.1f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    out = out_dir / "placement_low_q_roc"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frames_full = load_per_map_frames_for_lomo(
        manifest=args.manifest,
        sphere_radius_a=args.sphere_radius_a,
    )
    if len(frames_full) < 3:
        print("[placement_lomo] need >= 3 maps with Q-scores", file=sys.stderr)
        return 2

    per_map = [(eid, df) for eid, df, _ in frames_full]
    lomo = run_lomo_placement_validation(frames_full, q_threshold=args.q_threshold)

    paths = write_lomo_placement_csvs(lomo, args.out_dir)
    md_path = write_lomo_placement_markdown(lomo, args.out_dir / "PLACEMENT_LOMO.md")

    meta = {
        "q_threshold": args.q_threshold,
        "n_maps": len(frames_full),
        "predictor_medians": lomo.predictor_medians,
        "csv_paths": {k: str(v) for k, v in paths.items()},
    }
    json_path = args.out_dir / "placement_lomo.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[placement_lomo] {len(frames_full)} maps, {len(lomo.fold_rows)} fold rows", flush=True)
    for k, p in paths.items():
        print(f"  {k}: {p}", flush=True)
    print(f"  markdown: {md_path}", flush=True)

    rel_meds = lomo.predictor_medians.get("reliability_below_0_33", {})
    print(
        f"  median held-out AUC (reliability): {rel_meds.get('median_auc', float('nan')):.3f}",
        flush=True,
    )

    if not args.no_figures:
        fig_paths = [
            _plot_lomo_violin(lomo, args.out_dir, args.dpi),
            _plot_low_q_roc(per_map, args.q_threshold, args.out_dir, args.dpi),
        ]
        for fp in fig_paths:
            print(f"  figure: {fp}", flush=True)
        sync_thesis_doc_figure(
            args.out_dir / "placement_lomo_held_out.png",
            "fig_placement_lomo_held_out.png",
        )
        sync_thesis_doc_figure(
            args.out_dir / "placement_low_q_roc.png",
            "fig_placement_low_q_roc.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
