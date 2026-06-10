"""Design A: leave-one-map-out nested prediction — does V beat variance on held-out maps?

Baseline predictors (Cα, 2 Å sphere, in-mask): local variance, windowed half-map CC,
BlocRes local resolution. Full model adds constraint V. Targets: Q-score (cryo-EM-native)
then deposited B_iso (b_factor manifest rows only).

Writes ``outputs/cohort_summary/v_incremental_prediction.csv`` and a summary figure.

Example::

    source .venv/bin/activate
    python scripts/run_v_incremental_prediction.py
    python scripts/run_v_incremental_prediction.py --target q_score --figure-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.incremental_prediction import (
    TARGET_B,
    TARGET_Q,
    IncrementalPredictionSummary,
    load_map_frame,
    run_lomo_incremental_prediction,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

logger = logging.getLogger(__name__)

# RNA-only EMD-33736: no protein Cα Q-scores (matches qscore cohort panel).
QSCORE_PANEL_EXCLUDE = frozenset({"33736"})

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "v_incremental_prediction.csv"
OUTPUT_JSON = OUTPUTS_ROOT / "cohort_summary" / "v_incremental_prediction.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--target",
        choices=("both", TARGET_Q, TARGET_B),
        default="both",
        help="External target for held-out prediction",
    )
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--min-residues", type=int, default=30)
    p.add_argument("--figure-only", action="store_true", help="Rebuild figure from existing CSV")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _discover_qscore_ids() -> list[str]:
    root = OUTPUTS_ROOT
    ids: list[str] = []
    for path in sorted(root.glob("emd_*/lh_map_reliability/qscore_validation.csv")):
        emdb_id = path.parts[-3].replace("emd_", "")
        if emdb_id not in QSCORE_PANEL_EXCLUDE:
            ids.append(emdb_id)
    return ids


def _discover_bfactor_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flexibility_source", "").strip() == "b_factor":
                ids.append(str(row["emdb_id"]).strip())
    return ids


def _load_frames(
    emdb_ids: list[str],
    *,
    target: str,
    manifest: Path,
    sphere_radius_a: float,
    min_residues: int,
) -> list:
    frames = []
    for emdb_id in emdb_ids:
        frame = load_map_frame(
            emdb_id,
            target=target,  # type: ignore[arg-type]
            manifest=manifest,
            sphere_radius_a=sphere_radius_a,
            min_residues=min_residues,
        )
        if frame is None:
            logger.warning("skip EMD-%s (%s): ineligible or missing inputs", emdb_id, target)
            continue
        frames.append(frame)
        print(
            f"[v_incremental] EMD-{emdb_id} ({target}): n={frame.n_residues}",
            flush=True,
        )
    return frames


def _summary_to_rows(summary: IncrementalPredictionSummary) -> list[dict]:
    rows: list[dict] = []
    for fold in summary.fold_results:
        rows.append(
            {
                "target": summary.target,
                "emdb_id": fold.emdb_id,
                "n_residues": fold.n_residues,
                "r2_baseline": fold.r2_baseline,
                "r2_full": fold.r2_full,
                "delta_r2": fold.delta_r2,
                "delta_loglik": fold.delta_loglik,
                "delta_loglik_per_residue": fold.delta_loglik_per_residue,
                "n_maps": summary.n_maps,
                "median_delta_r2": summary.median_delta_r2,
                "mean_delta_r2": summary.mean_delta_r2,
                "n_positive_delta_r2": summary.n_positive_delta_r2,
                "sign_test_p_value": summary.sign_test_p_value,
                "median_delta_loglik": summary.median_delta_loglik,
            }
        )
    return rows


def _load_csv_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            for key in (
                "n_residues",
                "r2_baseline",
                "r2_full",
                "delta_r2",
                "delta_loglik",
                "n_maps",
                "median_delta_r2",
                "mean_delta_r2",
                "n_positive_delta_r2",
                "sign_test_p_value",
                "median_delta_loglik",
            ):
                if key in rec and rec[key] not in ("", "nan"):
                    rec[key] = float(rec[key])
            rows.append(rec)
    return rows


def _build_figure(rows: list[dict], out_dir: Path, dpi: int) -> Path:
    """Per-fold ΔR² by target (Q-score vs B-factor)."""
    by_target: dict[str, list[float]] = {}
    medians: dict[str, float] = {}
    n_maps: dict[str, int] = {}
    for row in rows:
        target = row["target"]
        delta = float(row["delta_r2"])
        by_target.setdefault(target, []).append(delta)
        medians[target] = float(row.get("median_delta_r2", np.nanmedian(by_target[target])))
        n_maps[target] = int(row.get("n_maps", len(by_target[target])))

    targets = [t for t in (TARGET_Q, TARGET_B) if t in by_target]
    if not targets:
        raise ValueError("no fold rows in CSV for figure")

    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2), squeeze=False)
    for ax, target in zip(axes.ravel(), targets):
        apply(ax)
        vals = np.array(by_target[target], dtype=np.float64)
        bp = ax.boxplot(
            [vals],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.15", "linewidth": 1.0},
        )
        bp["boxes"][0].set_facecolor(PALETTES["categorical"][0])
        bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][0].set_edgecolor("0.25")
        jitter = (np.random.default_rng(42).random(len(vals)) - 0.5) * 0.08
        ax.scatter(1 + jitter, vals, s=16, c="0.2", alpha=0.5, edgecolors="none", zorder=3)
        ax.axhline(0.0, color="0.35", linewidth=0.6, zorder=1)
        label = "Q-score" if target == TARGET_Q else "B_iso"
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=9)
        ax.set_ylabel("Held-out ΔR² (full − baseline)")
        ax.set_title(
            f"{label}: median ΔR² = {medians[target]:+.3f}\n(n = {n_maps[target]} maps, LOMO CV)",
            fontsize=9,
        )

    label_panel(axes.ravel()[0], "a")
    if len(targets) > 1:
        label_panel(axes.ravel()[1], "b")
    fig.suptitle("Incremental value of V after variance + CC + locres", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v_incremental_prediction"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _run_target(
    target: str,
    emdb_ids: list[str],
    *,
    manifest: Path,
    sphere_radius_a: float,
    min_residues: int,
) -> IncrementalPredictionSummary | None:
    frames = _load_frames(
        emdb_ids,
        target=target,
        manifest=manifest,
        sphere_radius_a=sphere_radius_a,
        min_residues=min_residues,
    )
    if len(frames) < 3:
        print(f"[v_incremental] {target}: only {len(frames)} eligible maps (need ≥3)", file=sys.stderr)
        return None
    summary = run_lomo_incremental_prediction(frames, target=target)
    print(
        f"[v_incremental] {target}: n={summary.n_maps} "
        f"median ΔR²={summary.median_delta_r2:+.4f} "
        f"({summary.n_positive_delta_r2}/{summary.n_maps} folds ΔR²>0, "
        f"sign p={summary.sign_test_p_value:.3g})",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    csv_path = args.out_dir / "v_incremental_prediction.csv"
    if args.figure_only:
        if not csv_path.is_file():
            print(f"[v_incremental] missing {csv_path}", file=sys.stderr)
            return 2
        fig_path = _build_figure(_load_csv_rows(csv_path), args.out_dir, args.dpi)
        print(f"[v_incremental] figure → {fig_path}", flush=True)
        return 0

    summaries: list[IncrementalPredictionSummary] = []
    all_rows: list[dict] = []

    targets: list[str] = []
    if args.target in ("both", TARGET_Q):
        targets.append(TARGET_Q)
    if args.target in ("both", TARGET_B):
        targets.append(TARGET_B)

    for target in targets:
        if target == TARGET_Q:
            ids = _discover_qscore_ids()
        else:
            ids = _discover_bfactor_ids(args.manifest)
        summary = _run_target(
            target,
            ids,
            manifest=args.manifest,
            sphere_radius_a=args.sphere_radius_a,
            min_residues=args.min_residues,
        )
        if summary is not None:
            summaries.append(summary)
            all_rows.extend(_summary_to_rows(summary))

    if not all_rows:
        print("[v_incremental] no results", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(
                {
                    k: (
                        f"{v:.6f}"
                        if isinstance(v, float) and np.isfinite(v)
                        else ("" if isinstance(v, float) else v)
                    )
                    for k, v in row.items()
                }
            )
    json_path = args.out_dir / "v_incremental_prediction.json"
    json_path.write_text(json.dumps(all_rows, indent=2) + "\n")

    print(f"[v_incremental] {len(all_rows)} fold rows → {csv_path}", flush=True)
    fig_path = _build_figure(all_rows, args.out_dir, args.dpi)
    from cryoem_mrc.repo_paths import sync_thesis_doc_figure

    sync_thesis_doc_figure(fig_path, "fig_3_4_v_incremental_prediction.png")
    print(f"[v_incremental] figure → {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
