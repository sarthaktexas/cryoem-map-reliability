"""Cohort audit: deposited Cα in low-CC regions vs reliability placement.

Reads ``residue_validation.csv`` per map (or runs validation when missing) and
writes ``outputs/cohort_summary/model_placement_audit.csv`` plus scatter plots
comparing tercile-based omit fractions and absolute reliability/CC cutoffs.

Example::

    source .venv/bin/activate
    python scripts/run_cohort_model_placement_audit.py --all
    python scripts/run_cohort_model_placement_audit.py --emd-id 11638 --cc-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from style.nature import PALETTES, apply, savefig as save_nature

from cryoem_mrc.placement_supplement import plot_placement_supplement
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, sync_thesis_doc_figure
from cryoem_mrc.structure_validation import (
    compute_model_placement_audit_stats,
    default_reliability_out_dir,
    load_cohort_manifest_row,
    read_residue_validation_csv,
    run_emdb_bfactor_validation,
    write_model_placement_audit_csv,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 11638)")
    p.add_argument("--all", action="store_true", help="All manifest rows with a local deposited model")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--cc-threshold",
        type=float,
        default=0.5,
        help="Absolute windowed half-map correlation cutoff for questionable placement (default 0.5)",
    )
    p.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.33,
        help="Absolute reliability_score cutoff at Cα (default 0.33 = bottom tercile)",
    )
    p.add_argument(
        "--highlight-disagreement-min",
        type=float,
        default=0.30,
        help="Highlight maps when |x − y| ≥ this (default 0.30); decoupled maps always highlighted",
    )
    p.add_argument(
        "--run-validation",
        action="store_true",
        help="Generate missing residue_validation.csv via run_emdb_bfactor_validation",
    )
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument(
        "--no-wt2a-supplement",
        action="store_true",
        help="Skip ClpB WT-2A per-residue placement supplement figure (EMD-4941)",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _manifest_rows_with_pdb(manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            pdb = Path(row.get("flexibility_path_or_pdb", "").strip())
            if not pdb.suffix.lower() in {".cif", ".pdb"}:
                continue
            if not pdb.exists():
                print(
                    f"[model_placement] skip EMD-{row['emdb_id']}: no model {pdb}",
                    flush=True,
                )
                continue
            rows.append(row)
    return rows


def _global_resolution(row: dict[str, str]) -> float:
    raw = row.get("global_resolution_a", "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _load_or_validate(
    emd_id: str,
    *,
    manifest: Path,
    run_validation: bool,
) -> list | None:
    out_dir = default_reliability_out_dir(emd_id)
    csv_path = out_dir / "residue_validation.csv"
    if csv_path.is_file():
        return read_residue_validation_csv(csv_path)

    if not run_validation:
        print(
            f"[model_placement] skip EMD-{emd_id}: missing {csv_path} "
            "(pass --run-validation to generate)",
            flush=True,
        )
        return None

    try:
        _, rows, stats, _ = run_emdb_bfactor_validation(
            emd_id,
            manifest=manifest,
            require_b_factor_source=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[model_placement] ERROR EMD-{emd_id}: {exc}", file=sys.stderr)
        return None

    if stats is None or not rows:
        print(f"[model_placement] skip EMD-{emd_id}: validation produced no rows", flush=True)
        return None
    return rows


def _short_label(display_name: str, emdb_id: str) -> str:
    return display_name.split("(")[0].strip() or f"EMD-{emdb_id}"


def _truncate_label(label: str, *, max_len: int = 24) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _is_decoupled(stats) -> bool:
    return np.isfinite(stats.spearman_reliability_vs_cc) and stats.spearman_reliability_vs_cc < 0


def _is_highlighted(stats, xi: float, yi: float, *, disagreement_min: float) -> bool:
    """Maps worth labeling: reliability/CC decoupled or large |x − y| gap."""
    if _is_decoupled(stats):
        return True
    return abs(xi - yi) >= disagreement_min


def _plot_model_placement_scatter(
    usable: list,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    reference_lines: list[tuple],
    disagreement_min: float,
    dpi: int,
) -> None:
    """Gray cohort context; color + side legend for decoupled or high-|x−y| maps."""
    if not usable:
        print(f"[model_placement] no rows for {out_path.name}", flush=True)
        return

    colors = PALETTES["categorical"]
    gaps = np.abs(x_values - y_values)
    highlighted_idx = [
        i
        for i, s in enumerate(usable)
        if _is_highlighted(s, float(x_values[i]), float(y_values[i]), disagreement_min=disagreement_min)
    ]
    highlighted_idx = sorted(highlighted_idx, key=lambda i: float(gaps[i]), reverse=True)
    background_idx = [i for i in range(len(usable)) if i not in highlighted_idx]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    apply(ax)

    if background_idx:
        ax.scatter(
            x_values[background_idx],
            y_values[background_idx],
            s=34,
            c="#c8c8c8",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
            label="other maps",
        )

    hi_colors: list[str] = []
    if highlighted_idx:
        hi_colors = [colors[i % len(colors)] for i in range(len(highlighted_idx))]
        ax.scatter(
            x_values[highlighted_idx],
            y_values[highlighted_idx],
            s=72,
            c=hi_colors,
            edgecolors="0.15",
            linewidths=0.8,
            zorder=4,
        )

    lim_hi = max(0.35, float(np.max(np.concatenate([x_values, y_values]))) + 0.05)
    for spec in reference_lines:
        kind = spec[0]
        if kind == "diag":
            ax.plot([0, lim_hi], [0, lim_hi], color="0.55", linewidth=0.8, linestyle="--", label="y = x")
        elif kind == "hline":
            ax.axhline(spec[1], color="0.75", linewidth=0.6, linestyle=":", label=spec[2])
        elif kind == "vline":
            ax.axvline(spec[1], color="0.75", linewidth=0.6, linestyle=":", label=spec[2])

    ax.set_xlim(-0.02, lim_hi)
    ax.set_ylim(-0.02, lim_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ref_legend = ax.legend(loc="upper left", frameon=False, fontsize=6)

    if highlighted_idx:
        flag_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=hi_colors[j],
                markeredgecolor="0.15",
                markeredgewidth=0.8,
                markersize=7,
                label=_truncate_label(_short_label(usable[idx].display_name, usable[idx].emdb_id))
                + (" *" if _is_decoupled(usable[idx]) else ""),
            )
            for j, idx in enumerate(highlighted_idx)
        ]
        ax.add_artist(ref_legend)
        ax.legend(
            handles=flag_handles,
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=0.93,
            facecolor="white",
            edgecolor="0.85",
            fontsize=5.5,
            title="Flagged maps",
            title_fontsize=6,
        )

    fig.tight_layout()
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)


def _plot_tercile_vs_absolute(
    stats_rows: list,
    *,
    out_path: Path,
    cc_threshold: float,
    disagreement_min: float,
    dpi: int,
) -> None:
    usable = [
        s
        for s in stats_rows
        if np.isfinite(s.frac_cc_below_threshold) and np.isfinite(s.frac_in_omit_zone)
    ]
    x = np.array([s.frac_in_omit_zone for s in usable], dtype=np.float64)
    y = np.array([s.frac_cc_below_threshold for s in usable], dtype=np.float64)

    _plot_model_placement_scatter(
        usable,
        x,
        y,
        out_path=out_path,
        title="Tercile omit vs absolute low-CC placement",
        xlabel="Deposited Cα in omit tercile (fraction)",
        ylabel=f"Deposited Cα with local CC < {cc_threshold:.2f} (fraction)",
        reference_lines=[
            ("diag",),
            ("hline", 0.33, "33% (per-map omit share)"),
            ("vline", 0.33, None),
        ],
        disagreement_min=disagreement_min,
        dpi=dpi,
    )


def _plot_absolute_vs_absolute(
    stats_rows: list,
    *,
    out_path: Path,
    cc_threshold: float,
    reliability_threshold: float,
    disagreement_min: float,
    dpi: int,
) -> None:
    """Both axes use cross-map comparable absolute cutoffs at deposited Cα."""
    usable = [
        s
        for s in stats_rows
        if np.isfinite(s.frac_cc_below_threshold)
        and np.isfinite(s.frac_reliability_below_threshold)
    ]
    x = np.array([s.frac_reliability_below_threshold for s in usable], dtype=np.float64)
    y = np.array([s.frac_cc_below_threshold for s in usable], dtype=np.float64)

    _plot_model_placement_scatter(
        usable,
        x,
        y,
        out_path=out_path,
        title="Absolute reliability vs absolute low-CC placement",
        xlabel=f"Deposited Cα with reliability_score < {reliability_threshold:.2f} (fraction)",
        ylabel=f"Deposited Cα with local CC < {cc_threshold:.2f} (fraction)",
        reference_lines=[
            ("diag",),
            ("hline", cc_threshold, None),
            ("vline", reliability_threshold, None),
        ],
        disagreement_min=disagreement_min,
        dpi=dpi,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    if args.emd_id:
        manifest_row = load_cohort_manifest_row(args.manifest, args.emd_id.strip())
        manifest_rows = [manifest_row]
    else:
        manifest_rows = _manifest_rows_with_pdb(args.manifest)

    stats_rows = []
    for row in manifest_rows:
        emd_id = str(row["emdb_id"]).strip()
        rows = _load_or_validate(emd_id, manifest=args.manifest, run_validation=args.run_validation)
        if rows is None:
            continue
        stats = compute_model_placement_audit_stats(
            rows,
            emdb_id=emd_id,
            display_name=str(row.get("display_name", "")).strip(),
            global_resolution_a=_global_resolution(row),
            cc_threshold=args.cc_threshold,
            reliability_threshold=args.reliability_threshold,
        )
        stats_rows.append(stats)
        decoupled = (
            " decoupled"
            if np.isfinite(stats.spearman_reliability_vs_cc)
            and stats.spearman_reliability_vs_cc < 0
            else ""
        )
        print(
            f"[model_placement] EMD-{emd_id}: mask={stats.frac_in_contour_mask:.1%}, "
            f"omit={stats.frac_in_omit_zone:.1%}, "
            f"rel<{args.reliability_threshold}={stats.frac_reliability_below_threshold:.1%}, "
            f"CC<0.5/0.6/0.7="
            f"{stats.frac_cc_below_0_50:.1%}/"
            f"{stats.frac_cc_below_0_60:.1%}/"
            f"{stats.frac_cc_below_0_70:.1%}, "
            f"rho(rel,CC)={stats.spearman_reliability_vs_cc:+.2f}{decoupled}",
            flush=True,
        )

    if not stats_rows:
        print("[model_placement] no maps processed", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "model_placement_audit.csv"
    write_model_placement_audit_csv(csv_path, stats_rows)
    tercile_png = args.out_dir / "model_placement_tercile_vs_absolute.png"
    absolute_png = args.out_dir / "model_placement_absolute_vs_absolute.png"
    _plot_tercile_vs_absolute(
        stats_rows,
        out_path=tercile_png,
        cc_threshold=args.cc_threshold,
        disagreement_min=args.highlight_disagreement_min,
        dpi=args.dpi,
    )
    _plot_absolute_vs_absolute(
        stats_rows,
        out_path=absolute_png,
        cc_threshold=args.cc_threshold,
        reliability_threshold=args.reliability_threshold,
        disagreement_min=args.highlight_disagreement_min,
        dpi=args.dpi,
    )
    sync_thesis_doc_figure(tercile_png, "fig_3_6_model_placement_tercile_vs_absolute.png")
    sync_thesis_doc_figure(absolute_png, "fig_3_6_model_placement_absolute_vs_absolute.png")
    print(f"[model_placement] wrote {csv_path}", flush=True)

    if not args.no_wt2a_supplement and any(s.emdb_id == "4941" for s in stats_rows):
        _write_wt2a_supplement(
            manifest=args.manifest,
            manifest_rows=manifest_rows,
            run_validation=args.run_validation,
            cohort_out_dir=args.out_dir,
            dpi=args.dpi,
        )

    return 0


def _write_wt2a_supplement(
    *,
    manifest: Path,
    manifest_rows: list[dict[str, str]],
    run_validation: bool,
    cohort_out_dir: Path,
    dpi: int,
) -> None:
    """Per-residue CC/B/zone panel for ClpB WT-2A reviewer pushback."""
    rows = _load_or_validate("4941", manifest=manifest, run_validation=run_validation)
    if rows is None:
        print("[model_placement] skip WT-2A supplement: no residue_validation.csv", flush=True)
        return

    display_name = ""
    for row in manifest_rows:
        if str(row.get("emdb_id", "")).strip() == "4941":
            display_name = str(row.get("display_name", "")).strip()
            break

    out_dir = default_reliability_out_dir("4941") / "figures"
    out_path = out_dir / "clpb_wt2a_placement_supplement.png"
    plot_placement_supplement(
        rows,
        emdb_id="4941",
        display_name=display_name,
        out_path=out_path,
        n_residues=len(rows),
        dpi=dpi,
    )
    cohort_copy = cohort_out_dir / "clpb_wt2a_placement_supplement.png"
    shutil.copy2(out_path, cohort_copy)
    pdf_src = out_path.with_suffix(".pdf")
    if pdf_src.is_file():
        shutil.copy2(pdf_src, cohort_copy.with_suffix(".pdf"))
    sync_thesis_doc_figure(cohort_copy, "fig_s4_clpb_wt2a_placement_supplement.png")
    print(f"[model_placement] wrote {out_path}", flush=True)
    print(f"[model_placement] wrote {cohort_copy}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
