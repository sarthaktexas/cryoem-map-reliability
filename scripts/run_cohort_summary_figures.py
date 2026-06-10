"""Cohort-wide summary from per-map LH reliability exports.

Writes ``outputs/cohort_summary/cohort_metrics.csv`` plus a small set of
figures that show clear cohort-level structure (variance coupling, resolution
bins, protein class). Weak or redundant diagnostics are omitted.

Example::

    source .venv/bin/activate
    python scripts/run_cohort_summary_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import style.nature  # noqa: F401 — apply Nature rcParams before thesis figure imports

from cryoem_mrc.repo_paths import OUTPUTS_ROOT, sync_thesis_narrative_cohort_figures
from cryoem_mrc.thesis_figures import (
    collect_cohort_metrics,
    plot_cohort_metrics_heatmap,
    plot_cohort_reliability_by_class,
    plot_cohort_reliability_by_resolution_bin,
    plot_cohort_reliability_by_resolution_bin_by_na_fraction,
    plot_cohort_reliability_by_var_cc_bin,
    plot_cohort_variance_vs_reliability_cc,
    write_cohort_metrics_csv,
)

# Figures no longer exported — removed from disk when this script runs.
_RETIRED_FIGURE_STEMS = (
    "cohort_resolution_vs_reliability",
    "cohort_resolution_vs_reliability_by_na_fraction",
    "cohort_mask_size_vs_reliability",
    "cohort_mask_size_vs_reliability_by_na_fraction",
    "cohort_contour_vs_reliability",
    "cohort_contour_vs_reliability_by_na_fraction",
    "cohort_size_vs_reliability",
    "cohort_size_vs_reliability_by_na_fraction",
    "cohort_partial_reliability_by_class",
    "cohort_partial_reliability_by_class_by_na_fraction",
    "cohort_partial_reliability_by_resolution_bin",
    "cohort_partial_reliability_by_resolution_bin_by_na_fraction",
    "cohort_partial_reliability_by_var_cc_bin",
    "cohort_partial_reliability_by_var_cc_bin_by_na_fraction",
    "cohort_reliability_by_flexibility_source",
    "cohort_reliability_by_flexibility_source_by_na_fraction",
    "cohort_bfactor_vs_reliability_cc",
    "cohort_bfactor_vs_reliability_cc_by_na_fraction",
    "cohort_reliability_by_class_by_na_fraction",
    "cohort_reliability_by_var_cc_bin_by_na_fraction",
    "cohort_variance_vs_reliability_cc_by_na_fraction",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-b-factor", action="store_true", help="Omit B_iso column")
    p.add_argument(
        "--emdb-ids",
        type=str,
        default=None,
        help="Comma-separated EMDB IDs to include (default: all maps with run_metadata.json)",
    )
    return p.parse_args(argv)


def _retire_old_figures(out_dir: Path) -> None:
    for stem in _RETIRED_FIGURE_STEMS:
        for path in (out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"):
            if path.is_file():
                path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = collect_cohort_metrics(args.outputs_root)
    if args.emdb_ids:
        keep = {e.strip() for e in args.emdb_ids.split(",") if e.strip()}
        rows = [r for r in rows if r.emdb_id in keep]
    if not rows:
        print("[cohort_summary] no run_metadata.json found under outputs/emd_*/lh_map_reliability/",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _retire_old_figures(args.out_dir)
    csv_path = args.out_dir / "cohort_metrics.csv"
    figure_jobs: list[tuple[str, object]] = [
        ("cohort_metrics_heatmap.png", lambda: plot_cohort_metrics_heatmap(
            rows,
            save_path=args.out_dir / "cohort_metrics_heatmap.png",
            dpi=args.dpi,
            include_b_factor=not args.no_b_factor,
        )),
        ("cohort_variance_vs_reliability_cc.png", lambda: plot_cohort_variance_vs_reliability_cc(
            rows, save_path=args.out_dir / "cohort_variance_vs_reliability_cc.png", dpi=args.dpi,
        )),
        ("cohort_reliability_by_var_cc_bin.png", lambda: plot_cohort_reliability_by_var_cc_bin(
            rows, save_path=args.out_dir / "cohort_reliability_by_var_cc_bin.png", dpi=args.dpi,
        )),
        ("cohort_reliability_by_resolution_bin.png", lambda: plot_cohort_reliability_by_resolution_bin(
            rows, save_path=args.out_dir / "cohort_reliability_by_resolution_bin.png", dpi=args.dpi,
        )),
        ("cohort_reliability_by_resolution_bin_by_na_fraction.png", lambda: plot_cohort_reliability_by_resolution_bin_by_na_fraction(
            rows, save_path=args.out_dir / "cohort_reliability_by_resolution_bin_by_na_fraction.png", dpi=args.dpi,
        )),
        ("cohort_reliability_by_class.png", lambda: plot_cohort_reliability_by_class(
            rows, save_path=args.out_dir / "cohort_reliability_by_class.png", dpi=args.dpi,
        )),
    ]

    write_cohort_metrics_csv(rows, csv_path)
    written = [csv_path.name]
    for name, plot_fn in figure_jobs:
        plot_fn()
        written.append(name)

    synced = sync_thesis_narrative_cohort_figures(args.out_dir)
    print(f"[cohort_summary] {len(rows)} maps -> {', '.join(written)}", flush=True)
    if synced:
        print(f"[cohort_summary] synced {len(synced)} thesis figures -> docs/figures/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
