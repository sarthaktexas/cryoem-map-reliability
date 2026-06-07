"""Cohort-wide summary heatmap from per-map LH reliability exports.

Reads ``outputs/emd_<ID>/lh_map_reliability/run_metadata.json`` (and optional
``bfactor_validation_stats.json``) and writes:

- ``outputs/cohort_summary/cohort_metrics.csv``
- ``outputs/cohort_summary/cohort_metrics_heatmap.png``

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

from cryoem_mrc.repo_paths import OUTPUTS_ROOT
from cryoem_mrc.thesis_figures import (
    collect_cohort_metrics,
    plot_cohort_metrics_heatmap,
    write_cohort_metrics_csv,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-b-factor", action="store_true", help="Omit B_iso column")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = collect_cohort_metrics(args.outputs_root)
    if not rows:
        print("[cohort_summary] no run_metadata.json found under outputs/emd_*/lh_map_reliability/",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "cohort_metrics.csv"
    png_path = args.out_dir / "cohort_metrics_heatmap.png"
    write_cohort_metrics_csv(rows, csv_path)
    plot_cohort_metrics_heatmap(
        rows,
        save_path=png_path,
        dpi=args.dpi,
        include_b_factor=not args.no_b_factor,
    )
    print(f"[cohort_summary] {len(rows)} maps -> {csv_path} and {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
