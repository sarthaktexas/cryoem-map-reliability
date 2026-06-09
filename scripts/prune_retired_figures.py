#!/usr/bin/env python3
"""Delete retired per-map analysis/lh_map_reliability figure exports.

Example::

    uv run python scripts/prune_retired_figures.py
    uv run python scripts/prune_retired_figures.py --emd-id 49450
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryoem_mrc.figure_cleanup import (
    prune_analysis_scatter_figures,
    prune_lh_retired_figures,
    prune_retired_figures_under_outputs,
)
from cryoem_mrc.repo_paths import OUTPUTS_ROOT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (default: entire cohort)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.emd_id:
        eid = args.emd_id.strip()
        emd_dir = args.outputs_root / f"emd_{eid}"
        removed = prune_analysis_scatter_figures(emd_dir / "analysis" / "figures")
        removed += prune_lh_retired_figures(emd_dir / "lh_map_reliability" / "figures")
        print(f"[prune] EMD-{eid}: removed {len(removed)} file(s)", flush=True)
        for path in removed:
            print(f"  {path}", flush=True)
        return 0

    summary = prune_retired_figures_under_outputs(args.outputs_root)
    n_analysis = len(summary["analysis"])
    n_lh = len(summary["lh_map_reliability"])
    print(f"[prune] removed {n_analysis} analysis + {n_lh} lh_map_reliability file(s)", flush=True)
    if n_analysis + n_lh == 0:
        print("[prune] nothing to delete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
