"""Export per-residue cross-metric tables via :func:`cryoem_mrc.metric_comparison.load_all_metrics`.

Writes under ``outputs/emd_<ID>/metric_comparison/``:

- ``residue_metrics.csv``
- ``cross_metric_correlations.csv``

Example::

    source .venv/bin/activate
    python scripts/run_metric_comparison_export.py --emd-id 49450
    python scripts/run_metric_comparison_export.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from cryoem_mrc.metric_comparison import compute_cross_metric_correlations, load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST, emd_output_dir, lh_map_reliability_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    return p.parse_args(argv)


def _eligible_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            pdb = row.get("flexibility_path_or_pdb", "").strip()
            if not pdb or not Path(pdb).is_file():
                continue
            if not (lh_map_reliability_dir(eid) / "reliability.npz").is_file():
                print(f"[metric_export] skip EMD-{eid}: no reliability.npz", flush=True)
                continue
            ids.append(eid)
    return ids


def _export_one(emdb_id: str, *, manifest: Path) -> int:
    try:
        df = load_all_metrics(emdb_id, manifest=manifest)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[metric_export] FAIL EMD-{emdb_id}: {exc}", file=sys.stderr, flush=True)
        return 1

    out_dir = emd_output_dir(emdb_id) / "metric_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "residue_metrics.csv"
    df.to_csv(metrics_path, index=False)

    corr = compute_cross_metric_correlations(df)
    corr_path = out_dir / "cross_metric_correlations.csv"
    corr.to_csv(corr_path)

    n_loc = int(df["local_resolution"].notna().sum()) if "local_resolution" in df.columns else 0
    print(
        f"[metric_export] EMD-{emdb_id}: {len(df)} residues, "
        f"local_resolution finite={n_loc} -> {metrics_path}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    ids = _eligible_ids(args.manifest) if args.all else [args.emd_id.strip()]
    rc = 0
    for emdb_id in ids:
        rc = max(rc, _export_one(emdb_id, manifest=args.manifest))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
