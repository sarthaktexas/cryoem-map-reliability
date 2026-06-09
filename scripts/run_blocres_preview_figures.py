"""Generate BlocRes-based overview panels on small cohort maps (quick preview).

Skips entries without ``outputs/emd_<ID>/locres_blocres.mrc``. Use ``--emd-id`` to
target one map; default runs the smallest completed BlocRes entries.

Example::

    python scripts/run_blocres_preview_figures.py
    python scripts/run_blocres_preview_figures.py --emd-id 11638
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from cryoem_mrc.repo_paths import (
    COHORT_MANIFEST,
    analysis_dir,
    find_features_npz,
    locres_blocres_mrc,
)

# Smallest completed maps first; exclude IDs the cohort BlocRes job may be running.
DEFAULT_PREVIEW_IDS = ("11638", "16091", "41756", "45261")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", action="append", dest="emd_ids", help="Repeatable; default small-map set")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--exclude", nargs="*", default=(), help="EMDB IDs to skip")
    return p.parse_args(argv)


def _manifest_row(manifest: Path, emdb_id: str) -> dict[str, str] | None:
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("emdb_id", "")).strip() == emdb_id:
                return row
    return None


def _data_dir(reference_mrc: str) -> Path:
    return Path(reference_mrc).parent


def _ready_ids(manifest: Path, candidates: tuple[str, ...], exclude: set[str]) -> list[str]:
    ready: list[str] = []
    for emdb_id in candidates:
        if emdb_id in exclude:
            continue
        if not locres_blocres_mrc(emdb_id).is_file():
            print(f"[blocres_preview] skip EMD-{emdb_id}: no locres_blocres.mrc", flush=True)
            continue
        row = _manifest_row(manifest, emdb_id)
        if row is None:
            print(f"[blocres_preview] skip EMD-{emdb_id}: not in manifest", flush=True)
            continue
        ready.append(emdb_id)
    return ready


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exclude = {str(x).strip() for x in args.exclude}
    candidates = tuple(args.emd_ids) if args.emd_ids else DEFAULT_PREVIEW_IDS
    ready = _ready_ids(args.manifest, candidates, exclude)
    if not ready:
        print("[blocres_preview] no entries ready", file=sys.stderr)
        return 1

    script = Path(__file__).resolve().parent / "run_thesis_overview_figures.py"
    failures = 0
    for emdb_id in ready:
        row = _manifest_row(args.manifest, emdb_id)
        assert row is not None
        data_dir = _data_dir(row["reference_mrc"])
        contour = float(row["contour"])
        out_dir = Path("outputs") / f"emd_{emdb_id}" / "thesis_overview_blocres"
        features = find_features_npz(data_dir, emdb_id, contour)
        cmd = [
            sys.executable,
            str(script),
            "--emd-id",
            emdb_id,
            "--data-dir",
            str(data_dir),
            "--contour",
            str(contour),
            "--halfmap-npz",
            str(analysis_dir(emdb_id) / "halfmap_metrics.npz"),
            "--out-dir",
            str(out_dir),
            "--only",
            "local_resolution_slice",
            "parallel_readouts_density_cc_localres",
            "parallel_readouts_cc_localres",
        ]
        if features is not None:
            cmd.extend(["--features", str(features)])
        print(f"[blocres_preview] EMD-{emdb_id} ({row.get('display_name', '')}) -> {out_dir}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            failures += 1
            print(f"[blocres_preview] FAIL EMD-{emdb_id} exit {rc}", file=sys.stderr, flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
