"""Fetch EMDB global resolutions and write them into ``cohort/manifest.csv``.

Example::

    source .venv/bin/activate
    python scripts/update_cohort_manifest_resolutions.py
    python scripts/update_cohort_manifest_resolutions.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryoem_mrc.cohort_emdb import fetch_emdb_global_resolution_a
from cryoem_mrc.repo_paths import COHORT_MANIFEST


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--dry-run", action="store_true", help="Print resolutions without writing CSV")
    p.add_argument("--delay-s", type=float, default=0.15, help="Pause between EMDB API calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = args.manifest
    if not manifest.is_file():
        print(f"[update_resolutions] missing {manifest}", file=sys.stderr)
        return 2

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[update_resolutions] empty {manifest}", file=sys.stderr)
        return 2

    fieldnames = list(rows[0].keys())
    if "global_resolution_a" not in fieldnames:
        fieldnames.append("global_resolution_a")

    updated = 0
    for row in rows:
        eid = str(row.get("emdb_id", "")).strip()
        if not eid:
            continue
        try:
            res_a = fetch_emdb_global_resolution_a(eid)
        except RuntimeError as exc:
            print(f"[update_resolutions] EMD-{eid}: {exc}", file=sys.stderr)
            res_a = None
        if res_a is None:
            print(f"[update_resolutions] EMD-{eid}: no resolution", file=sys.stderr)
            continue
        new_val = f"{res_a:.2f}"
        old_val = str(row.get("global_resolution_a", "")).strip()
        if old_val != new_val:
            updated += 1
        row["global_resolution_a"] = new_val
        print(f"EMD-{eid}: {new_val} Å")
        time.sleep(args.delay_s)

    if args.dry_run:
        print(f"[update_resolutions] dry run: would update {updated} rows")
        return 0

    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[update_resolutions] wrote {manifest} ({updated} values changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
