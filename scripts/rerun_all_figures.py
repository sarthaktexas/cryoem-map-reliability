#!/usr/bin/env python3
"""Regenerate thesis/publication figures (PDF+PNG for 2D; PNG only if 3D).

Example::

    uv run python scripts/rerun_all_figures.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from cryoem_mrc.repo_paths import (
    BFACTOR_VALIDATION_EMDB_IDS,
    COHORT_MANIFEST,
    find_features_npz,
    halfmap_metrics_npz,
    lh_map_reliability_dir,
)

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
SKIP_SOURCES = frozenset({"excluded", "optional"})

CONFORMATION_PAIRS = [
    ("23129", "23130"),
    ("41596", "41598"),
    ("48923", "48534"),  # MgtA E2·Mg·BeF₃ vs E2P·Mg (canonical cycle pair)
    ("24120", "25418"),
    ("45604", "45603"),
    ("28498", "28487"),
    ("13308", "16119"),
    ("4940", "4941"),
    ("11497", "11494"),
    # 49450-based MgtA pairs: assembly mismatch (3556 vs ~888–1778 Cα); supplementary only
    ("49450", "48923"),
    ("49450", "48534"),
]


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 60}\n[figures] {label}\n{'=' * 60}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"[figures] FAILED ({rc}): {label}", file=sys.stderr, flush=True)
    return rc


def manifest_rows(*, emdb_ids: frozenset[str] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with COHORT_MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            if emdb_ids is not None and eid not in emdb_ids:
                continue
            if row.get("flexibility_source", "").strip() in SKIP_SOURCES:
                continue
            if row.get("contour", "").strip().upper() == "TBD":
                continue
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    if not PY.is_file():
        print("Missing .venv/bin/python — run: uv pip install -e .", file=sys.stderr)
        return 2

    rows = manifest_rows(emdb_ids=None)
    if not rows:
        print("[figures] no manifest rows matched", file=sys.stderr)
        return 2

    rc = 0
    cohort_ids_arg = ",".join(sorted(r["emdb_id"].strip() for r in rows))

    rc = max(
        rc,
        run(
            [str(PY), "scripts/run_cohort_summary_figures.py", "--emdb-ids", cohort_ids_arg],
            "cohort summary heatmap",
        ),
    )

    rc = max(
        rc,
        run([str(PY), "scripts/run_thesis_overview_figures.py"], "thesis overview (EMD-49450)"),
    )

    for row in rows:
        eid = str(row["emdb_id"]).strip()
        lh_dir = lh_map_reliability_dir(eid)
        if not (lh_dir / "reliability.npz").is_file():
            print(f"[figures] skip lh_export EMD-{eid}: no reliability.npz", flush=True)
            continue
        ref = Path(row["reference_mrc"])
        features = find_features_npz(ref.parent, eid, float(row["contour"]))
        if features is None:
            print(f"[figures] skip lh_export EMD-{eid}: no features NPZ", flush=True)
            continue
        metrics = halfmap_metrics_npz(eid)
        if not metrics.is_file():
            print(f"[figures] skip lh_export EMD-{eid}: no halfmap_metrics.npz", flush=True)
            continue
        rc = max(
            rc,
            run(
                [
                    str(PY),
                    "scripts/run_lh_map_reliability_export.py",
                    "--data-dir",
                    str(ref.parent),
                    "--emd-id",
                    eid,
                    "--contour",
                    str(row["contour"]),
                    "--features",
                    str(features),
                    "--halfmap-npz",
                    str(metrics),
                    "--out-dir",
                    str(lh_dir),
                    "--prune-retired-figures",
                ],
                f"lh_map_reliability EMD-{eid}",
            ),
        )

    rc = max(
        rc,
        run(
            [str(PY), "scripts/prune_retired_figures.py"],
            "prune retired analysis/lh scatter figures",
        ),
    )

    manifest_by_id = {str(r["emdb_id"]).strip(): r for r in rows}
    for eid in BFACTOR_VALIDATION_EMDB_IDS:
        row = manifest_by_id.get(eid)
        if row is None:
            print(f"[figures] skip bfactor EMD-{eid}: not in manifest", flush=True)
            continue
        if row.get("flexibility_source", "").strip() != "b_factor":
            print(f"[figures] skip bfactor EMD-{eid}: flexibility_source != b_factor", flush=True)
            continue
        pdb = Path(row.get("flexibility_path_or_pdb", ""))
        if not pdb.is_file():
            print(f"[figures] skip bfactor EMD-{eid}: no PDB", flush=True)
            continue
        rc = max(
            rc,
            run(
                [
                    str(PY),
                    "scripts/run_residue_bfactor_validation.py",
                    "--emd-id",
                    eid,
                    "--prune-retired-figures",
                ],
                f"B-factor validation EMD-{eid}",
            ),
        )

    for emd_a, emd_b in CONFORMATION_PAIRS:
        rc = max(
            rc,
            run(
                [
                    str(PY),
                    "scripts/run_residue_bfactor_conformation_pair.py",
                    "--emd-a",
                    emd_a,
                    "--emd-b",
                    emd_b,
                ],
                f"conformation pair EMD-{emd_a} vs {emd_b}",
            ),
        )

    print(f"\n[figures] batch finished (max exit code {rc})", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
