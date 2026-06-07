"""Run avg → features → analysis → LH export → B-factor for cohort manifest rows.

Skips entries already present under ``outputs/emd_<ID>/lh_map_reliability/reliability.npz``
unless ``--force``. Excludes ``excluded`` / ``optional`` manifest sources.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_cohort_pipeline.py --pending
    PYTHONUNBUFFERED=1 python scripts/run_cohort_pipeline.py --emd-id 23130
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc, save_volume_like_reference
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.repo_paths import COHORT_MANIFEST, analysis_dir, lh_map_reliability_dir

REPO = Path(__file__).resolve().parents[1]
DONE_IDS = frozenset({"49450", "11638", "23129"})
SKIP_SOURCES = frozenset({"excluded", "optional"})


def _contour_tag(contour: float) -> str:
    return f"t{int(round(float(contour) * 1000)):04d}"


def _feature_start_threshold(avg_path: Path, ref_path: Path, mask_contour: float) -> float:
    """
    Depositor contour applies to the sharpened reference map (Decision 002).

    Averaged half maps are unsharpened and often sit on a lower absolute scale.
    When max(avg) inside the reference mask is below ``mask_contour``, use 0 so
    feature extraction is not wiped inside the macromolecule (EMD-52525 case).
    """
    ref = load_mrc(ref_path, dtype=np.float32)
    avg = load_mrc(avg_path, dtype=np.float32)
    mask = build_contour_mask(ref, mask_contour)
    if not mask.any():
        return mask_contour
    avg_max_in_mask = float(np.max(avg[mask]))
    if avg_max_in_mask < mask_contour:
        print(
            f"[cohort] avg max in mask ({avg_max_in_mask:.4g}) < depositor contour "
            f"({mask_contour:g}); feature --start-threshold 0",
            flush=True,
        )
        return 0.0
    return mask_contour


def _rows(manifest: Path, *, emd_id: str | None, pending: bool) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            if row.get("flexibility_source", "").strip() in SKIP_SOURCES:
                continue
            if emd_id is not None and eid != emd_id.strip():
                continue
            if pending and eid in DONE_IDS:
                continue
            ref = Path(row["reference_mrc"])
            if not ref.is_file():
                print(f"[cohort] skip EMD-{eid}: missing {ref}", flush=True)
                continue
            out.append(row)
    return out


def _ensure_avg(row: dict[str, str]) -> Path:
    ref = Path(row["reference_mrc"])
    data_dir = ref.parent
    eid = str(row["emdb_id"]).strip()
    emd = f"emd_{eid}"
    avg_path = data_dir / f"{emd}_avg.map"
    if avg_path.is_file():
        print(f"[cohort] EMD-{eid}: reuse {avg_path.name}", flush=True)
        return avg_path
    print(f"[cohort] EMD-{eid}: writing {avg_path.name}", flush=True)
    bundle = load_full_and_half_maps(
        ref,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        reference="full",
        dtype=np.float32,
    )
    avg = (0.5 * (bundle.half1.data + bundle.half2.data)).astype(np.float32)
    save_volume_like_reference(ref, avg, avg_path)
    return avg_path


def _run(cmd: list[str], *, label: str) -> None:
    print(f"[cohort] {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def _process_row(row: dict[str, str], *, force: bool, skip_bfactor: bool) -> int:
    eid = str(row["emdb_id"]).strip()
    contour = float(row["contour"])
    if row["contour"].strip().upper() == "TBD":
        print(f"[cohort] skip EMD-{eid}: contour TBD in manifest", flush=True)
        return 0

    ref = Path(row["reference_mrc"])
    data_dir = ref.parent
    emd = f"emd_{eid}"
    lh_dir = lh_map_reliability_dir(eid)
    reliability_npz = lh_dir / "reliability.npz"

    if reliability_npz.is_file() and not force:
        print(f"[cohort] EMD-{eid}: already done ({reliability_npz})", flush=True)
        return 0

    avg_path = _ensure_avg(row)
    feature_thr = _feature_start_threshold(avg_path, ref, contour)
    tag = _contour_tag(feature_thr)
    features = data_dir / f"{emd}_avg_features_{tag}.npz"
    py = sys.executable

    if not features.is_file() or force:
        _run(
            [
                py,
                "-m",
                "cryoem_mrc",
                str(avg_path),
                "--start-threshold",
                str(feature_thr),
                "--float32",
                "--out",
                str(features),
            ],
            label=f"EMD-{eid} features",
        )
    else:
        print(f"[cohort] EMD-{eid}: reuse {features.name}", flush=True)

    out_analysis = analysis_dir(eid)
    metrics_npz = out_analysis / "halfmap_metrics.npz"
    analysis_cmd = [
        py,
        "scripts/run_analysis.py",
        "--features",
        str(features),
        "--half1",
        row["half1_path"],
        "--half2",
        row["half2_path"],
        "--reference",
        str(ref),
        "--contour",
        str(contour),
        "--out-dir",
        str(out_analysis),
    ]
    # Half-map metrics depend only on the halves, not the feature threshold.
    if metrics_npz.is_file():
        analysis_cmd.append("--skip-halfmap-metrics")
    _run(analysis_cmd, label=f"EMD-{eid} analysis")

    _run(
        [
            py,
            "scripts/run_lh_map_reliability_export.py",
            "--data-dir",
            str(data_dir),
            "--emd-id",
            eid,
            "--contour",
            str(contour),
            "--features",
            str(features),
            "--halfmap-npz",
            str(metrics_npz),
            "--out-dir",
            str(lh_dir),
        ],
        label=f"EMD-{eid} lh_export",
    )

    src = row.get("flexibility_source", "").strip()
    pdb = Path(row.get("flexibility_path_or_pdb", ""))
    if not skip_bfactor and src == "b_factor" and pdb.is_file():
        bfac_cmd = [py, "scripts/run_residue_bfactor_validation.py", "--emd-id", eid]
        if features.is_file():
            bfac_cmd.extend(["--features-npz", str(features)])
        _run(bfac_cmd, label=f"EMD-{eid} bfactor")
    elif src == "b_factor" and not pdb.is_file():
        print(f"[cohort] EMD-{eid}: skip bfactor (no PDB {pdb})", flush=True)

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pending", action="store_true", help="All manifest rows except validated trio")
    g.add_argument("--emd-id", type=str, help="Single EMDB ID")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--force", action="store_true", help="Re-run even if reliability.npz exists")
    p.add_argument("--skip-bfactor", action="store_true", help="Stop after lh_export")
    p.add_argument(
        "--max-voxels",
        type=float,
        default=None,
        metavar="N",
        help="Skip maps with more than N million voxels (e.g. 100 for ~8 GB RAM)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _rows(args.manifest, emd_id=args.emd_id, pending=args.pending)
    if not rows:
        print("[cohort] nothing to run", flush=True)
        return 0

    # Smallest boxes first (faster feedback, lower peak RAM early).
    import mrcfile

    sized: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        with mrcfile.open(row["reference_mrc"], permissive=True) as m:
            nvox = float(m.data.size)
        sized.append((nvox, row))
    sized.sort(key=lambda x: x[0])

    rc = 0
    for nvox, row in sized:
        eid = str(row["emdb_id"]).strip()
        if args.max_voxels is not None and nvox > args.max_voxels * 1e6:
            print(
                f"[cohort] skip EMD-{eid}: {nvox/1e6:.1f}M voxels > {args.max_voxels}M limit",
                flush=True,
            )
            continue
        try:
            rc = max(rc, _process_row(row, force=args.force, skip_bfactor=args.skip_bfactor))
        except subprocess.CalledProcessError as e:
            print(f"[cohort] FAILED EMD-{eid}: {e}", file=sys.stderr, flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
