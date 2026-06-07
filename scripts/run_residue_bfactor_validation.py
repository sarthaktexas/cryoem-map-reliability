"""Residue-level B-factor vs map reliability (gemmi + cohort manifest).

Reads ``cohort/manifest.csv``, samples ``reliability.npz`` at deposited-model Cα
positions, and writes CSV + figures under ``outputs/emd_<ID>/lh_map_reliability/``.

Example::

    source .venv/bin/activate
    uv pip install gemmi   # or: pip install -e .

    # EMD-49450 (anchor)
    python scripts/run_residue_bfactor_validation.py --emd-id 49450

    # All manifest rows with flexibility_source=b_factor and local PDB
    python scripts/run_residue_bfactor_validation.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryoem_mrc.repo_paths import COHORT_MANIFEST
from cryoem_mrc.structure_validation import (
    BfactorValidationStats,
    load_cohort_manifest_row,
    run_emdb_bfactor_validation,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument("--all", action="store_true", help="Run all b_factor rows in manifest with local PDB")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--reliability-npz", type=Path, default=None)
    p.add_argument("--reference", type=Path, default=None, help="Override deposited primary map")
    p.add_argument("--pdb", type=Path, default=None, help="Override mmCIF/PDB path")
    p.add_argument("--contour", type=float, default=None)
    p.add_argument("--halfmap-npz", type=Path, default=None)
    p.add_argument("--features-npz", type=Path, default=None, help="Optional local_variance")
    p.add_argument("--window-radius", type=int, default=0, help="0=nearest voxel; 1=3^3 mean, etc.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def _sampling_label(radius: int) -> str:
    if radius <= 0:
        return "nearest voxel"
    side = 2 * radius + 1
    return f"{side}³ voxel window mean"


def _plot_scatter(rows_in_mask, out: Path, *, emd_id: str, dpi: int) -> None:
    b = np.array([r.b_iso for r in rows_in_mask])
    rel = np.array([r.reliability_score for r in rows_in_mask])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(b, rel, s=8, alpha=0.35, c="#1f77b4", edgecolors="none")
    ax.set_xlabel("Deposited B_iso (model)")
    ax.set_ylabel("reliability_score (map)")
    ax.set_title(f"EMD-{emd_id}: B-factor vs reliability (in-mask Cα)")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_zone_medians(stats: BfactorValidationStats, out: Path, *, emd_id: str, dpi: int) -> None:
    labels = ["omit", "caution", "build"]
    vals = [stats.median_b_by_zone.get(z, float("nan")) for z in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, color=["#d62728", "#ffbb78", "#2ca02c"], alpha=0.9)
    ax.set_ylabel("Median B_iso")
    ax.set_title(f"EMD-{emd_id}: B-factor by build zone")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _run_one(
    emd_id: str,
    *,
    manifest: Path,
    reliability_npz: Path | None,
    reference: Path | None,
    pdb: Path | None,
    contour: float | None,
    halfmap_npz: Path | None,
    features_npz: Path | None,
    window_radius: int,
    dpi: int,
) -> int:
    try:
        code, rows, stats, out_dir = run_emdb_bfactor_validation(
            emd_id,
            manifest=manifest,
            reliability_npz=reliability_npz,
            reference=reference,
            pdb=pdb,
            contour=contour,
            halfmap_npz=halfmap_npz,
            features_npz=features_npz,
            window_radius=window_radius,
        )
    except FileNotFoundError as e:
        print(f"[bfactor_validation] ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[bfactor_validation] ERROR: {e}", file=sys.stderr)
        return 2

    if stats is None:
        print(f"[bfactor_validation] skip EMD-{emd_id}", flush=True)
        return code

    row = load_cohort_manifest_row(manifest, emd_id)
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    rows_in_mask = [r for r in rows if r.in_contour_mask]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bfactor_validation_stats.json").write_text(
        json.dumps(
            {
                "emdb_id": stats.emdb_id,
                "n_residues": stats.n_residues,
                "n_in_mask": stats.n_in_mask,
                "spearman_b_vs_reliability": stats.spearman_b_vs_reliability,
                "spearman_b_vs_H_repro": stats.spearman_b_vs_H_repro,
                "spearman_b_vs_build_zone": stats.spearman_b_vs_build_zone,
                "partial_b_vs_reliability_given_variance": stats.partial_b_vs_reliability_given_variance,
                "median_b_by_zone": stats.median_b_by_zone,
                "pdb": str(pdb_path),
                "contour": contour_val,
            },
            indent=2,
        )
        + "\n"
    )
    if rows_in_mask:
        _plot_scatter(rows_in_mask, fig_dir / "bfactor_vs_reliability.png", emd_id=emd_id, dpi=dpi)
    _plot_zone_medians(stats, fig_dir / "bfactor_by_build_zone.png", emd_id=emd_id, dpi=dpi)

    print(
        f"[bfactor_validation] EMD-{emd_id}: n={stats.n_in_mask} in-mask, "
        f"ρ(B, reliability)={stats.spearman_b_vs_reliability:+.3f}",
        flush=True,
    )
    return 0


def _emd_ids_for_all(manifest: Path) -> list[str]:
    import csv

    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flexibility_source", "").strip() != "b_factor":
                continue
            pdb = Path(row.get("flexibility_path_or_pdb", ""))
            if not pdb.exists():
                print(f"[bfactor_validation] skip EMD-{row['emdb_id']}: no PDB {pdb}", flush=True)
                continue
            ids.append(str(row["emdb_id"]).strip())
    return ids


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    ids = _emd_ids_for_all(args.manifest) if args.all else [args.emd_id.strip()]
    rc = 0
    for emd_id in ids:
        code = _run_one(
            emd_id,
            manifest=args.manifest,
            reliability_npz=args.reliability_npz,
            reference=args.reference,
            pdb=args.pdb,
            contour=args.contour,
            halfmap_npz=args.halfmap_npz,
            features_npz=args.features_npz,
            window_radius=args.window_radius,
            dpi=args.dpi,
        )
        rc = max(rc, code)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
