"""Residue-level B-factor vs map reliability (gemmi + cohort manifest).

Reads ``cohort/manifest.csv``, samples ``reliability.npz`` at deposited-model Cα
positions, and writes CSV + figures under ``outputs/emd_<ID>/lh_map_reliability/``.

Example::

    source .venv/bin/activate
    uv pip install gemmi   # or: pip install -e .

    # EMD-49450 (anchor)
    python scripts/run_residue_bfactor_validation.py --emd-id 49450

    # Thesis anchor maps (default for rerun_all_figures)
    python scripts/run_residue_bfactor_validation.py --anchors

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

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.figure_cleanup import prune_lh_retired_figures
from cryoem_mrc.reliability import BUILD_ZONE_COLORS, BUILD_ZONE_LABELS
from cryoem_mrc.repo_paths import BFACTOR_VALIDATION_EMDB_IDS, COHORT_MANIFEST
from cryoem_mrc.structure_validation import (
    BfactorValidationStats,
    load_cohort_manifest_row,
    run_emdb_bfactor_validation,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument(
        "--anchors",
        action="store_true",
        help=f"Run thesis anchor IDs: {', '.join(BFACTOR_VALIDATION_EMDB_IDS)}",
    )
    p.add_argument("--all", action="store_true", help="Run all b_factor rows in manifest with local PDB")
    p.add_argument(
        "--prune-retired-figures",
        action="store_true",
        help="Delete retired bfactor_vs_reliability / bfactor_by_build_zone exports",
    )
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


def _plot_bfactor_validation_panel(
    rows_in_mask,
    stats: BfactorValidationStats,
    out: Path,
    *,
    emd_id: str,
    dpi: int,
) -> None:
    b = np.array([r.b_iso for r in rows_in_mask])
    rel = np.array([r.reliability_score for r in rows_in_mask])
    rho = stats.spearman_b_vs_reliability

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ax_sc, ax_bar = axes

    apply(ax_sc)
    ax_sc.scatter(b, rel, s=8, alpha=0.35, c=PALETTES["categorical"][0], edgecolors="none")
    ax_sc.set_xlabel("Deposited B_iso (model)")
    ax_sc.set_ylabel("reliability_score (map)")
    rho_txt = f", ρ={rho:+.3f}" if np.isfinite(rho) else ""
    ax_sc.set_title(f"B-factor vs reliability (in-mask Cα{rho_txt})")
    label_panel(ax_sc, "a")

    apply(ax_bar)
    zones = [z for z in (0, 1, 2) if z in stats.median_b_by_zone]
    labels = [BUILD_ZONE_LABELS[z] for z in zones]
    medians = [stats.median_b_by_zone[z] for z in zones]
    colors = [BUILD_ZONE_COLORS[z] for z in zones]
    bars = ax_bar.bar(labels, medians, color=colors, edgecolor="0.2", linewidth=0.6)
    ax_bar.set_ylabel("Median B_iso")
    ax_bar.set_title("Median B_iso by build zone")
    for bar, val in zip(bars, medians):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    from matplotlib.patches import Patch

    ax_bar.legend(
        handles=[Patch(facecolor=BUILD_ZONE_COLORS[z], label=BUILD_ZONE_LABELS[z]) for z in (0, 1, 2)],
        loc="upper right",
        fontsize=8,
    )
    label_panel(ax_bar, "b")

    fig.suptitle(f"EMD-{emd_id}: B-factor external validation", fontsize=12)
    fig.tight_layout()
    save_nature(fig, out, dpi=dpi)
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
    prune_retired_figures: bool,
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
        _plot_bfactor_validation_panel(
            rows_in_mask,
            stats,
            fig_dir / "bfactor_validation_panel.png",
            emd_id=emd_id,
            dpi=dpi,
        )
    if prune_retired_figures:
        removed = prune_lh_retired_figures(fig_dir)
        if removed:
            print(f"[bfactor_validation] pruned {len(removed)} retired figure(s)", flush=True)

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
    if not args.all and not args.emd_id and not args.anchors:
        print("Specify --emd-id, --anchors, or --all", file=sys.stderr)
        return 2

    if args.all:
        ids = _emd_ids_for_all(args.manifest)
    elif args.anchors:
        ids = list(BFACTOR_VALIDATION_EMDB_IDS)
    else:
        ids = [args.emd_id.strip()]
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
            prune_retired_figures=args.prune_retired_figures,
        )
        rc = max(rc, code)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
