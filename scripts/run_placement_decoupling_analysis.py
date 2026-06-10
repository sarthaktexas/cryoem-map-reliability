"""Cohort placement-decoupling analysis with Tier-1 controls and summary figures.

Reads ``residue_validation.csv`` per map, quantifies reliability/CC decoupling at
deposited Cα, runs sharpening/contour/window controls on flagged maps, and writes
``outputs/cohort_summary/placement_decoupling*.csv|.png``.

Example::

    source .venv/bin/activate
    python scripts/run_placement_decoupling_analysis.py --all
    python scripts/run_placement_decoupling_analysis.py --controls-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import PALETTES, apply, savefig as save_nature

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc
from cryoem_mrc.placement_decoupling import (
    PlacementDecouplingRow,
    analyze_residue_rows,
    load_decoupling_cohort,
    recompute_rho_at_ca,
    write_decoupling_csv,
)
from cryoem_mrc.placement_supplement import plot_placement_supplement
from cryoem_mrc.reliability import percentile_rank_in_mask
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, lh_map_reliability_dir
from cryoem_mrc.structure_validation import (
    iter_ca_residues,
    load_cohort_manifest_row,
    read_residue_validation_csv,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="Full cohort (default)")
    p.add_argument("--controls-only", action="store_true", help="Skip cohort CSV; rerun controls from audit")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--rho-threshold",
        type=float,
        default=-0.3,
        help="Flag strong decoupling when ρ(rel,CC) below this (default -0.3)",
    )
    return p.parse_args(argv)


def _enrich_t_rank_from_npz(
    rows: list[PlacementDecouplingRow],
    *,
    manifest: Path,
) -> list[PlacementDecouplingRow]:
    """Add ρ(T-rank, CC) from stored reliability_fluctuation volumes."""
    out: list[PlacementDecouplingRow] = []
    for rec in rows:
        npz = lh_map_reliability_dir(rec.emdb_id) / "reliability.npz"
        rv = lh_map_reliability_dir(rec.emdb_id) / "residue_validation.csv"
        if not npz.is_file() or not rv.is_file():
            out.append(rec)
            continue
        row = load_cohort_manifest_row(manifest, rec.emdb_id)
        ref = Path(row["reference_mrc"])
        contour = float(row["contour"])
        residues = read_residue_validation_csv(rv)
        with np.load(npz, allow_pickle=False) as d:
            t = np.asarray(d["reliability_fluctuation"], dtype=np.float32)
        ref_vol = load_mrc(ref, dtype=np.float32)
        mask = build_contour_mask(ref_vol, contour)
        t_rank = percentile_rank_in_mask(t, mask)
        from cryoem_mrc.map_grid import load_map_grid
        from cryoem_mrc.structure_validation import sample_volume_at_ca
        from scipy import stats as sp_stats

        grid = load_map_grid(ref, dtype=np.float32)
        ca = iter_ca_residues(Path(row["flexibility_path_or_pdb"]))
        t_s = sample_volume_at_ca(t_rank, grid, ca)
        cc_vals = []
        t_vals = []
        for i, r in enumerate(residues):
            if not r.in_contour_mask:
                continue
            if not np.isfinite(r.windowed_halfmap_correlation):
                continue
            if i < len(t_s) and np.isfinite(t_s[i]):
                cc_vals.append(r.windowed_halfmap_correlation)
                t_vals.append(float(t_s[i]))
        rho_t = float("nan")
        if len(cc_vals) >= 30:
            rho_t, _ = sp_stats.spearmanr(t_vals, cc_vals)
        out.append(
            PlacementDecouplingRow(
                **{
                    **rec.__dict__,
                    "rho_t_rank_vs_cc": float(rho_t),
                }
            )
        )
    return out


def _map_too_heavy_for_controls(emd_id: str, *, manifest: Path) -> bool:
    """Skip full-map recomputation on huge grids (e.g. 70S ribosome, spliceosome)."""
    return emd_id in {"24120", "62841", "13308", "16119"}


def _build_controls(
    decoupled_ids: list[str],
    *,
    manifest: Path,
    rho_threshold: float,
) -> list[dict]:
    """Tier-1 controls on decoupled maps + matched high-ρ controls."""
    controls: list[dict] = []
    targets = {eid for eid in decoupled_ids if not _map_too_heavy_for_controls(eid, manifest=manifest)}
    # Matched controls: strong positive ρ, similar resolution band
    with manifest.open(newline="") as f:
        audit_rows = list(csv.DictReader(f))
    audit_path = OUTPUTS_ROOT / "cohort_summary" / "model_placement_audit.csv"
    positives: list[str] = []
    if audit_path.is_file():
        with audit_path.open(newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get("spearman_reliability_vs_cc", "").strip()
                rho = float(raw) if raw else float("nan")
                if np.isfinite(rho) and rho > 0.85:
                    positives.append(row["emdb_id"])
    control_ids = sorted(targets | {e for e in positives[:3] if not _map_too_heavy_for_controls(e, manifest=manifest)})
    for emd_id in control_ids:
        print(f"[decouple] controls EMD-{emd_id}", flush=True)
        base = recompute_rho_at_ca(emd_id, manifest=manifest, rho_source="avg_half")
        primary = recompute_rho_at_ca(emd_id, manifest=manifest, rho_source="primary")
        t_only = recompute_rho_at_ca(emd_id, manifest=manifest, rho_source="t_only")
        c09 = recompute_rho_at_ca(emd_id, manifest=manifest, contour_scale=0.9)
        c11 = recompute_rho_at_ca(emd_id, manifest=manifest, contour_scale=1.1)
        w3 = recompute_rho_at_ca(emd_id, manifest=manifest, cc_window=3)
        w7 = recompute_rho_at_ca(emd_id, manifest=manifest, cc_window=7)
        controls.append(
            {
                "emdb_id": emd_id,
                "cohort_flag": emd_id in targets,
                "rho_avg_half": base["rho_rel_vs_cc"],
                "rho_primary_map": primary["rho_rel_vs_cc"],
                "rho_t_only_rank": t_only["rho_rel_vs_cc"],
                "rho_contour_0_9x": c09["rho_rel_vs_cc"],
                "rho_contour_1_1x": c11["rho_rel_vs_cc"],
                "rho_cc_window_3": w3["rho_rel_vs_cc"],
                "rho_cc_window_7": w7["rho_rel_vs_cc"],
                "sign_flip_primary": bool(
                    np.isfinite(base["rho_rel_vs_cc"])
                    and np.isfinite(primary["rho_rel_vs_cc"])
                    and np.sign(base["rho_rel_vs_cc"]) != np.sign(primary["rho_rel_vs_cc"])
                ),
                "strong_decoupling": bool(
                    np.isfinite(base["rho_rel_vs_cc"]) and base["rho_rel_vs_cc"] <= rho_threshold
                ),
            }
        )
    return controls


def _write_controls_csv(path: Path, controls: list[dict]) -> None:
    if not controls:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(controls[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in controls:
            w.writerow(row)


def _plot_cohort(rows: list[PlacementDecouplingRow], out_path: Path, *, dpi: int) -> None:
    usable = [r for r in rows if np.isfinite(r.rho_rel_vs_cc) and r.frac_in_contour_mask >= 0.3]
    if not usable:
        return

    colors = PALETTES["categorical"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    apply(axes[0])
    apply(axes[1])

    rho = np.array([r.rho_rel_vs_cc for r in usable])
    gaps = np.array([r.tercile_absolute_gap for r in usable])
    dec = [r for r in usable if r.decoupled]
    coup = [r for r in usable if not r.decoupled]

    axes[0].hist(rho, bins=20, color="#b0b0b0", edgecolor="0.35", linewidth=0.5)
    axes[0].axvline(0, color="0.2", linewidth=0.8, linestyle="--")
    axes[0].axvline(-0.3, color=colors[1], linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("ρ(reliability_score, windowed half-map CC) at Cα")
    axes[0].set_ylabel("Maps")
    axes[0].set_title(f"Cohort ρ distribution (n = {len(usable)})")

    if coup:
        axes[1].scatter(
            [r.frac_omit_zone for r in coup],
            [r.frac_cc_below_0_5 for r in coup],
            s=28,
            c="#c8c8c8",
            edgecolors="none",
            alpha=0.85,
            label="coupled",
        )
    for i, r in enumerate(dec):
        axes[1].scatter(
            r.frac_omit_zone,
            r.frac_cc_below_0_5,
            s=52,
            c=colors[i % len(colors)],
            edgecolors="0.15",
            linewidths=0.4,
            zorder=3,
            label=f"EMD-{r.emdb_id}",
        )
    lims = [0, max(0.45, float(max(r.frac_omit_zone for r in usable) * 1.05))]
    axes[1].plot(lims, lims, "k--", linewidth=0.7, alpha=0.5)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel("Fraction Cα in omit tercile")
    axes[1].set_ylabel("Fraction Cα with CC < 0.5")
    axes[1].set_title("Tercile vs absolute low-CC placement")
    if dec:
        axes[1].legend(loc="upper left", fontsize=5, frameon=False, ncol=2)

    n_dec = sum(1 for r in usable if r.decoupled)
    n_inv = sum(1 for r in usable if r.zone_cc_inverted)
    n_strong = sum(1 for r in usable if r.rho_rel_vs_cc <= -0.3)
    fig.suptitle(
        f"Placement decoupling — {n_dec}/{len(usable)} maps ρ<0; "
        f"{n_strong} strong (≤−0.3); {n_inv} zone CC inverted",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)


def _write_supplements(decoupled_ids: list[str], *, manifest: Path, out_dir: Path, dpi: int) -> None:
    for emd_id in decoupled_ids:
        row = load_cohort_manifest_row(manifest, emd_id)
        rv = lh_map_reliability_dir(emd_id) / "residue_validation.csv"
        if not rv.is_file():
            continue
        rows = read_residue_validation_csv(rv)
        out_path = out_dir / f"placement_supplement_emd_{emd_id}.png"
        plot_placement_supplement(
            rows,
            emdb_id=emd_id,
            display_name=str(row.get("display_name", "")),
            out_path=out_path,
            dpi=dpi,
        )
        print(f"[decouple] supplement {out_path.name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.controls_only:
        print("[decouple] loading cohort...", flush=True)
        rows = load_decoupling_cohort(manifest=args.manifest)
        print(f"[decouple] enriching T-rank for {len(rows)} maps...", flush=True)
        rows = _enrich_t_rank_from_npz(rows, manifest=args.manifest)
        csv_path = out_dir / "placement_decoupling_cohort.csv"
        write_decoupling_csv(csv_path, rows)
        _plot_cohort(rows, out_dir / "placement_decoupling_cohort.png", dpi=args.dpi)
        print(f"[decouple] wrote {csv_path}", flush=True)
    else:
        csv_path = out_dir / "placement_decoupling_cohort.csv"
        rows = []
        if csv_path.is_file():
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    rows.append(
                        PlacementDecouplingRow(
                            emdb_id=row["emdb_id"],
                            display_name=row.get("display_name", ""),
                            global_resolution_a=float(row.get("global_resolution_a") or "nan"),
                            n_in_mask=int(row["n_in_mask"]),
                            frac_in_contour_mask=float(row["frac_in_contour_mask"]),
                            rho_rel_vs_cc=float(row["rho_rel_vs_cc"]),
                            rho_t_rank_vs_cc=float(row.get("rho_t_rank_vs_cc") or "nan"),
                            rho_h_raw_vs_cc=float(row.get("rho_h_raw_vs_cc") or "nan"),
                            median_cc_omit=float(row["median_cc_omit"]),
                            median_cc_build=float(row["median_cc_build"]),
                            zone_cc_inverted=bool(int(row["zone_cc_inverted"])),
                            frac_omit_zone=float(row["frac_omit_zone"]),
                            frac_cc_below_0_5=float(row["frac_cc_below_0_5"]),
                            tercile_absolute_gap=float(row["tercile_absolute_gap"]),
                            permutation_p=float(row["permutation_p"]),
                            decoupled=bool(int(row["decoupled"])),
                            notes=row.get("notes", ""),
                        )
                    )

    decoupled_ids = [r.emdb_id for r in rows if r.decoupled]
    if not decoupled_ids and (out_dir / "model_placement_audit.csv").is_file():
        with (out_dir / "model_placement_audit.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get("spearman_reliability_vs_cc", "").strip()
                rho = float(raw) if raw else float("nan")
                if np.isfinite(rho) and rho < 0:
                    decoupled_ids.append(row["emdb_id"])

    print(f"[decouple] running controls on {len(decoupled_ids)} decoupled + positives...", flush=True)
    controls = _build_controls(decoupled_ids, manifest=args.manifest, rho_threshold=args.rho_threshold)
    ctrl_path = out_dir / "placement_decoupling_controls.csv"
    _write_controls_csv(ctrl_path, controls)
    (out_dir / "placement_decoupling_controls.json").write_text(json.dumps(controls, indent=2) + "\n")

    _write_supplements(decoupled_ids, manifest=args.manifest, out_dir=out_dir, dpi=args.dpi)
    from cryoem_mrc.repo_paths import sync_thesis_narrative_cohort_figures

    sync_thesis_narrative_cohort_figures(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
