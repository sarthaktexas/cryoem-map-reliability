"""Compare ΔB vs Δreliability across two conformations (matched Cα, separate maps/models).

Each state uses its own deposited map, contour, reliability.npz, and fitted PDB.
Residues are matched by (chain, seq_num, seq_icode). Δ statistics use per-map
coordinates (no superposition). Kabsch alignment is for visualization / ChimeraX only.

Example (when both maps are processed)::

    python scripts/run_residue_bfactor_conformation_pair.py --emd-a 23129 --emd-b 23130
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

from cryoem_mrc.conformation_pair import (
    compute_conformation_pair_coverage,
    kabsch_align_coords,
    residue_key_to_coupling_map,
    write_aligned_ca_pdb,
    write_chimerax_coupling_script,
    write_coupling_colored_pdb,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, bfactor_conformation_pairs_dir
from cryoem_mrc.structure_validation import (
    CaResidue,
    compute_conformation_pair_stats,
    default_reliability_out_dir,
    iter_ca_residues,
    load_cohort_manifest_row,
    match_residue_rows_by_key,
    read_residue_validation_csv,
    run_emdb_bfactor_validation,
    write_conformation_pair_md,
)
from cryoem_mrc.thesis_figures import (
    compute_conformation_coupling,
    plot_conformation_delta_joint_heatmap,
    plot_conformation_pair_coupling_heatmap,
    plot_conformation_pair_summary_triptych,
    plot_conformation_sequence_strip,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-a", type=str, required=True)
    p.add_argument("--emd-b", type=str, required=True)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=bfactor_conformation_pairs_dir())
    p.add_argument("--window-radius", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


def _coverage_note(coverage) -> str:
    if coverage.coverage_flag:
        return (
            f"coverage A {coverage.missing_pct_a:.0f}% / B {coverage.missing_pct_b:.0f}% missing"
        )
    return (
        f"coverage OK (A {100 * coverage.frac_analysis_of_a:.0f}%, "
        f"B {100 * coverage.frac_analysis_of_b:.0f}% of deposited Cα)"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pair_name = f"emd_{args.emd_a}_vs_{args.emd_b}"
    out_dir = args.out_dir / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)
    chimerax_dir = out_dir / "chimerax"
    chimerax_dir.mkdir(parents=True, exist_ok=True)

    row_a = load_cohort_manifest_row(args.manifest, args.emd_a)
    row_b = load_cohort_manifest_row(args.manifest, args.emd_b)
    pdb_a = Path(row_a["flexibility_path_or_pdb"])
    pdb_b = Path(row_b["flexibility_path_or_pdb"])

    for emd_id in (args.emd_a, args.emd_b):
        try:
            code, _, stats, _ = run_emdb_bfactor_validation(
                emd_id,
                manifest=args.manifest,
                window_radius=args.window_radius,
                require_b_factor_source=False,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[conformation_pair] ERROR: {e}", file=sys.stderr)
            return 2
        if code != 0 or stats is None:
            return 2

    dir_a = default_reliability_out_dir(args.emd_a)
    dir_b = default_reliability_out_dir(args.emd_b)
    csv_a = dir_a / "residue_validation.csv"
    csv_b = dir_b / "residue_validation.csv"
    if not csv_a.exists() or not csv_b.exists():
        print("[conformation_pair] ERROR: missing residue_validation.csv", file=sys.stderr)
        return 2

    rows_a = read_residue_validation_csv(csv_a)
    rows_b = read_residue_validation_csv(csv_b)
    pairs = match_residue_rows_by_key(rows_a, rows_b)
    pair_stats = compute_conformation_pair_stats(
        pairs, emdb_a=args.emd_a, emdb_b=args.emd_b, in_mask_both=True
    )
    coverage = compute_conformation_pair_coverage(
        pairs,
        emdb_a=args.emd_a,
        emdb_b=args.emd_b,
        n_ca_total_a=len(iter_ca_residues(pdb_a)),
        n_ca_total_b=len(iter_ca_residues(pdb_b)),
    )
    write_conformation_pair_md(out_dir / "CONFORMATION_PAIR.md", pair_stats, coverage=coverage)
    (out_dir / "conformation_pair_stats.json").write_text(
        json.dumps(
            {
                "emdb_a": pair_stats.emdb_a,
                "emdb_b": pair_stats.emdb_b,
                "n_matched": pair_stats.n_matched,
                "n_matched_in_mask_both": pair_stats.n_matched_in_mask_both,
                "spearman_delta_b_vs_delta_reliability": pair_stats.spearman_delta_b_vs_delta_reliability,
                "spearman_delta_b_vs_delta_H_repro": pair_stats.spearman_delta_b_vs_delta_H_repro,
                "n_ca_total_a": coverage.n_ca_total_a,
                "n_ca_total_b": coverage.n_ca_total_b,
                "frac_analysis_of_a": coverage.frac_analysis_of_a,
                "frac_analysis_of_b": coverage.frac_analysis_of_b,
                "missing_pct_a": coverage.missing_pct_a,
                "missing_pct_b": coverage.missing_pct_b,
                "coverage_flag": coverage.coverage_flag,
            },
            indent=2,
        )
        + "\n"
    )

    use = [(a, b) for a, b in pairs if a.in_contour_mask and b.in_contour_mask]
    if len(use) >= 10:
        db = np.array([b.b_iso - a.b_iso for a, b in use])
        drel = np.array([b.reliability_score - a.reliability_score for a, b in use])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(db, drel, s=8, alpha=0.35, c="#9467bd", edgecolors="none")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel(f"ΔB_iso ({args.emd_b} − {args.emd_a})")
        ax.set_ylabel(f"Δreliability_score ({args.emd_b} − {args.emd_a})")
        ax.set_title(f"Conformation pair: matched in-mask Cα (n={len(use)})")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_b_vs_delta_reliability.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        rho = pair_stats.spearman_delta_b_vs_delta_reliability
        cov_note = _coverage_note(coverage)

        strip = plot_conformation_sequence_strip(
            pairs,
            emdb_a=args.emd_a,
            emdb_b=args.emd_b,
            in_mask_both=True,
            save_path=out_dir / "conformation_sequence_strip.png",
            dpi=args.dpi,
        )
        if strip is not None:
            plt.close(strip)

        coupling = plot_conformation_pair_coupling_heatmap(
            pairs,
            emdb_a=args.emd_a,
            emdb_b=args.emd_b,
            in_mask_both=True,
            spearman_rho=rho,
            save_path=out_dir / "conformation_coupling_heatmap.png",
            dpi=args.dpi,
        )
        if coupling is not None:
            plt.close(coupling)

        joint = plot_conformation_delta_joint_heatmap(
            pairs,
            emdb_a=args.emd_a,
            emdb_b=args.emd_b,
            in_mask_both=True,
            spearman_rho=rho,
            save_path=out_dir / "conformation_delta_joint_heatmap.png",
            dpi=args.dpi,
        )
        if joint is not None:
            plt.close(joint)

        coupling_data = compute_conformation_coupling(pairs, in_mask_both=True)
        coords_b_aligned = None
        if coupling_data is not None:
            use_full = coupling_data["use"]
            row_mean = np.asarray(coupling_data["row_mean_abs"], dtype=np.float64)
            coords_a = np.array([[a.x, a.y, a.z] for a, _ in use_full], dtype=np.float64)
            coords_b = np.array([[b.x, b.y, b.z] for _, b in use_full], dtype=np.float64)
            coords_b_aligned, _ = kabsch_align_coords(coords_b, coords_a)

            interior_use = coupling_data["interior_use"]
            idx = coupling_data["interior_indices"]
            coords_b_int = coords_b_aligned[idx]

            triptych = plot_conformation_pair_summary_triptych(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                spearman_rho=rho,
                coverage_note=cov_note,
                coords_b_aligned=coords_b_int,
                save_path=out_dir / "conformation_pair_summary_triptych.png",
                dpi=args.dpi,
            )
            if triptych is not None:
                plt.close(triptych)

            coupling_map = residue_key_to_coupling_map(use_full, row_mean)
            colored_pdb = write_coupling_colored_pdb(
                pdb_a,
                chimerax_dir / f"emd_{args.emd_a}_coupling_colored.pdb",
                coupling_map,
            )
            aligned_res = [
                CaResidue(
                    chain=a.chain,
                    seq_num=a.seq_num,
                    seq_icode=a.seq_icode,
                    res_name=a.res_name,
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                    b_iso=0.0,
                )
                for (a, _), xyz in zip(use_full, coords_b_aligned)
            ]
            aligned_pdb = write_aligned_ca_pdb(
                aligned_res,
                coords_b_aligned,
                chimerax_dir / f"emd_{args.emd_b}_aligned_to_{args.emd_a}.pdb",
            )
            write_chimerax_coupling_script(
                colored_pdb_a=colored_pdb,
                aligned_pdb_b=aligned_pdb,
                session_path=chimerax_dir / "coupling_view.cxc",
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
            )
        else:
            triptych = plot_conformation_pair_summary_triptych(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                spearman_rho=rho,
                coverage_note=cov_note,
                save_path=out_dir / "conformation_pair_summary_triptych.png",
                dpi=args.dpi,
            )
            if triptych is not None:
                plt.close(triptych)

    flag = " [COVERAGE FLAG]" if coverage.coverage_flag else ""
    print(
        f"[conformation_pair] matched={pair_stats.n_matched} in-mask={pair_stats.n_matched_in_mask_both} "
        f"ρ(ΔB,Δrel)={pair_stats.spearman_delta_b_vs_delta_reliability:+.3f} "
        f"missing A={coverage.missing_pct_a:.1f}% B={coverage.missing_pct_b:.1f}%{flag}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
