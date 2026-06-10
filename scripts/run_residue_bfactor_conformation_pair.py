"""Compare ΔB vs Δreliability across two conformations (matched Cα, separate maps/models).

Each state uses its own deposited map, contour, reliability.npz, and fitted PDB.
Residues are matched by mmCIF (label_asym_id, label_seq_id, insertion); domain
bands use auth chain/seq from the deposited model. Δ statistics use per-map
coordinates (no superposition).

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
    get_domain_assignments,
    get_domain_regions_for_pair,
    kabsch_align_coords,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, bfactor_conformation_pairs_dir
from cryoem_mrc.structure_validation import (
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
    DEFAULT_CLUSTER_SEPARATION_THRESHOLD,
    compute_conformation_coupling,
    compute_coupling_cluster_separation_score,
    compute_domain_coupling_block_colors,
    plot_conformation_pair_delta_reliability_supplement,
    plot_conformation_pair_domain_coupling_supplement,
    plot_conformation_pair_summary_triptych,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-a", type=str, required=True)
    p.add_argument("--emd-b", type=str, required=True)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=bfactor_conformation_pairs_dir())
    p.add_argument("--window-radius", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--layout",
        choices=("auto", "block", "domain"),
        default="auto",
        help="Main figure layout: auto from cluster separation score, or force block/domain",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=None,
        help="Block vs domain threshold (default: cryoem_mrc DEFAULT_CLUSTER_SEPARATION_THRESHOLD)",
    )
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

    use = [(a, b) for a, b in pairs if a.in_contour_mask and b.in_contour_mask]
    cluster_sep_score = float("nan")
    figure_layout = "block"
    recommended_layout = "domain"
    cluster_threshold = (
        args.cluster_threshold
        if args.cluster_threshold is not None
        else DEFAULT_CLUSTER_SEPARATION_THRESHOLD
    )

    if len(use) >= 10:
        rho = pair_stats.spearman_delta_b_vs_delta_reliability
        cov_note = _coverage_note(coverage)

        coupling_data = compute_conformation_coupling(pairs, in_mask_both=True)
        if coupling_data is not None:
            cluster_sep_score, _, _ = compute_coupling_cluster_separation_score(
                coupling_data["interior_corr"]
            )

        if coupling_data is not None:
            use_full = coupling_data["use"]
            coords_a = np.array([[a.x, a.y, a.z] for a, _ in use_full], dtype=np.float64)
            coords_b = np.array([[b.x, b.y, b.z] for _, b in use_full], dtype=np.float64)
            coords_b_aligned, _ = kabsch_align_coords(coords_b, coords_a)

            has_domains = bool(get_domain_regions_for_pair(args.emd_a, args.emd_b))
            summary_name = (
                "conformation_pair_summary.png"
                if has_domains
                else "conformation_pair_summary_triptych.png"
            )

            chimerax_domain_png = None
            chimerax_coupling_png = None
            if has_domains:
                from cryoem_mrc.chimerax_figures import (
                    chimerax_render_png,
                    render_chimerax_domain_colored_surface,
                )

                chimerax_domain_png = chimerax_render_png(args.emd_a, "domain")
                regions = get_domain_regions_for_pair(args.emd_a, args.emd_b)
                domain_order = [reg.name for reg in regions]
                use_int = coupling_data["interior_use"]
                assignments = get_domain_assignments(use_int, regions)
                block_hex, _ = compute_domain_coupling_block_colors(
                    coupling_data["interior_corr"], assignments, domain_order
                )
                chimerax_coupling_png = render_chimerax_domain_colored_surface(
                    args.emd_a,
                    domain_colors=block_hex,
                    out_png=out_dir / f"chimerax_emd_{args.emd_a}_domain_coupling.png",
                    preview=True,
                )

            triptych, recommended_layout = plot_conformation_pair_summary_triptych(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                spearman_rho=rho,
                spearman_rho_h=pair_stats.spearman_delta_b_vs_delta_H_repro,
                coverage_note=cov_note,
                coords_b_aligned=coords_b_aligned,
                cluster_separation_threshold=cluster_threshold,
                layout=args.layout,
                manifest=args.manifest,
                include_structure_panel=has_domains,
                chimerax_domain_png=chimerax_domain_png,
                chimerax_coupling_png=chimerax_coupling_png,
                save_path=out_dir / summary_name,
                dpi=args.dpi,
            )
            if triptych is not None:
                plt.close(triptych)

            if has_domains:
                delta_supp = plot_conformation_pair_delta_reliability_supplement(
                    pairs,
                    emdb_a=args.emd_a,
                    emdb_b=args.emd_b,
                    in_mask_both=True,
                    coverage_note=cov_note,
                    manifest=args.manifest,
                    save_path=out_dir / "conformation_pair_delta_reliability_supplement.png",
                    dpi=args.dpi,
                )
                if delta_supp is not None:
                    plt.close(delta_supp)

            supplement = plot_conformation_pair_domain_coupling_supplement(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                coverage_note=cov_note,
                manifest=args.manifest,
                save_path=out_dir / "conformation_pair_domain_coupling_supplement.png",
                dpi=args.dpi,
            )
            if supplement is not None:
                plt.close(supplement)

    (out_dir / "conformation_pair_stats.json").write_text(
        json.dumps(
            {
                "emdb_a": pair_stats.emdb_a,
                "emdb_b": pair_stats.emdb_b,
                "n_matched": pair_stats.n_matched,
                "n_matched_in_mask_both": pair_stats.n_matched_in_mask_both,
                "spearman_delta_b_vs_delta_reliability": pair_stats.spearman_delta_b_vs_delta_reliability,
                "spearman_delta_b_vs_delta_H_repro": pair_stats.spearman_delta_b_vs_delta_H_repro,
                "cluster_separation_score": cluster_sep_score,
                "cluster_separation_threshold": cluster_threshold,
                "figure_layout": figure_layout,
                "recommended_layout": recommended_layout,
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

    flag = " [COVERAGE FLAG]" if coverage.coverage_flag else ""
    layout_txt = (
        f" main=cluster_matrix recommended={recommended_layout} "
        f"coupling_block_score={cluster_sep_score:+.3f}"
        if len(use) >= 10
        else ""
    )
    print(
        f"[conformation_pair] matched={pair_stats.n_matched} in-mask={pair_stats.n_matched_in_mask_both} "
        f"ρ(ΔB,Δrel)={pair_stats.spearman_delta_b_vs_delta_reliability:+.3f} "
        f"missing A={coverage.missing_pct_a:.1f}% B={coverage.missing_pct_b:.1f}%{layout_txt}{flag}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
