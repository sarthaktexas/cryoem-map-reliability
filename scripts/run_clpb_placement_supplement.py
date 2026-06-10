"""Per-residue placement supplement for ClpB WT-2A (reviewer pushback figure).

Reads ``residue_validation.csv`` and writes a three-panel figure showing
CC and B_iso stratified by build zone plus reliability vs CC at Cα.

Example::

    source .venv/bin/activate
    python scripts/run_clpb_placement_supplement.py
    python scripts/run_clpb_placement_supplement.py --emd-id 4940
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from cryoem_mrc.placement_supplement import plot_placement_supplement
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, sync_thesis_doc_figure
from cryoem_mrc.structure_validation import (
    default_reliability_out_dir,
    load_cohort_manifest_row,
    read_residue_validation_csv,
    run_emdb_bfactor_validation,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default="4941", help="Primary map (default ClpB WT-2A)")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--cohort-copy",
        type=Path,
        default=OUTPUTS_ROOT / "cohort_summary" / "clpb_wt2a_placement_supplement.png",
        help="Also write a cohort_summary copy for thesis embedding",
    )
    p.add_argument(
        "--run-validation",
        action="store_true",
        help="Generate missing residue_validation.csv via run_emdb_bfactor_validation",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _load_rows(emd_id: str, *, manifest: Path, run_validation: bool):
    csv_path = default_reliability_out_dir(emd_id) / "residue_validation.csv"
    if csv_path.is_file():
        return read_residue_validation_csv(csv_path), csv_path

    if not run_validation:
        raise FileNotFoundError(
            f"EMD-{emd_id}: missing {csv_path} (pass --run-validation to generate)"
        )

    _, rows, _, _ = run_emdb_bfactor_validation(
        emd_id,
        manifest=manifest,
        require_b_factor_source=False,
    )
    if not rows:
        raise ValueError(f"EMD-{emd_id}: validation produced no rows")
    return rows, csv_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    emd_id = str(args.emd_id).strip()
    try:
        manifest_row = load_cohort_manifest_row(args.manifest, emd_id)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        rows, csv_path = _load_rows(emd_id, manifest=args.manifest, run_validation=args.run_validation)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[placement_supplement] ERROR: {exc}", file=sys.stderr)
        return 2

    display_name = str(manifest_row.get("display_name", "")).strip()
    out_dir = default_reliability_out_dir(emd_id) / "figures"
    stem = "clpb_wt2a_placement_supplement" if emd_id == "4941" else f"emd_{emd_id}_placement_supplement"
    out_path = out_dir / f"{stem}.png"

    stats = plot_placement_supplement(
        rows,
        emdb_id=emd_id,
        display_name=display_name,
        out_path=out_path,
        n_residues=len(rows),
        dpi=args.dpi,
    )

    if args.cohort_copy and emd_id == "4941":
        args.cohort_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, args.cohort_copy)
        pdf_src = out_path.with_suffix(".pdf")
        if pdf_src.is_file():
            shutil.copy2(pdf_src, args.cohort_copy.with_suffix(".pdf"))
        sync_thesis_doc_figure(args.cohort_copy, "fig_s4_clpb_wt2a_placement_supplement.png")

    stats_path = out_dir / f"{stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(
        f"[placement_supplement] EMD-{emd_id}: n={int(stats['n_in_mask'])}, "
        f"ρ(rel,CC)={stats['spearman_reliability_vs_cc']:+.2f}, "
        f"median CC omit/build={stats['median_cc_omit']:.2f}/{stats['median_cc_build']:.2f}",
        flush=True,
    )
    print(f"[placement_supplement] wrote {out_path}", flush=True)
    print(f"[placement_supplement] source {csv_path}", flush=True)
    if args.cohort_copy and emd_id == "4941":
        print(f"[placement_supplement] wrote {args.cohort_copy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
