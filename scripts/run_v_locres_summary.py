"""Export the per-structure V-metric vs BlocRes local-resolution summary.

For every cohort entry that enters the ``figures/v_vs_locres.py`` panel (a PDB plus
>=30 finite V and >=30 finite local-resolution values in the contour mask) this writes
one row of verified numbers so the thesis text can cite them instead of reading values
off the figure:

- ``rho`` / ``p_value``: in-mask Spearman between V at Cα and local resolution
- ``n_pairs``: paired in-mask residues used for ``rho``
- ``v_variance`` / ``locres_variance``: dynamic range available to each metric
- ``locres_min`` / ``locres_max`` / ``locres_range`` / ``locres_iqr``: resolution spread
- ``nan_reason``: why ``rho`` is NaN, distinguishing "too few pairs" from
  "metric has no variance to correlate against" (the scientifically diagnostic case)

Output: ``outputs/cohort_summary/v_vs_locres_summary.csv`` plus a printed median.

Example::

    source .venv/bin/activate
    python scripts/run_v_locres_summary.py --resume
    python scripts/run_v_locres_summary.py --emd-id 52525
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cryoem_mrc.metric_comparison import load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "v_vs_locres_summary.csv"
MIN_FINITE = 30  # matches figures/v_vs_locres.py panel-eligibility threshold
MIN_PAIRS = 10  # matches figures/v_vs_locres.py _spearman_in_mask floor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out", type=Path, default=OUTPUT_CSV)
    p.add_argument(
        "--emd-id",
        type=str,
        default=None,
        help="Process one EMDB entry only (for spot checks)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip EMDB IDs already present in the output CSV",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-residue local-resolution aggregation warnings",
    )
    return p.parse_args(argv)


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not verbose:
        logging.getLogger("cryoem_mrc.local_resolution").setLevel(logging.ERROR)


SUMMARY_COLUMNS = [
    "emdb_id",
    "display_name",
    "rho",
    "p_value",
    "n_pairs",
    "n_v_in_mask",
    "n_locres_in_mask",
    "v_variance",
    "locres_variance",
    "locres_min",
    "locres_max",
    "locres_range",
    "locres_iqr",
    "nan_reason",
]


def _load_existing_rows(out: Path) -> dict[str, dict]:
    if not out.is_file():
        return {}
    df = pd.read_csv(out)
    return {str(row["emdb_id"]).strip(): row.to_dict() for _, row in df.iterrows()}


def _write_rows(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values("rho", na_position="last").to_csv(
        out, index=False
    )


def _display_names(manifest: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                names[eid] = row.get("display_name", "").strip()
    return names


def _summarize_entry(emdb_id: str, *, manifest: Path) -> dict | None:
    """One summary row, or ``None`` if the entry never enters the figure panel."""
    try:
        df = load_all_metrics(emdb_id, manifest=manifest)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("skip EMD-%s: %s", emdb_id, exc)
        return None

    sub = df[df["in_contour_mask"].astype(bool)]
    n_v = int(sub["v_metric"].notna().sum())
    n_loc = int(sub["local_resolution"].notna().sum())
    # Panel eligibility: same gate as figures/v_vs_locres.py._entries_with_locres.
    if n_v < MIN_FINITE or n_loc < MIN_FINITE:
        return None

    paired = sub["v_metric"].notna() & sub["local_resolution"].notna()
    n_pairs = int(paired.sum())
    v = sub.loc[paired, "v_metric"].to_numpy(dtype=float)
    loc = sub.loc[paired, "local_resolution"].to_numpy(dtype=float)

    v_var = float(np.var(v)) if n_pairs else float("nan")
    loc_var = float(np.var(loc)) if n_pairs else float("nan")

    rho = float("nan")
    pval = float("nan")
    nan_reason = ""
    if n_pairs < MIN_PAIRS:
        nan_reason = f"n_pairs<{MIN_PAIRS}"
    elif v_var == 0.0 and loc_var == 0.0:
        nan_reason = "zero_v_and_locres_variance"
    elif v_var == 0.0:
        nan_reason = "zero_v_variance"
    elif loc_var == 0.0:
        nan_reason = "zero_locres_variance"
    else:
        r, p = stats.spearmanr(v, loc)
        rho, pval = float(r), float(p)
        if not np.isfinite(rho):
            nan_reason = "spearman_undefined"

    if n_pairs:
        loc_min = float(np.min(loc))
        loc_max = float(np.max(loc))
        loc_iqr = float(np.subtract(*np.percentile(loc, [75, 25])))
    else:
        loc_min = loc_max = loc_iqr = float("nan")

    return {
        "emdb_id": emdb_id,
        "rho": rho,
        "p_value": pval,
        "n_pairs": n_pairs,
        "n_v_in_mask": n_v,
        "n_locres_in_mask": n_loc,
        "v_variance": v_var,
        "locres_variance": loc_var,
        "locres_min": loc_min,
        "locres_max": loc_max,
        "locres_range": (loc_max - loc_min) if n_pairs else float("nan"),
        "locres_iqr": loc_iqr,
        "nan_reason": nan_reason,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)

    names = _display_names(args.manifest)
    existing = _load_existing_rows(args.out) if args.resume else {}
    rows: list[dict] = list(existing.values())

    manifest_rows: list[tuple[str, str]] = []
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            emdb_id = str(row.get("emdb_id", "")).strip()
            pdb = row.get("flexibility_path_or_pdb", "").strip()
            if not emdb_id or not pdb or not Path(pdb).is_file():
                continue
            if args.emd_id and emdb_id != args.emd_id.strip():
                continue
            manifest_rows.append((emdb_id, pdb))

    for emdb_id, _pdb in manifest_rows:
        if args.resume and emdb_id in existing:
            print(f"[v_locres] EMD-{emdb_id}: resume skip", flush=True)
            continue
        summary = _summarize_entry(emdb_id, manifest=args.manifest)
        if summary is None:
            continue
        summary["display_name"] = names.get(emdb_id, "")
        rows = [r for r in rows if str(r["emdb_id"]).strip() != emdb_id]
        rows.append(summary)
        _write_rows(args.out, rows)
        print(
            f"[v_locres] EMD-{emdb_id}: rho={summary['rho']:+.3f} "
            f"n={summary['n_pairs']} v_var={summary['v_variance']:.3g} "
            f"locres_range={summary['locres_range']:.2f} "
            f"{summary['nan_reason']}",
            flush=True,
        )

    if not rows:
        print("[v_locres] no eligible entries", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values("rho", na_position="last")
    finite = out_df["rho"].dropna()
    median = float(finite.median()) if len(finite) else float("nan")
    print(
        f"[v_locres] {len(out_df)} eligible structures, {len(finite)} with finite rho, "
        f"{len(out_df) - len(finite)} NaN -> median rho = {median:+.3f}",
        flush=True,
    )
    print(f"[v_locres] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
