"""Per-residue Q-score vs LH constraint V validation (3dem/qscore + cohort manifest).

Requires ``qscore`` from https://github.com/3dem/qscore (plus biopython, tqdm)::

    source .venv/bin/activate
    uv pip install "git+https://github.com/3dem/qscore.git" biopython tqdm

Example::

    # EMD-49450 anchor
    python scripts/run_qscore_validation.py --emd-id 49450

    # Thesis anchor maps
    python scripts/run_qscore_validation.py --anchors
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

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.qscore_validation import (
    QscoreResidueRow,
    QscoreValidationStats,
    run_emdb_qscore_validation,
)
from cryoem_mrc.repo_paths import ANCHOR_EMDB_ID, BFACTOR_VALIDATION_EMDB_IDS, COHORT_MANIFEST, OUTPUTS_ROOT

# Omit from cohort ρ figure / headline stats (degenerate V or no Cα anchor).
# See outputs/cohort_summary/QSCORE_COHORT_SUMMARY.md Finding 2.
QSCORE_PANEL_EXCLUDE = frozenset({"33736", "52525"})

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument(
        "--anchors",
        action="store_true",
        help=f"Run thesis anchor IDs: {', '.join(BFACTOR_VALIDATION_EMDB_IDS)}",
    )
    p.add_argument("--all", action="store_true", help="All manifest rows with a local deposited PDB")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--reliability-npz", type=Path, default=None)
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--pdb", type=Path, default=None)
    p.add_argument("--contour", type=float, default=None)
    p.add_argument("--window-radius", type=int, default=0)
    p.add_argument("--num-points", type=int, default=8, help="Q-score radial samples per radius")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--cohort-summary",
        action="store_true",
        help="Write outputs/cohort_summary/qscore_correlations.csv after batch run",
    )
    p.add_argument(
        "--cohort-figure",
        action="store_true",
        help="Build cohort summary figure from existing qscore_correlations.csv (standalone-capable)",
    )
    return p.parse_args(argv)


def _plot_qscore_vs_v(
    rows: list[QscoreResidueRow],
    stats: QscoreValidationStats,
    out: Path,
    *,
    emd_id: str,
    dpi: int,
) -> None:
    use = [
        r
        for r in rows
        if r.in_contour_mask and np.isfinite(r.q_score) and np.isfinite(r.reliability_constraint_V_rank)
    ]
    if not use:
        return

    q = np.array([r.q_score for r in use])
    v_rank = np.array([r.reliability_constraint_V_rank for r in use])
    b = np.array([r.b_iso for r in use])
    rho = stats.spearman_q_vs_V

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    apply(ax)
    scatter = ax.scatter(
        q,
        v_rank,
        s=10,
        alpha=0.4,
        c=b,
        cmap="viridis",
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Deposited B_iso", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel("Per-residue Q-score")
    ax.set_ylabel("Constraint V (in-mask percentile rank)")
    rho_txt = f", ρ={rho:+.3f}" if np.isfinite(rho) else ""
    ax.set_title(f"Q-score vs V — EMD-{emd_id} (n={len(use)}{rho_txt})")
    label_panel(ax, "a")
    fig.tight_layout()
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)


def _emd_ids_for_all(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            src = row.get("flexibility_source", "").strip()
            if src in ("excluded", "skip", ""):
                continue
            pdb = Path(row.get("flexibility_path_or_pdb", ""))
            if not pdb.is_file():
                print(f"[qscore_validation] skip EMD-{row['emdb_id']}: no PDB {pdb}", flush=True)
                continue
            ids.append(str(row["emdb_id"]).strip())
    return ids


def _resolution_by_id(manifest: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[str(row["emdb_id"]).strip()] = float(row["global_resolution_a"])
            except (KeyError, ValueError):
                continue
    return out


def _name_by_id(manifest: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["emdb_id"]).strip()] = row.get("display_name", "").strip()
    return out


def _build_cohort_figure(manifest: Path, dpi: int) -> Path | None:
    """Two-panel cohort summary: per-structure ρ ranking and ρ vs resolution."""
    from scipy import stats as _stats

    csv_path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if not csv_path.is_file():
        print(f"[qscore_validation] no cohort CSV at {csv_path}", file=sys.stderr)
        return None

    res_by_id = _resolution_by_id(manifest)
    name_by_id = _name_by_id(manifest)
    rows = list(csv.DictReader(csv_path.open()))

    recs = []
    for r in rows:
        raw = r.get("spearman_q_vs_V", "")
        if raw in ("", "nan"):
            continue
        rho = float(raw)
        if not np.isfinite(rho):
            continue
        eid = str(r["emdb_id"]).strip()
        if eid in QSCORE_PANEL_EXCLUDE:
            continue
        recs.append(
            {
                "emdb_id": eid,
                "rho": rho,
                "n": int(r["n_in_mask"]),
                "res": res_by_id.get(eid, float("nan")),
                "name": name_by_id.get(eid, eid),
            }
        )
    if not recs:
        print("[qscore_validation] no finite ρ rows for figure", file=sys.stderr)
        return None

    recs.sort(key=lambda d: d["rho"])
    rhos = np.array([d["rho"] for d in recs])
    res = np.array([d["res"] for d in recs])
    labels = [f"EMD-{d['emdb_id']}" for d in recs]
    median_rho = float(np.median(rhos))

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(11.0, 6.5))

    apply(ax_bar)
    res_finite = res[np.isfinite(res)]
    vmin = float(res_finite.min()) if res_finite.size else 0.0
    vmax = float(res_finite.max()) if res_finite.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(norm(v)) if np.isfinite(v) else "0.6" for v in res]
    ypos = np.arange(len(recs))
    ax_bar.barh(ypos, rhos, color=colors, edgecolor="0.2", linewidth=0.4)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=5)
    ax_bar.axvline(0.0, color="0.3", linewidth=0.6)
    ax_bar.axvline(median_rho, color=PALETTES["categorical"][1], linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Spearman ρ(Q-score, V), in-mask Cα")
    ax_bar.set_title(f"Per-structure Q-score vs V (median ρ={median_rho:+.2f})")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, fraction=0.046, pad=0.02)
    cbar.set_label("Global resolution (Å)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    label_panel(ax_bar, "a")

    apply(ax_sc)
    m = np.isfinite(res)
    ax_sc.scatter(res[m], rhos[m], s=24, c=PALETTES["categorical"][0], edgecolors="0.2", linewidths=0.4)
    if m.sum() >= 3:
        rho_rr = _stats.spearmanr(res[m], rhos[m]).statistic
        coef = np.polyfit(res[m], rhos[m], 1)
        xline = np.linspace(res[m].min(), res[m].max(), 50)
        ax_sc.plot(xline, np.polyval(coef, xline), color=PALETTES["categorical"][1], linewidth=0.9)
        ax_sc.set_title(f"ρ(Q,V) vs resolution (Spearman={rho_rr:+.2f})")
    else:
        ax_sc.set_title("ρ(Q,V) vs resolution")
    ax_sc.axhline(0.0, color="0.3", linewidth=0.6)
    ax_sc.set_xlabel("Global resolution (Å)")
    ax_sc.set_ylabel("Spearman ρ(Q-score, V)")
    label_panel(ax_sc, "b")

    fig.suptitle("Q-score vs constraint V — cohort summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTPUTS_ROOT / "cohort_summary" / "qscore_vs_V_cohort"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    print(f"[qscore_validation] cohort figure → {out}.png", flush=True)
    return out


def _write_cohort_summary(records: list[dict]) -> Path:
    out_dir = OUTPUTS_ROOT / "cohort_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qscore_correlations.csv"
    fieldnames = ["emdb_id", "pdb_id", "spearman_q_vs_V", "spearman_q_vs_V_rank", "n_residues", "n_in_mask"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    return path


def _run_one(
    emd_id: str,
    *,
    manifest: Path,
    reliability_npz: Path | None,
    reference: Path | None,
    pdb: Path | None,
    contour: float | None,
    window_radius: int,
    num_points: int,
    dpi: int,
) -> tuple[int, dict | None]:
    try:
        code, rows, stats, out_dir = run_emdb_qscore_validation(
            emd_id,
            manifest=manifest,
            reliability_npz=reliability_npz,
            reference=reference,
            pdb=pdb,
            contour=contour,
            window_radius=window_radius,
            num_points=num_points,
        )
    except FileNotFoundError as e:
        print(f"[qscore_validation] ERROR: {e}", file=sys.stderr)
        return 2, None
    except ImportError as e:
        print(
            "[qscore_validation] ERROR: qscore not installed. "
            'Run: uv pip install "git+https://github.com/3dem/qscore.git" biopython tqdm',
            file=sys.stderr,
        )
        print(f"[qscore_validation] ({e})", file=sys.stderr)
        return 2, None

    if stats is None:
        print(f"[qscore_validation] skip EMD-{emd_id}", flush=True)
        return code, None

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_qscore_vs_v(
        rows,
        stats,
        fig_dir / "qscore_vs_V_scatter.png",
        emd_id=emd_id,
        dpi=dpi,
    )
    (out_dir / "qscore_validation_stats.json").write_text(
        json.dumps(
            {
                "emdb_id": stats.emdb_id,
                "pdb_id": stats.pdb_id,
                "n_residues": stats.n_residues,
                "n_in_mask": stats.n_in_mask,
                "n_with_q_score": stats.n_with_q_score,
                "spearman_q_vs_V": stats.spearman_q_vs_V,
                "spearman_q_vs_V_rank": stats.spearman_q_vs_V_rank,
                "spearman_q_vs_b_iso": stats.spearman_q_vs_b_iso,
                "median_q_by_b_tercile": stats.median_q_by_b_tercile,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[qscore_validation] EMD-{emd_id}: n={stats.n_in_mask} in-mask, "
        f"ρ(Q, V)={stats.spearman_q_vs_V:+.3f}",
        flush=True,
    )
    record = {
        "emdb_id": stats.emdb_id,
        "pdb_id": stats.pdb_id,
        "spearman_q_vs_V": f"{stats.spearman_q_vs_V:.6f}",
        "spearman_q_vs_V_rank": f"{stats.spearman_q_vs_V_rank:.6f}",
        "n_residues": stats.n_in_mask,
        "n_in_mask": stats.n_in_mask,
    }
    return 0, record


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.cohort_figure and not (args.all or args.emd_id or args.anchors):
        out = _build_cohort_figure(args.manifest, args.dpi)
        return 0 if out is not None else 2

    if not args.all and not args.emd_id and not args.anchors:
        print("Specify --emd-id, --anchors, --all, or --cohort-figure", file=sys.stderr)
        return 2

    if args.all:
        ids = _emd_ids_for_all(args.manifest)
    elif args.anchors:
        ids = list(BFACTOR_VALIDATION_EMDB_IDS)
    else:
        ids = [args.emd_id.strip()]

    rc = 0
    records: list[dict] = []
    for emd_id in ids:
        code, record = _run_one(
            emd_id,
            manifest=args.manifest,
            reliability_npz=args.reliability_npz,
            reference=args.reference,
            pdb=args.pdb,
            contour=args.contour,
            window_radius=args.window_radius,
            num_points=args.num_points,
            dpi=args.dpi,
        )
        rc = max(rc, code)
        if record is not None:
            records.append(record)

    if args.cohort_summary and records:
        path = _write_cohort_summary(records)
        print(f"[qscore_validation] cohort summary → {path}", flush=True)

    if args.cohort_figure:
        _build_cohort_figure(args.manifest, args.dpi)

    if rc == 0 and len(ids) == 1 and ids[0] == ANCHOR_EMDB_ID:
        print(f"[qscore_validation] anchor complete — check outputs/emd_{ANCHOR_EMDB_ID}/lh_map_reliability/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
