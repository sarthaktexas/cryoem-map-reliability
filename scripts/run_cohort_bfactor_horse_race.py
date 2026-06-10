"""Cohort table: variance vs V vs H_repro for B-factor prediction (reviewer horse-race).

Writes ``outputs/cohort_summary/bfactor_horse_race.csv`` with per-map Spearman ρ and
partial correlations at Cα (2 Å sphere, in-mask residues).

Example::

    source .venv/bin/activate
    python scripts/run_cohort_bfactor_horse_race.py
    python scripts/run_cohort_bfactor_horse_race.py --figure-only
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
from scipy import stats

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.half_map_repro import load_windowed_halfmap_correlation
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, halfmap_metrics_npz, lh_map_reliability_dir
from cryoem_mrc.structure_validation import (
    _partial_spearman,
    iter_ca_residues,
    load_cohort_manifest_row,
    physical_xyz_to_voxel_indices,
    sample_volume_at_ca,
)
from cryoem_mrc.map_grid import load_map_grid

REPO = Path(__file__).resolve().parents[1]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    return float(stats.spearmanr(x[m], y[m]).statistic)


def _horse_race_one(emd_id: str, *, manifest: Path, sphere_radius_a: float) -> dict | None:
    row = load_cohort_manifest_row(manifest, emd_id)
    if row.get("flexibility_source", "").strip() != "b_factor":
        return None

    ref_path = Path(row["reference_mrc"])
    pdb_path = Path(row["flexibility_path_or_pdb"])
    rel_npz = lh_map_reliability_dir(emd_id) / "reliability.npz"
    if not ref_path.is_file() or not pdb_path.is_file() or not rel_npz.is_file():
        print(f"[horse_race] skip EMD-{emd_id}: missing inputs", file=sys.stderr, flush=True)
        return None

    contour = float(row["contour"])
    grid = load_map_grid(ref_path, dtype=np.float32)
    ref = np.asarray(grid.data, dtype=np.float32)
    mask = build_contour_mask(ref, contour)

    with np.load(rel_npz, allow_pickle=False) as d:
        t = np.asarray(d["reliability_fluctuation"], dtype=np.float32)
        v = np.asarray(d["reliability_smoothness"], dtype=np.float32)
        h = np.asarray(d["reliability_H_repro"], dtype=np.float32)

    from cryoem_mrc.repo_paths import find_features_npz

    feat_path = find_features_npz(ref_path.parent, emd_id, contour)
    if feat_path is None:
        print(f"[horse_race] skip EMD-{emd_id}: no features NPZ", file=sys.stderr, flush=True)
        return None
    with np.load(feat_path, allow_pickle=False) as feat:
        var = np.asarray(feat["local_variance"], dtype=np.float32)

    cc = None
    cc_path = halfmap_metrics_npz(emd_id)
    if cc_path.is_file():
        with np.load(cc_path, allow_pickle=False) as hm:
            cc = load_windowed_halfmap_correlation(hm)

    residues = iter_ca_residues(pdb_path)
    b = np.array([r.b_iso for r in residues], dtype=np.float64)
    v_s = sample_volume_at_ca(v, grid, residues, sphere_radius_a=sphere_radius_a)
    h_s = sample_volume_at_ca(h, grid, residues, sphere_radius_a=sphere_radius_a)
    var_s = sample_volume_at_ca(var, grid, residues, sphere_radius_a=sphere_radius_a)

    in_mask = []
    for res in residues:
        iz, iy, ix = physical_xyz_to_voxel_indices(res.x, res.y, res.z, grid)
        in_mask.append(
            0 <= iz < mask.shape[0]
            and 0 <= iy < mask.shape[1]
            and 0 <= ix < mask.shape[2]
            and bool(mask[iz, iy, ix])
        )
    m = np.array(in_mask, dtype=bool)

    b_m, v_m, h_m, var_m = b[m], v_s[m], h_s[m], var_s[m]
    t_m = None
    if cc is not None:
        t_s = sample_volume_at_ca(t, grid, residues, sphere_radius_a=sphere_radius_a)
        t_m = t_s[m]

    out = {
        "emdb_id": emd_id,
        "n_in_mask": int(m.sum()),
        "rho_b_vs_V": _spearman(b_m, v_m),
        "rho_b_vs_variance": _spearman(b_m, var_m),
        "rho_b_vs_H": _spearman(b_m, h_m),
        "partial_b_vs_V_given_variance": _partial_spearman(b_m, v_m, var_m),
        "partial_b_vs_variance_given_V": _partial_spearman(b_m, var_m, v_m),
        "partial_b_vs_H_given_variance": _partial_spearman(b_m, h_m, var_m),
    }
    if t_m is not None:
        cc_m = sample_volume_at_ca(cc, grid, residues, sphere_radius_a=sphere_radius_a)[m]
        out["rho_cc_vs_T"] = _spearman(t_m, cc_m)
        out["rho_cc_vs_V"] = _spearman(v_m, cc_m)
        out["rho_cc_vs_variance"] = _spearman(var_m, cc_m)

    return out


def _load_horse_race_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rec: dict = {"emdb_id": row["emdb_id"].strip()}
            for key in row:
                if key == "emdb_id":
                    continue
                raw = row[key].strip()
                if raw in ("", "nan"):
                    rec[key] = float("nan")
                else:
                    try:
                        rec[key] = float(raw)
                    except ValueError:
                        rec[key] = raw
            rows.append(rec)
    return rows


def _build_figure(rows: list[dict], out_dir: Path, dpi: int) -> Path:
    """Two-panel horse-race: marginal ρ(B,·) and partial ρ after controlling for variance."""
    usable = [r for r in rows if int(r.get("n_in_mask", 0) or 0) >= 30]
    if len(usable) < 3:
        raise ValueError("Need at least three maps with n_in_mask >= 30 for horse-race figure")

    def _col(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in usable], dtype=np.float64)

    fig, (ax_marg, ax_part) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    apply(ax_marg)
    data_marg = [_col("rho_b_vs_V"), _col("rho_b_vs_variance"), _col("rho_b_vs_H")]
    labels_marg = ["V", "Local variance", "H_repro"]
    positions = np.arange(1, 4)
    bp = ax_marg.boxplot(
        data_marg,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.0},
    )
    colors = PALETTES["categorical"][:3]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.25")
    for i, vals in enumerate(data_marg):
        jitter = (np.random.default_rng(42).random(len(vals)) - 0.5) * 0.12
        ax_marg.scatter(positions[i] + jitter, vals, s=14, c="0.2", alpha=0.45, edgecolors="none")
    ax_marg.axhline(0.0, color="0.35", linewidth=0.6)
    ax_marg.set_xticks(positions)
    ax_marg.set_xticklabels(labels_marg, fontsize=8)
    ax_marg.set_ylabel("Spearman ρ(B_iso, metric)")
    ax_marg.set_title(f"Marginal B-factor coupling (n = {len(usable)} maps)")
    label_panel(ax_marg, "a")

    apply(ax_part)
    data_part = [_col("partial_b_vs_V_given_variance"), _col("partial_b_vs_variance_given_V")]
    positions_p = np.arange(1, 3)
    bp2 = ax_part.boxplot(
        data_part,
        positions=positions_p,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.0},
    )
    colors_p = [PALETTES["categorical"][0], PALETTES["categorical"][2]]
    for patch, color in zip(bp2["boxes"], colors_p):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.25")
    for i, vals in enumerate(data_part):
        jitter = (np.random.default_rng(43).random(len(vals)) - 0.5) * 0.12
        ax_part.scatter(positions_p[i] + jitter, vals, s=14, c="0.2", alpha=0.45, edgecolors="none")
    ax_part.axhline(0.0, color="0.35", linewidth=0.6)
    med_v = float(np.nanmedian(_col("partial_b_vs_V_given_variance")))
    med_var = float(np.nanmedian(_col("partial_b_vs_variance_given_V")))
    ax_part.set_xticks(positions_p)
    ax_part.set_xticklabels(["B | variance\n(V residual)", "B | V\n(variance residual)"], fontsize=8)
    ax_part.set_ylabel("Partial Spearman ρ")
    ax_part.set_title(
        f"Partial horse-race (median |V|={abs(med_v):.2f}, |var|={abs(med_var):.2f})"
    )
    label_panel(ax_part, "b")

    fig.suptitle("B-factor prediction: constraint V vs local variance", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "bfactor_horse_race"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--figure-only", action="store_true", help="Rebuild figure from existing CSV")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.figure_only:
        csv_path = args.out_dir / "bfactor_horse_race.csv"
        if not csv_path.is_file():
            print(f"[horse_race] missing {csv_path}", file=sys.stderr)
            return 2
        rows = _load_horse_race_rows(csv_path)
        fig_path = _build_figure(rows, args.out_dir, args.dpi)
        print(f"[horse_race] figure → {fig_path}", flush=True)
        return 0

    ids: list[str] = []
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flexibility_source", "").strip() != "b_factor":
                continue
            ids.append(str(row["emdb_id"]).strip())

    rows: list[dict] = []
    for eid in ids:
        payload = _horse_race_one(eid, manifest=args.manifest, sphere_radius_a=args.sphere_radius_a)
        if payload is not None:
            rows.append(payload)
            print(
                f"[horse_race] EMD-{eid}: ρ(B,V)={payload['rho_b_vs_V']:+.3f} "
                f"ρ(B,var)={payload['rho_b_vs_variance']:+.3f} "
                f"partial(B|V,var)={payload['partial_b_vs_V_given_variance']:+.3f}",
                flush=True,
            )

    if not rows:
        print("[horse_race] no rows", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "bfactor_horse_race.csv"
    json_path = args.out_dir / "bfactor_horse_race.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else v) for k, v in row.items()})
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[horse_race] {len(rows)} maps -> {csv_path}", flush=True)
    fig_path = _build_figure(rows, args.out_dir, args.dpi)
    print(f"[horse_race] figure → {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
