"""LH metrics vs B-factors, half-map CC, and local FSC (per-map cohort table).

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_lh_vs_b_and_fsc.py --emd-id 11638
    PYTHONUNBUFFERED=1 python scripts/run_lh_vs_b_and_fsc.py --emd-id 23129
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc
from cryoem_mrc.local_resolution_io import load_local_resolution_map, resample_local_resolution_onto_reference
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.mechanics import fluctuation_constraint_decomposition
from cryoem_mrc.repo_paths import halfmap_metrics_npz
from cryoem_mrc.structure_validation import iter_ca_residues, sample_volume_at_ca

MAP_CONFIG: dict[str, dict] = {
    "49450": {
        "data_dir": "data/emd_49450-mgtA_e2p+e1",
        "contour": 0.116,
        "features": "emd_49450_avg_features_t0116.npz",
        "local_fsc": "emd_49450_local_fsc_t0143_P17_s4.mrc",
        "pdb": "pdb/9nhz.cif",
    },
    "11638": {
        "data_dir": "data/emd_11638-atomic_apoferritin",
        "contour": 0.116,
        "features": "emd_11638_avg_features_t0116.npz",
        "local_fsc": "emd_11638_local_fsc_t0143_P17_s4.mrc",
        "pdb": "pdb/7a4m.cif",
    },
    "23129": {
        "data_dir": "data/emd_23129-trpv1_ph6a",
        "contour": 0.009,
        "features": "emd_23129_avg_features_t0009.npz",
        "local_fsc": "emd_23129_local_fsc_t0143_P17_s4.mrc",
        "pdb": None,
    },
}


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 30:
        return float("nan")
    xr, yr, zr = stats.rankdata(x[m]), stats.rankdata(y[m]), stats.rankdata(z[m])
    if min(xr.std(), yr.std(), zr.std()) == 0:
        return float("nan")
    r_xy = float(np.corrcoef(xr, yr)[0, 1])
    r_xz = float(np.corrcoef(xr, zr)[0, 1])
    r_yz = float(np.corrcoef(yr, zr)[0, 1])
    denom = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if denom <= 0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / np.sqrt(denom))


def run_one(
    emd_id: str,
    *,
    sphere_radius_a: float = 2.0,
    max_voxel_samples: int = 400_000,
    window: int = 5,
) -> dict:
    cfg = MAP_CONFIG[emd_id]
    d = Path(cfg["data_dir"])
    emd = f"emd_{emd_id}"

    with np.load(d / cfg["features"], allow_pickle=False) as z:
        rho = np.asarray(z["density_normalized"], dtype=np.float32)
        var = np.asarray(z["local_variance"], dtype=np.float32)

    bundle = load_full_and_half_maps(
        d / f"{emd}.map", d / f"{emd}_half_map_1.map", d / f"{emd}_half_map_2.map"
    )
    delta = bundle.half1.data.astype(np.float32) - bundle.half2.data.astype(np.float32)
    lh = fluctuation_constraint_decomposition(rho, delta, window=window)

    with np.load(halfmap_metrics_npz(emd_id), allow_pickle=False) as z:
        cc = np.asarray(z["local_cross_correlation"], dtype=np.float32)

    grid = load_map_grid(d / f"{emd}.map")
    lfsc_local = load_local_resolution_map(d / cfg["local_fsc"])
    lfsc = resample_local_resolution_onto_reference(lfsc_local, grid).data.astype(np.float32)

    ref = load_mrc(d / f"{emd}.map", dtype=np.float32)
    mask = build_contour_mask(ref, float(cfg["contour"]))

    maps = {
        "local_variance": var,
        "halfmap_CC": cc,
        "local_fsc_A": lfsc,
        "lh_T": lh["fluctuation_T"],
        "lh_V": lh["constraint_V"],
        "lh_L": lh["L_balance"],
        "lh_H": lh["H_sum"],
        "V_minus_T": (lh["constraint_V"] - lh["fluctuation_T"]).astype(np.float32),
    }

    rows: list[dict] = []
    partial_l_b_var = float("nan")

    pdb = cfg.get("pdb")
    if pdb and Path(pdb).exists():
        residues = iter_ca_residues(pdb)
        b = np.array([r.b_iso for r in residues], dtype=np.float64)
        for name, vol in maps.items():
            s = sample_volume_at_ca(vol, grid, residues, sphere_radius_a=sphere_radius_a)
            m = np.isfinite(s) & np.isfinite(b)
            rows.append(
                {
                    "feature": name,
                    "target": "B_iso",
                    "level": f"residue_{sphere_radius_a}A_sphere",
                    "spearman": float(stats.spearmanr(s[m], b[m]).statistic),
                    "n": int(m.sum()),
                }
            )
        s_l = sample_volume_at_ca(lh["L_balance"], grid, residues, sphere_radius_a=sphere_radius_a)
        s_v = sample_volume_at_ca(var, grid, residues, sphere_radius_a=sphere_radius_a)
        m = np.isfinite(s_l) & np.isfinite(s_v) & np.isfinite(b)
        partial_l_b_var = _partial_spearman(s_l[m], b[m], s_v[m])

    idx = np.flatnonzero(mask)
    if idx.size > max_voxel_samples:
        idx = np.random.default_rng(0).choice(idx, max_voxel_samples, replace=False)
    for name, vol in maps.items():
        for target_name, target in (("halfmap_CC", cc), ("local_fsc_A", lfsc)):
            x = vol.ravel()[idx].astype(np.float64)
            y = target.ravel()[idx].astype(np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "feature": name,
                    "target": target_name,
                    "level": "voxel_masked",
                    "spearman": float(stats.spearmanr(x[m], y[m]).statistic),
                    "n": int(m.sum()),
                }
            )

    return {
        "emd_id": emd_id,
        "contour": cfg["contour"],
        "window": window,
        "sphere_radius_a": sphere_radius_a,
        "rows": rows,
        "partial_lh_L_vs_B_given_variance": partial_l_b_var,
        "pdb": pdb,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", action="append", required=True)
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--window", type=int, default=5)
    args = p.parse_args(argv)

    for emd_id in args.emd_id:
        if emd_id not in MAP_CONFIG:
            print(f"Unknown emd-id {emd_id}", file=sys.stderr)
            return 2
        print(f"[lh_b_fsc] EMD-{emd_id} ...", flush=True)
        payload = run_one(emd_id, sphere_radius_a=args.sphere_radius_a, window=args.window)
        out = Path("outputs") / f"emd_{emd_id}" / "extended_feature_validation" / "lh_vs_b_and_fsc.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

        print(f"  residue vs B_iso ({args.sphere_radius_a} Å):", flush=True)
        for r in payload["rows"]:
            if r["target"] == "B_iso":
                print(f"    {r['feature']:16s}  rho={r['spearman']:+.4f}", flush=True)
        if np.isfinite(payload["partial_lh_L_vs_B_given_variance"]):
            print(f"  partial rho(lh_L, B | variance) = {payload['partial_lh_L_vs_B_given_variance']:+.4f}", flush=True)
        print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
