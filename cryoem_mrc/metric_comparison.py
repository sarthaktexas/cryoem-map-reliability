"""Cross-metric residue tables and correlations for cohort validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import build_contour_mask
from .local_resolution import aggregate_locres_to_ca, locres_blocres_path
from .map_grid import load_map_grid
from .repo_paths import COHORT_MANIFEST, find_features_npz, halfmap_metrics_npz, lh_map_reliability_dir
from .structure_validation import (
    build_residue_validation_table,
    iter_ca_residues,
    load_cohort_manifest_row,
    sample_volume_at_ca,
)

logger = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "v_metric",
    "reliability_score",
    "reliability_H_repro",
    "b_factor",
    "local_cross_correlation",
    "local_variance",
    "local_resolution",
)


def load_all_metrics(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
) -> pd.DataFrame:
    """
    Per-residue metrics for one EMDB entry.

    ``local_resolution`` is filled from ``outputs/emd_<ID>/locres_blocres.mrc`` when
    present; otherwise NaN. Other columns come from the existing LH / validation
    pipeline (``reliability.npz``, half-map metrics, deposited B-factors).
    """
    emdb_id = str(emdb_id).strip()
    row = load_cohort_manifest_row(manifest, emdb_id)
    ref_path = Path(row["reference_mrc"])
    pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
    pdb_path = Path(pdb_raw) if pdb_raw else None
    contour = float(row["contour"])

    out_dir = lh_map_reliability_dir(emdb_id)
    npz_path = out_dir / "reliability.npz"

    if not ref_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing reference: {ref_path}")
    if pdb_path is None or not pdb_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing structure: {pdb_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing reliability.npz: {npz_path}")

    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)

    with np.load(npz_path, allow_pickle=False) as d:
        reliability_score = np.asarray(d["reliability_score"], dtype=np.float32)
        reliability_H_repro = np.asarray(d["reliability_H_repro"], dtype=np.float32)
        build_zone = np.asarray(d["build_zone"], dtype=np.uint8)
        v_metric_vol = np.asarray(
            d.get("reliability_smoothness", d.get("reliability_constraint_V")),
            dtype=np.float32,
        )

    halfmap_npz = halfmap_metrics_npz(emdb_id)
    cc = None
    if halfmap_npz.is_file():
        with np.load(halfmap_npz, allow_pickle=False) as hm:
            cc = np.asarray(hm["local_cross_correlation"], dtype=np.float32)

    features_npz = find_features_npz(ref_path.parent, emdb_id, contour)
    local_var = None
    if features_npz is not None and features_npz.is_file():
        with np.load(features_npz, allow_pickle=False) as feat:
            local_var = np.asarray(feat["local_variance"], dtype=np.float32)

    residues = iter_ca_residues(pdb_path)
    rows = build_residue_validation_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour,
        reliability_score=reliability_score,
        reliability_H_repro=reliability_H_repro,
        build_zone=build_zone,
        local_cross_correlation=cc,
        local_variance=local_var,
        window_radius=0,
    )
    v_at_ca = sample_volume_at_ca(
        v_metric_vol,
        grid,
        residues,
        sphere_radius_a=sphere_radius_a,
    )

    df = pd.DataFrame(
        {
            "emdb_id": emdb_id,
            "chain": [r.chain for r in rows],
            "seq_num": [r.seq_num for r in rows],
            "seq_icode": [r.seq_icode for r in rows],
            "res_name": [r.res_name for r in rows],
            "v_metric": v_at_ca,
            "reliability_score": [r.reliability_score for r in rows],
            "reliability_H_repro": [r.reliability_H_repro for r in rows],
            "b_factor": [r.b_iso for r in rows],
            "local_cross_correlation": [r.local_cross_correlation for r in rows],
            "local_variance": [r.local_variance for r in rows],
            "build_zone": [r.build_zone for r in rows],
            "in_contour_mask": [r.in_contour_mask for r in rows],
            "local_resolution": np.nan,
        }
    )

    locres_path = locres_blocres_path(emdb_id)
    if locres_path.is_file():
        try:
            loc_df = aggregate_locres_to_ca(
                locres_path,
                pdb_path,
                radius_angstrom=sphere_radius_a,
                reference_path=ref_path,
            )
            loc_df = loc_df.rename(columns={"local_resolution_mean": "local_resolution"})
            df = df.drop(columns=["local_resolution"]).merge(
                loc_df[["chain", "seq_num", "local_resolution"]],
                on=["chain", "seq_num"],
                how="left",
            )
        except Exception as exc:
            logger.warning(
                "EMD-%s: failed to aggregate %s: %s",
                emdb_id,
                locres_path,
                exc,
            )
    else:
        logger.warning(
            "EMD-%s: no BlocRes map at %s; local_resolution left as NaN",
            emdb_id,
            locres_path,
        )

    return df


def compute_cross_metric_correlations(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...] = METRIC_COLUMNS,
    min_pairs: int = 30,
    mask_column: str = "in_contour_mask",
) -> pd.DataFrame:
    """
    Pairwise Spearman ρ between numeric metric columns (in-mask residues by default).
    """
    use = df
    if mask_column in df.columns:
        use = df[df[mask_column].astype(bool)]

    avail = [c for c in columns if c in use.columns]
    numeric = use[avail].apply(pd.to_numeric, errors="coerce")
    n = len(avail)
    rho = np.full((n, n), np.nan, dtype=np.float64)
    pval = np.full((n, n), np.nan, dtype=np.float64)

    for i, ci in enumerate(avail):
        for j, cj in enumerate(avail):
            if j < i:
                rho[i, j] = rho[j, i]
                pval[i, j] = pval[j, i]
                continue
            m = numeric[ci].notna() & numeric[cj].notna()
            if m.sum() < min_pairs:
                continue
            r, p = stats.spearmanr(numeric.loc[m, ci], numeric.loc[m, cj])
            rho[i, j] = float(r)
            pval[i, j] = float(p)

    out = pd.DataFrame(rho, index=avail, columns=avail)
    out.attrs["p_values"] = pd.DataFrame(pval, index=avail, columns=avail)
    if mask_column in use.columns:
        out.attrs["n_residues"] = int(use[mask_column].astype(bool).sum())
    else:
        out.attrs["n_residues"] = len(use)
    return out
