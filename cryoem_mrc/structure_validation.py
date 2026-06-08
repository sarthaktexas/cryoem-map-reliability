"""Residue-level validation: deposited model B-factors vs map reliability (gemmi + MapGrid)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from .repo_paths import COHORT_MANIFEST, find_features_npz, halfmap_metrics_npz, lh_map_reliability_dir

import numpy as np
from scipy import stats

from .analysis import build_contour_mask
from .map_grid import MapGrid, load_map_grid


@dataclass(frozen=True)
class CaResidue:
    """One Cα atom with mmCIF/PDB identifiers and Cartesian coordinates (Å)."""

    chain: str
    seq_num: int
    seq_icode: str
    res_name: str
    x: float
    y: float
    z: float
    b_iso: float
    auth_chain: str = ""
    auth_seq_num: int = 0

    @property
    def residue_key(self) -> tuple[str, int, str]:
        """Deposit-stable match key: mmCIF (label_asym_id, label_seq_id, insertion)."""
        return (self.chain, self.seq_num, self.seq_icode)


@dataclass
class ResidueValidationRow:
    """Per-residue map samples and mask membership for external validation tables."""

    chain: str
    seq_num: int
    seq_icode: str
    res_name: str
    x: float
    y: float
    z: float
    b_iso: float
    reliability_score: float
    reliability_H_repro: float
    build_zone: int
    in_contour_mask: bool
    local_cross_correlation: float = float("nan")
    local_variance: float = float("nan")
    auth_chain: str = ""
    auth_seq_num: int = 0

    @property
    def residue_key(self) -> tuple[str, int, str]:
        """Deposit-stable match key: mmCIF (label_asym_id, label_seq_id, insertion)."""
        return (self.chain, self.seq_num, self.seq_icode)


@dataclass
class BfactorValidationStats:
    """Spearman correlations and zone summaries for one map + model pair."""

    emdb_id: str
    n_residues: int
    n_in_mask: int
    spearman_b_vs_reliability: float
    spearman_b_vs_H_repro: float
    spearman_b_vs_build_zone: float
    partial_b_vs_reliability_given_variance: float = float("nan")
    median_b_by_zone: dict[int, float] = field(default_factory=dict)
    notes: str = ""


def physical_xyz_to_voxel_indices(
    x: float,
    y: float,
    z: float,
    grid: MapGrid,
) -> tuple[int, int, int]:
    """
    Convert Cartesian coordinates (Å, standard x/y/z) to (iz, iy, ix) on ``grid.data``.

    Uses the same convention as :func:`map_grid.resample_volume_onto_grid` (voxel
    corners at ``origin + index * voxel_size``).
    """
    o = grid.origin_zyx
    v = grid.voxel_size_zyx
    iz = int(round((z - o[0]) / v[0]))
    iy = int(round((y - o[1]) / v[1]))
    ix = int(round((x - o[2]) / v[2]))
    nz, ny, nx = grid.shape_zyx
    iz = int(np.clip(iz, 0, nz - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    ix = int(np.clip(ix, 0, nx - 1))
    return iz, iy, ix


def iter_ca_residues(structure_path: str | Path) -> list[CaResidue]:
    """Load a deposited model and return one Cα per residue (first altloc).

    ``chain`` / ``seq_num`` use mmCIF label_asym_id and label_seq_id so conformation
    pairs match across deposits with different auth chain naming. ``auth_chain`` /
    ``auth_seq_num`` retain PDB auth ids for domain-band annotations.
    """
    import gemmi

    path = Path(structure_path)
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    residues: list[CaResidue] = []
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.entity_type == gemmi.EntityType.Water:
                    continue
                ca = residue.find_atom("CA", "\0")
                if ca is None:
                    continue
                label_chain = str(residue.subchain).strip() or chain.name
                auth_chain = chain.name
                auth_seq = int(residue.seqid.num)
                label_seq = int(residue.label_seq) if residue.label_seq else auth_seq
                residues.append(
                    CaResidue(
                        chain=label_chain,
                        seq_num=label_seq,
                        seq_icode=str(residue.seqid.icode).strip(),
                        res_name=residue.name,
                        x=float(ca.pos.x),
                        y=float(ca.pos.y),
                        z=float(ca.pos.z),
                        b_iso=float(ca.b_iso),
                        auth_chain=auth_chain,
                        auth_seq_num=auth_seq,
                    )
                )
        break  # first model only
    return residues


def _sample_nearest(volume: np.ndarray, iz: int, iy: int, ix: int) -> float:
    return float(volume[iz, iy, ix])


def _sample_window_mean(
    volume: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
    *,
    radius: int,
) -> float:
    nz, ny, nx = volume.shape
    z0, z1 = max(0, iz - radius), min(nz, iz + radius + 1)
    y0, y1 = max(0, iy - radius), min(ny, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(nx, ix + radius + 1)
    block = volume[z0:z1, y0:y1, x0:x1]
    if block.size == 0:
        return float("nan")
    return float(np.nanmean(block))


SphereAgg = Literal["mean", "median"]


def _sphere_voxel_indices_zyx(
    iz: int,
    iy: int,
    ix: int,
    grid: MapGrid,
    *,
    radius_a: float,
) -> np.ndarray:
    """Voxel centers within ``radius_a`` (Å) of (iz, iy, ix); shape (N, 3) int32 Z,Y,X."""
    nz, ny, nx = grid.shape_zyx
    vz, vy, vx = grid.voxel_size_zyx
    oz, oy, ox = grid.origin_zyx
    rz = int(np.ceil(radius_a / max(vz, 1e-6))) + 1
    ry = int(np.ceil(radius_a / max(vy, 1e-6))) + 1
    rx = int(np.ceil(radius_a / max(vx, 1e-6))) + 1
    z0, z1 = max(0, iz - rz), min(nz, iz + rz + 1)
    y0, y1 = max(0, iy - ry), min(ny, iy + ry + 1)
    x0, x1 = max(0, ix - rx), min(nx, ix + rx + 1)
    cz = oz + iz * vz
    cy = oy + iy * vy
    cx = ox + ix * vx
    r2_max = float(radius_a) ** 2
    triples: list[tuple[int, int, int]] = []
    for zz in range(z0, z1):
        dz = (oz + zz * vz) - cz
        for yy in range(y0, y1):
            dy = (oy + yy * vy) - cy
            for xx in range(x0, x1):
                dx = (ox + xx * vx) - cx
                if dz * dz + dy * dy + dx * dx <= r2_max:
                    triples.append((zz, yy, xx))
    if not triples:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triples, dtype=np.int32)


def build_ca_sphere_index_caches(
    residues: Sequence[CaResidue],
    grid: MapGrid,
    radius_a: float,
) -> list[np.ndarray]:
    """Per-residue (N, 3) Z,Y,X indices for an isotropic sphere of ``radius_a`` Å."""
    return [
        _sphere_voxel_indices_zyx(
            *physical_xyz_to_voxel_indices(res.x, res.y, res.z, grid),
            grid,
            radius_a=radius_a,
        )
        for res in residues
    ]


def aggregate_sphere_samples(
    volume: np.ndarray,
    indices_zyx: np.ndarray,
    *,
    agg: SphereAgg = "mean",
) -> tuple[float, int, int]:
    """
    Aggregate finite voxels inside a sphere index list.

    Returns ``(value, n_finite, n_sphere_voxels)``.
    """
    if indices_zyx.size == 0:
        return float("nan"), 0, 0
    vals = volume[indices_zyx[:, 0], indices_zyx[:, 1], indices_zyx[:, 2]].astype(np.float64)
    finite = vals[np.isfinite(vals)]
    n_sphere = int(indices_zyx.shape[0])
    n_finite = int(finite.size)
    if n_finite == 0:
        return float("nan"), 0, n_sphere
    if agg == "median":
        return float(np.median(finite)), n_finite, n_sphere
    return float(np.mean(finite)), n_finite, n_sphere


def sample_volume_sphere_cached(
    volume: np.ndarray,
    caches: Sequence[np.ndarray],
    *,
    agg: SphereAgg = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one volume at precomputed sphere caches; returns (values, n_finite)."""
    values = np.empty(len(caches), dtype=np.float64)
    n_finite = np.empty(len(caches), dtype=np.int32)
    for i, idx in enumerate(caches):
        v, nf, _ = aggregate_sphere_samples(volume, idx, agg=agg)
        values[i] = v
        n_finite[i] = nf
    return values, n_finite


def _sample_sphere_mean(
    volume: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
    grid: MapGrid,
    *,
    radius_a: float,
) -> float:
    """Mean over finite voxels whose physical centers fall within ``radius_a`` (Å)."""
    idx = _sphere_voxel_indices_zyx(iz, iy, ix, grid, radius_a=radius_a)
    val, _, _ = aggregate_sphere_samples(volume, idx, agg="mean")
    return val


def sample_volume_at_ca(
    volume: np.ndarray,
    grid: MapGrid,
    residues: Sequence[CaResidue],
    *,
    window_radius: int = 0,
    sphere_radius_a: float | None = None,
) -> np.ndarray:
    """
    Sample ``volume`` at each Cα.

    - ``window_radius=0`` and no sphere: nearest voxel
    - ``window_radius > 0``: cubic box mean (±radius voxels)
    - ``sphere_radius_a``: isotropic sphere mean in Å (overrides box when set)
    """
    out = np.empty(len(residues), dtype=np.float64)
    for i, res in enumerate(residues):
        iz, iy, ix = physical_xyz_to_voxel_indices(res.x, res.y, res.z, grid)
        if sphere_radius_a is not None and sphere_radius_a > 0:
            out[i] = _sample_sphere_mean(
                volume, iz, iy, ix, grid, radius_a=float(sphere_radius_a)
            )
        elif window_radius <= 0:
            out[i] = _sample_nearest(volume, iz, iy, ix)
        else:
            out[i] = _sample_window_mean(volume, iz, iy, ix, radius=window_radius)
    return out


def build_residue_validation_table(
    residues: Sequence[CaResidue],
    *,
    grid: MapGrid,
    reference_density: np.ndarray,
    contour: float,
    reliability_score: np.ndarray,
    reliability_H_repro: np.ndarray,
    build_zone: np.ndarray,
    local_cross_correlation: np.ndarray | None = None,
    local_variance: np.ndarray | None = None,
    window_radius: int = 0,
) -> list[ResidueValidationRow]:
    """Join Cα coordinates with reliability volumes and contour mask."""
    if reference_density.shape != reliability_score.shape:
        raise ValueError("reference_density and reliability volumes must share shape")
    mask = build_contour_mask(reference_density, contour)

    score_s = sample_volume_at_ca(
        reliability_score, grid, residues, window_radius=window_radius
    )
    h_s = sample_volume_at_ca(
        reliability_H_repro, grid, residues, window_radius=window_radius
    )
    zone_s = sample_volume_at_ca(
        build_zone.astype(np.float64), grid, residues, window_radius=window_radius
    )
    in_mask_s = sample_volume_at_ca(
        mask.astype(np.float64), grid, residues, window_radius=window_radius
    )
    cc_s = None
    var_s = None
    if local_cross_correlation is not None:
        cc_s = sample_volume_at_ca(
            local_cross_correlation, grid, residues, window_radius=window_radius
        )
    if local_variance is not None:
        var_s = sample_volume_at_ca(
            local_variance, grid, residues, window_radius=window_radius
        )

    rows: list[ResidueValidationRow] = []
    for i, res in enumerate(residues):
        rows.append(
            ResidueValidationRow(
                chain=res.chain,
                seq_num=res.seq_num,
                seq_icode=res.seq_icode,
                res_name=res.res_name,
                x=res.x,
                y=res.y,
                z=res.z,
                b_iso=res.b_iso,
                reliability_score=float(score_s[i]),
                reliability_H_repro=float(h_s[i]),
                build_zone=int(round(zone_s[i])),
                in_contour_mask=bool(in_mask_s[i] >= 0.5),
                local_cross_correlation=float(cc_s[i]) if cc_s is not None else float("nan"),
                local_variance=float(var_s[i]) if var_s is not None else float("nan"),
                auth_chain=res.auth_chain or res.chain,
                auth_seq_num=res.auth_seq_num or res.seq_num,
            )
        )
    return rows


def _b_iso_is_uniform(b: np.ndarray) -> bool:
    """True when deposited B-factors carry no in-mask variation (Spearman undefined)."""
    bf = b[np.isfinite(b)]
    return bf.size < 2 or np.unique(bf).size < 2


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xr, yr, zr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    r_xy = np.corrcoef(xr, yr)[0, 1]
    r_xz = np.corrcoef(xr, zr)[0, 1]
    r_yz = np.corrcoef(yr, zr)[0, 1]
    d = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if d <= 0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / np.sqrt(d))


def compute_bfactor_validation_stats(
    rows: Sequence[ResidueValidationRow],
    *,
    emdb_id: str,
    in_mask_only: bool = True,
) -> BfactorValidationStats:
    """Spearman correlations for B_iso vs reliability fields (in-mask residues by default)."""
    use = [r for r in rows if (r.in_contour_mask if in_mask_only else True)]
    n_all = len(rows)
    n_use = len(use)
    if n_use < 10:
        return BfactorValidationStats(
            emdb_id=emdb_id,
            n_residues=n_all,
            n_in_mask=n_use,
            spearman_b_vs_reliability=float("nan"),
            spearman_b_vs_H_repro=float("nan"),
            spearman_b_vs_build_zone=float("nan"),
            notes="too few residues for correlation",
        )

    b = np.array([r.b_iso for r in use], dtype=np.float64)
    rel = np.array([r.reliability_score for r in use], dtype=np.float64)
    h = np.array([r.reliability_H_repro for r in use], dtype=np.float64)
    zones = np.array([r.build_zone for r in use], dtype=np.float64)
    var = np.array([r.local_variance for r in use], dtype=np.float64)

    med_by_zone: dict[int, float] = {}
    for z in (0, 1, 2):
        zb = b[zones == z]
        if zb.size:
            med_by_zone[z] = float(np.median(zb))

    if _b_iso_is_uniform(b):
        return BfactorValidationStats(
            emdb_id=emdb_id,
            n_residues=n_all,
            n_in_mask=n_use,
            spearman_b_vs_reliability=float("nan"),
            spearman_b_vs_H_repro=float("nan"),
            spearman_b_vs_build_zone=float("nan"),
            partial_b_vs_reliability_given_variance=float("nan"),
            median_b_by_zone=med_by_zone,
            notes=(
                "Uniform deposited B-factors (B_iso has zero variance in mask); "
                "Spearman correlations skipped."
            ),
        )

    rho_rel, _ = stats.spearmanr(b, rel)
    rho_h, _ = stats.spearmanr(b, h)
    rho_z, _ = stats.spearmanr(b, zones)

    partial = float("nan")
    if np.isfinite(var).sum() >= 10:
        ok = np.isfinite(var)
        partial = _partial_spearman(b[ok], rel[ok], var[ok])

    return BfactorValidationStats(
        emdb_id=emdb_id,
        n_residues=n_all,
        n_in_mask=n_use,
        spearman_b_vs_reliability=float(rho_rel),
        spearman_b_vs_H_repro=float(rho_h),
        spearman_b_vs_build_zone=float(rho_z),
        partial_b_vs_reliability_given_variance=partial,
        median_b_by_zone=med_by_zone,
    )


@dataclass(frozen=True)
class BfactorDistributionSummary:
    """Quick sanity check on deposited B-factors before external validation."""

    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    notes: str = ""


@dataclass(frozen=True)
class BfactorScoreCorrelationRow:
    """One Spearman comparison: aggregated map score vs per-residue B_iso."""

    score_name: str
    atom_mode: str
    radius_a: float
    aggregation: str
    mask_policy: str
    n_residues: int
    n_used: int
    n_dropped: int
    median_n_voxels: float
    spearman_rho: float
    p_value: float
    spearman_vs_local_variance: float = float("nan")


def summarize_b_iso_distribution(b: np.ndarray) -> BfactorDistributionSummary:
    """Summarize B_iso; flag near-constant distributions (uninformative correlations)."""
    x = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return BfactorDistributionSummary(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                                          notes="no finite B_iso values")
    std = float(x.std())
    notes = ""
    if std < 1e-3:
        notes = "B_iso std near zero; Spearman vs map scores is not interpretable."
    return BfactorDistributionSummary(
        n=int(x.size),
        mean=float(x.mean()),
        std=std,
        min=float(x.min()),
        max=float(x.max()),
        median=float(np.median(x)),
        notes=notes,
    )


def rank_normalize_b_per_chain(residues: Sequence[CaResidue], b: np.ndarray) -> np.ndarray:
    """Rank-transform B_iso within each chain (removes chain-level offsets)."""
    out = np.asarray(b, dtype=np.float64).copy()
    by_chain: dict[str, list[int]] = {}
    for i, res in enumerate(residues):
        by_chain.setdefault(res.chain, []).append(i)
    for idxs in by_chain.values():
        sub = out[idxs]
        ok = np.isfinite(sub)
        if ok.sum() < 2:
            continue
        out[idxs[ok]] = stats.rankdata(sub[ok], method="average")
    return out


def mask_fraction_in_sphere_caches(
    mask: np.ndarray,
    caches: Sequence[np.ndarray],
) -> np.ndarray:
    """Fraction of sphere voxels inside a boolean/0-1 mask (nan-safe)."""
    frac = np.empty(len(caches), dtype=np.float64)
    for i, idx in enumerate(caches):
        v, _, _ = aggregate_sphere_samples(mask.astype(np.float64), idx, agg="mean")
        frac[i] = v
    return frac


def _spearman_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p), n


def compute_bfactor_score_correlation_rows(
    score_maps: Mapping[str, np.ndarray],
    residues: Sequence[CaResidue],
    b_iso: np.ndarray,
    *,
    sphere_caches_by_radius: Mapping[float, Sequence[np.ndarray]],
    radii_a: Sequence[float],
    aggregations: Sequence[SphereAgg] = ("mean",),
    atom_mode: str = "CA",
    residue_mask: np.ndarray | None = None,
    mask_policy: str = "all_residues",
    baseline_score: str = "local_variance",
) -> list[BfactorScoreCorrelationRow]:
    """
    Spearman ρ(aggregated score, B_iso) for each score × radius × aggregation.

    Also records ρ(score, baseline_score) at the same sampling (collinearity diagnostic).
    """
    n_res = len(residues)
    b = np.asarray(b_iso, dtype=np.float64)
    if residue_mask is not None:
        use_mask = np.asarray(residue_mask, dtype=bool)
    else:
        use_mask = np.ones(n_res, dtype=bool)

    if baseline_score not in score_maps:
        raise KeyError(f"baseline_score {baseline_score!r} not in score_maps")

    rows: list[BfactorScoreCorrelationRow] = []
    for radius in radii_a:
        caches = sphere_caches_by_radius[float(radius)]
        for agg in aggregations:
            baseline_vals, _ = sample_volume_sphere_cached(
                score_maps[baseline_score], caches, agg=agg
            )
            for score_name, vol in score_maps.items():
                scores, n_fin = sample_volume_sphere_cached(vol, caches, agg=agg)
                m = use_mask & np.isfinite(b) & np.isfinite(scores)
                n_used = int(m.sum())
                n_dropped = n_res - n_used
                rho, p, _ = _spearman_pair(scores[m], b[m])
                col_rho, _, _ = _spearman_pair(scores[m], baseline_vals[m])
                med_nv = float(np.median(n_fin[m])) if m.any() else float("nan")
                rows.append(
                    BfactorScoreCorrelationRow(
                        score_name=score_name,
                        atom_mode=atom_mode,
                        radius_a=float(radius),
                        aggregation=agg,
                        mask_policy=mask_policy,
                        n_residues=n_res,
                        n_used=n_used,
                        n_dropped=n_dropped,
                        median_n_voxels=med_nv,
                        spearman_rho=rho,
                        p_value=p,
                        spearman_vs_local_variance=col_rho,
                    )
                )
    return rows


def write_bfactor_score_correlation_csv(
    path: str | Path,
    rows: Sequence[BfactorScoreCorrelationRow],
) -> Path:
    path = Path(path)
    fields = [
        "score_name",
        "atom_mode",
        "radius",
        "aggregation",
        "mask_policy",
        "n_residues",
        "n_used",
        "n_dropped",
        "median_n_voxels",
        "spearman_rho",
        "p_value",
        "spearman_vs_local_variance",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "score_name": r.score_name,
                    "atom_mode": r.atom_mode,
                    "radius": f"{r.radius_a:.2f}",
                    "aggregation": r.aggregation,
                    "mask_policy": r.mask_policy,
                    "n_residues": r.n_residues,
                    "n_used": r.n_used,
                    "n_dropped": r.n_dropped,
                    "median_n_voxels": f"{r.median_n_voxels:.1f}",
                    "spearman_rho": f"{r.spearman_rho:.6f}",
                    "p_value": f"{r.p_value:.6e}",
                    "spearman_vs_local_variance": f"{r.spearman_vs_local_variance:.6f}",
                }
            )
    return path


def write_residue_validation_csv(path: str | Path, rows: Sequence[ResidueValidationRow]) -> Path:
    """Write ``residue_validation.csv`` for thesis / plotting."""
    path = Path(path)
    fieldnames = [
        "chain",
        "seq_num",
        "auth_chain",
        "auth_seq_num",
        "seq_icode",
        "res_name",
        "x",
        "y",
        "z",
        "b_iso",
        "reliability_score",
        "reliability_H_repro",
        "build_zone",
        "in_contour_mask",
        "local_cross_correlation",
        "local_variance",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "chain": r.chain,
                    "seq_num": r.seq_num,
                    "auth_chain": r.auth_chain or r.chain,
                    "auth_seq_num": r.auth_seq_num or r.seq_num,
                    "seq_icode": r.seq_icode,
                    "res_name": r.res_name,
                    "x": f"{r.x:.3f}",
                    "y": f"{r.y:.3f}",
                    "z": f"{r.z:.3f}",
                    "b_iso": f"{r.b_iso:.2f}",
                    "reliability_score": f"{r.reliability_score:.6f}",
                    "reliability_H_repro": f"{r.reliability_H_repro:.6f}",
                    "build_zone": r.build_zone,
                    "in_contour_mask": int(r.in_contour_mask),
                    "local_cross_correlation": (
                        f"{r.local_cross_correlation:.6f}"
                        if np.isfinite(r.local_cross_correlation)
                        else ""
                    ),
                    "local_variance": (
                        f"{r.local_variance:.6f}" if np.isfinite(r.local_variance) else ""
                    ),
                }
            )
    return path


def load_cohort_manifest_row(manifest_path: Path, emdb_id: str) -> dict[str, str]:
    """Return one ``cohort/manifest.csv`` row by ``emdb_id``."""
    emdb_id = str(emdb_id).strip()
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("emdb_id", "")).strip() == emdb_id:
                return row
    raise KeyError(f"emdb_id {emdb_id} not found in {manifest_path}")


def default_reliability_out_dir(emdb_id: str, base: Path | None = None) -> Path:
    """Per-map reliability bundle: ``outputs/emd_<ID>/lh_map_reliability/``."""
    _ = base  # deprecated; kept for call-site compatibility
    return lh_map_reliability_dir(emdb_id)


def match_residue_rows_by_key(
    rows_a: Sequence[ResidueValidationRow],
    rows_b: Sequence[ResidueValidationRow],
) -> list[tuple[ResidueValidationRow, ResidueValidationRow]]:
    """Pair residues by mmCIF (label_asym_id, label_seq_id, insertion) for conformation comparisons."""
    index_b = {r.residue_key: r for r in rows_b}
    pairs: list[tuple[ResidueValidationRow, ResidueValidationRow]] = []
    for ra in rows_a:
        key = ra.residue_key
        rb = index_b.get(key)
        if rb is not None:
            pairs.append((ra, rb))
    return pairs


@dataclass
class ConformationPairStats:
    """ΔB vs Δreliability on matched residues (two maps / two models)."""

    emdb_a: str
    emdb_b: str
    n_matched: int
    n_matched_in_mask_both: int
    spearman_delta_b_vs_delta_reliability: float
    spearman_delta_b_vs_delta_H_repro: float


def compute_conformation_pair_stats(
    pairs: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
) -> ConformationPairStats:
    use = pairs
    if in_mask_both:
        use = [(a, b) for a, b in pairs if a.in_contour_mask and b.in_contour_mask]
    n_match = len(pairs)
    n_use = len(use)
    if n_use < 10:
        return ConformationPairStats(
            emdb_a=emdb_a,
            emdb_b=emdb_b,
            n_matched=n_match,
            n_matched_in_mask_both=n_use,
            spearman_delta_b_vs_delta_reliability=float("nan"),
            spearman_delta_b_vs_delta_H_repro=float("nan"),
        )
    db = np.array([b.b_iso - a.b_iso for a, b in use], dtype=np.float64)
    drel = np.array([b.reliability_score - a.reliability_score for a, b in use], dtype=np.float64)
    dh = np.array([b.reliability_H_repro - a.reliability_H_repro for a, b in use], dtype=np.float64)
    r_rel, _ = stats.spearmanr(db, drel)
    r_h, _ = stats.spearmanr(db, dh)
    return ConformationPairStats(
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        n_matched=n_match,
        n_matched_in_mask_both=n_use,
        spearman_delta_b_vs_delta_reliability=float(r_rel),
        spearman_delta_b_vs_delta_H_repro=float(r_h),
    )


def read_residue_validation_csv(path: str | Path) -> list[ResidueValidationRow]:
    """Load rows written by :func:`write_residue_validation_csv`."""
    path = Path(path)
    rows: list[ResidueValidationRow] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                ResidueValidationRow(
                    chain=row["chain"],
                    seq_num=int(row["seq_num"]),
                    seq_icode=row["seq_icode"],
                    res_name=row["res_name"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    b_iso=float(row["b_iso"]),
                    reliability_score=float(row["reliability_score"]),
                    reliability_H_repro=float(row["reliability_H_repro"]),
                    build_zone=int(row["build_zone"]),
                    in_contour_mask=bool(int(row["in_contour_mask"])),
                    local_cross_correlation=float(row["local_cross_correlation"] or "nan"),
                    local_variance=float(row["local_variance"] or "nan"),
                    auth_chain=row.get("auth_chain") or row["chain"],
                    auth_seq_num=int(row.get("auth_seq_num") or row["seq_num"]),
                )
            )
    return rows


def run_emdb_bfactor_validation(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reliability_dir: Path | None = None,
    reliability_npz: Path | None = None,
    reference: Path | None = None,
    pdb: Path | None = None,
    contour: float | None = None,
    halfmap_npz: Path | None = None,
    features_npz: Path | None = None,
    window_radius: int = 0,
    require_b_factor_source: bool = True,
) -> tuple[int, list[ResidueValidationRow], BfactorValidationStats | None, Path]:
    """
    Run residue-level validation for one EMDB entry.

    Returns ``(exit_code, rows, stats, out_dir)``. ``stats`` is None when skipped.
    """
    row = load_cohort_manifest_row(manifest, emd_id)
    if require_b_factor_source and row.get("flexibility_source", "").strip() != "b_factor" and pdb is None:
        return 0, [], None, default_reliability_out_dir(emd_id)

    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    out_dir = default_reliability_out_dir(emd_id, reliability_dir)
    npz_path = reliability_npz or (out_dir / "reliability.npz")
    if halfmap_npz is None:
        candidate = halfmap_metrics_npz(emd_id)
        halfmap_npz = candidate if candidate.exists() else None

    for label, p in (("reference", ref_path), ("pdb", pdb_path), ("reliability.npz", npz_path)):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    residues = iter_ca_residues(pdb_path)
    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)

    with np.load(npz_path, allow_pickle=False) as d:
        reliability_score = np.asarray(d["reliability_score"], dtype=np.float32)
        reliability_H_repro = np.asarray(d["reliability_H_repro"], dtype=np.float32)
        build_zone = np.asarray(d["build_zone"], dtype=np.uint8)

    if reliability_score.shape != reference_density.shape:
        raise ValueError(
            f"EMD-{emd_id}: reliability shape {reliability_score.shape} != reference {reference_density.shape}"
        )

    cc = None
    if halfmap_npz is not None and halfmap_npz.exists():
        with np.load(halfmap_npz, allow_pickle=False) as hm:
            cc = np.asarray(hm["local_cross_correlation"], dtype=np.float32)
    local_var = None
    if features_npz is None:
        features_npz = find_features_npz(ref_path.parent, emd_id, contour_val)
    if features_npz is not None and features_npz.exists():
        with np.load(features_npz, allow_pickle=False) as feat:
            local_var = np.asarray(feat["local_variance"], dtype=np.float32)

    rows = build_residue_validation_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour_val,
        reliability_score=reliability_score,
        reliability_H_repro=reliability_H_repro,
        build_zone=build_zone,
        local_cross_correlation=cc,
        local_variance=local_var,
        window_radius=window_radius,
    )
    stats = compute_bfactor_validation_stats(rows, emdb_id=emd_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_residue_validation_csv(out_dir / "residue_validation.csv", rows)
    write_bfactor_validation_md(
        out_dir / "B_FACTOR_VALIDATION.md",
        stats,
        pdb_path=pdb_path,
        contour=contour_val,
        sampling="nearest voxel" if window_radius <= 0 else f"{2 * window_radius + 1}³ window mean",
    )
    return 0, rows, stats, out_dir


def write_bfactor_validation_md(
    path: Path,
    stats: BfactorValidationStats,
    *,
    pdb_path: Path,
    contour: float,
    sampling: str = "nearest voxel",
) -> None:
    """Short markdown summary for one map."""
    zlabels = {0: "omit", 1: "caution", 2: "build"}
    zone_lines = "\n".join(
        f"| {zlabels.get(z, z)} | {stats.median_b_by_zone.get(z, float('nan')):.1f} |"
        for z in (0, 1, 2)
        if z in stats.median_b_by_zone
    )
    caveat_line = f"\n\n**Caveat:** {stats.notes}" if stats.notes else ""
    text = f"""# B-factor external validation — EMD-{stats.emdb_id}

Exploratory comparison of **deposited model B-factors** vs **map reliability** (H_repro / build zones).
This does **not** claim H_repro measures molecular flexibility — B_iso reflects refinement displacement
and local order, while reliability_score reflects half-map agreement inside the contour mask.

**Model:** `{pdb_path}`  
**Mask:** deposited reference ρ ≥ {contour} (Cα: {sampling})  
**Residues:** {stats.n_residues:,} Cα total; **{stats.n_in_mask:,}** inside contour mask

---

## Spearman correlations (in-mask Cα)

| Comparison | ρ |
|------------|--:|
| B_iso vs reliability_score | {stats.spearman_b_vs_reliability:+.3f} |
| B_iso vs reliability_H_repro | {stats.spearman_b_vs_H_repro:+.3f} |
| B_iso vs build_zone (0/1/2) | {stats.spearman_b_vs_build_zone:+.3f} |
| Partial: B vs reliability \\| local_variance | {stats.partial_b_vs_reliability_given_variance:+.3f} |

**Sign note:** Higher B_iso ↔ more displacement; higher reliability_score ↔ more reliable map.
A **negative** ρ(B, reliability) is the naive expectation if both proxy local order.{caveat_line}

---

## Median B_iso by build zone

| Zone | Median B_iso |
|------|-------------:|
{zone_lines}

---

## Files

| File | Description |
|------|-------------|
| `residue_validation.csv` | Per-residue table |
| `figures/bfactor_vs_reliability.png` | Scatter (in-mask) |
"""
    path.write_text(text)


def write_conformation_pair_md(
    path: Path,
    pair_stats: ConformationPairStats,
    coverage: object | None = None,
) -> None:
    cov_block = ""
    if coverage is not None:
        flag = " **YES — discuss in thesis**" if coverage.coverage_flag else " no"
        cov_block = f"""
## Coverage vs deposited model

| Metric | State A (EMD-{coverage.emdb_a}) | State B (EMD-{coverage.emdb_b}) |
|--------|--------------------------------:|--------------------------------:|
| Deposited Cα total | {coverage.n_ca_total_a:,} | {coverage.n_ca_total_b:,} |
| Matched (any mask) | {coverage.n_matched:,} | {coverage.n_matched:,} |
| Both in contour mask | {coverage.n_matched_in_mask_both:,} | {coverage.n_matched_in_mask_both:,} |
| Analysis / deposited Cα | {100 * coverage.frac_analysis_of_a:.1f}% | {100 * coverage.frac_analysis_of_b:.1f}% |
| Missing from analysis | {coverage.missing_pct_a:.1f}% | {coverage.missing_pct_b:.1f}% |

Flag (>20% missing):{flag}

{coverage.notes}
"""
    text = f"""# Conformation pair — EMD-{pair_stats.emdb_a} vs EMD-{pair_stats.emdb_b}

Matched Cα by mmCIF (label_asym_id, label_seq_id, insertion). Δ = state B − state A.

| Metric | Value |
|--------|------:|
| Matched residues | {pair_stats.n_matched:,} |
| Both in contour mask | {pair_stats.n_matched_in_mask_both:,} |
| Spearman ρ(ΔB, Δreliability_score) | {pair_stats.spearman_delta_b_vs_delta_reliability:+.3f} |
| Spearman ρ(ΔB, ΔH_repro) | {pair_stats.spearman_delta_b_vs_delta_H_repro:+.3f} |
{cov_block}
Large |ΔB| often reflects **biochemical conformational change**, not map-quality change alone.

See `docs/CONFORMATION_PAIR_ANALYSIS.md` for clustering, coupling maps, and figure outputs.
"""
    path.write_text(text)
