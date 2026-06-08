"""Map reliability scores for model-building guidance (build / caution / omit zones).

LH fluctuation–constraint maps (T, V, L = T − V, H = T + V) from half-maps.
Primary ranked export: H_repro → reliability_score; L/T/V exported for analysis.
See docs/LH_MAP_RELIABILITY.md and docs/THESIS_AND_PUBLICATION.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import save_volume_like_reference
from .mechanics import compute_mechanics_headlines
from .mechanics import lh_map_metrics


def percentile_rank_in_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Map in-mask voxels to (0, 1] by rank; outside mask = 0.

    Higher values = higher rank among macromolecular voxels.
    """
    v = np.asarray(volume, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(v, dtype=np.float32)
    if not m.any():
        return out
    vals = v[m]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(vals, dtype=np.float64)
    ranks[order] = np.arange(1, vals.size + 1, dtype=np.float64)
    out[m] = (ranks / (vals.size + eps)).astype(np.float32)
    return out


def compute_reliability_maps(
    rho: np.ndarray,
    delta_rho: np.ndarray,
    *,
    window: int = 5,
    sigma: float = 1.0,
    kappa: float = 1.0,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Reproducibility-based reliability from averaged map rho and half difference.

    **Primary export:** ``reliability_score`` — in-mask percentile rank of
    :math:`H_{\\mathrm{repro}}` (higher = more reliable vs half-map CC on test maps).

    Also returns exploratory LH decomposition (T, V, L = T - V) for methods figures.
    """
    if rho.shape != delta_rho.shape:
        raise ValueError(f"Shape mismatch: rho {rho.shape} vs delta_rho {delta_rho.shape}")
    repro = lh_map_metrics(
        rho, delta_rho, window=window, sigma=sigma, kappa=kappa
    )
    h = np.asarray(repro["H_repro"], dtype=np.float32)
    t = np.asarray(repro["fluctuation_T"], dtype=np.float32)
    v = np.asarray(repro["constraint_V"], dtype=np.float32)
    out: dict[str, np.ndarray] = {
        "reliability_H_repro": h,
        "reliability_fluctuation": t,
        "reliability_smoothness": v,
        "reliability_fluctuation_T": t,
        "reliability_constraint_V": v,
        "reliability_L_balance": np.asarray(repro["L_balance"], dtype=np.float32),
    }
    if mask is not None:
        out["reliability_score"] = percentile_rank_in_mask(h, mask)
    else:
        out["reliability_score"] = h
    return out


def classify_build_zones(
    reliability_score: np.ndarray,
    mask: np.ndarray,
    *,
    build_pct: float = 66.67,
    caution_pct: float = 33.33,
) -> np.ndarray:
    """
    Discrete zone labels inside ``mask`` (uint8):

    - 0 = omit / low confidence (below ``caution_pct``)
    - 1 = caution (middle tercile by default)
    - 2 = build with confidence (top tercile by default)

    Outside mask = 0.
    """
    r = np.asarray(reliability_score, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    zones = np.zeros(r.shape, dtype=np.uint8)
    if not m.any():
        return zones
    vals = r[m]
    t_lo = float(np.percentile(vals, caution_pct))
    t_hi = float(np.percentile(vals, build_pct))
    inside = m.copy()
    zones[inside & (r < t_lo)] = 0
    zones[inside & (r >= t_lo) & (r < t_hi)] = 1
    zones[inside & (r >= t_hi)] = 2
    return zones


def attach_reliability_to_features(
    features: dict[str, np.ndarray],
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    window: int = 5,
    sigma: float = 1.0,
    kappa: float = 1.0,
    mask: np.ndarray | None = None,
    compute_zones: bool = True,
) -> dict[str, np.ndarray]:
    """
    Add reliability maps to a feature dict (requires ``density_normalized``).

    Returns the same dict, updated in place and also returned for chaining.
    """
    if "density_normalized" not in features:
        raise KeyError("features must contain density_normalized")
    if half1.shape != half2.shape:
        raise ValueError(f"Half-map shape mismatch: {half1.shape} vs {half2.shape}")
    rho = np.asarray(features["density_normalized"])
    if rho.shape != half1.shape:
        raise ValueError(f"Feature grid {rho.shape} != half-map grid {half1.shape}")
    delta = np.asarray(half1, dtype=np.float32) - np.asarray(half2, dtype=np.float32)
    rel = compute_reliability_maps(
        rho, delta, window=window, sigma=sigma, kappa=kappa, mask=mask
    )
    features.update(rel)
    if compute_zones and mask is not None:
        features["build_zone"] = classify_build_zones(rel["reliability_score"], mask)
    return features


def save_reliability_mrc(
    reference_path: str | Path,
    reliability: np.ndarray,
    out_path: str | Path | None = None,
    *,
    label: str = "reliability_score (cryoem_mrc)",
) -> Path:
    """Write reliability volume on the reference grid."""
    reference_path = Path(reference_path)
    if out_path is None:
        out_path = reference_path.with_name(f"{reference_path.stem}_reliability.mrc")
    else:
        out_path = Path(out_path)
    save_volume_like_reference(
        reference_path, reliability, out_path, dtype=np.float32, extra_label=label[:80]
    )
    return out_path


def save_build_zone_mrc(
    reference_path: str | Path,
    zones: np.ndarray,
    out_path: str | Path | None = None,
) -> Path:
    """Write build-zone labels (0/1/2) as MRC on the reference grid."""
    reference_path = Path(reference_path)
    if out_path is None:
        out_path = reference_path.with_name(f"{reference_path.stem}_build_zones.mrc")
    else:
        out_path = Path(out_path)
    save_volume_like_reference(
        reference_path,
        zones.astype(np.float32),
        out_path,
        dtype=np.float32,
        extra_label="build_zone 0=omit 1=caution 2=build",
    )
    return out_path
