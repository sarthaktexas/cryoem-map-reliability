"""Voxel-wise preliminary rigidity score from gradients, local variance, and multi-scale agreement."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _percentile_scale01(
    x: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Map ``x`` into approximately ``[0, 1]`` using robust percentiles (full-volume)."""
    a = np.asarray(x)
    teps = np.float32(1e-6) if a.dtype == np.float32 else eps
    pl, ph = np.percentile(a, (lo_pct, hi_pct))
    return np.clip((a - pl) / (ph - pl + teps), 0.0, 1.0)


def _cross_scale_consistency(
    per_scale: list[np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per voxel: volume-wise z-score each scale map, then ``1 / (1 + std)`` across scales.

    High when gradient (or other) structure is similar across smoothing scales.
    """
    a0 = np.asarray(per_scale[0])
    dt = np.float32 if a0.dtype == np.float32 else np.float64
    teps = np.float32(1e-6) if dt == np.float32 else eps
    normed: list[np.ndarray] = []
    for arr in per_scale:
        a = np.asarray(arr, dtype=dt)
        m = a.mean(dtype=np.float64)
        s = a.std(dtype=np.float64)
        normed.append(((a - m) / (s + teps)).astype(dt, copy=False))
    if not normed:
        raise ValueError("cross_scale_consistency requires at least one array")
    if len(normed) == 1:
        return np.ones_like(normed[0], dtype=dt)
    stack = np.stack(normed, axis=-1)
    sd = np.std(stack, axis=-1, dtype=np.float64).astype(dt, copy=False)
    return (np.float32(1.0) if dt == np.float32 else 1.0) / (
        (np.float32(1.0) if dt == np.float32 else 1.0) + sd
    )


def _per_scale_gradient_stack(features: dict[str, np.ndarray]) -> list[np.ndarray]:
    pairs: list[tuple[int, np.ndarray]] = []
    for k, v in features.items():
        m = re.fullmatch(r"gauss_s(\d+)_gradient_magnitude", k)
        if m is not None:
            pairs.append((int(m.group(1)), np.asarray(v)))
    pairs.sort(key=lambda t: t[0])
    return [arr for _, arr in pairs]


def compute_rigidity_map(
    features: dict[str, np.ndarray],
    *,
    w_gradient: float = 1.0 / 3.0,
    w_variance: float = 1.0 / 3.0,
    w_consistency: float = 1.0 / 3.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Preliminary voxel-wise rigidity (higher ≈ smoother, less locally variable, scale-stable).

    **Components** (each in ``[0, 1]`` before weighting):

    - **Gradient term**: ``1 - robust_norm(gradient_magnitude)`` — down-weights voxels
      with globally large gradient (often noisy boundaries / solvent).
    - **Variance term**: ``1 - robust_norm(local_variance)`` — rewards spatially
      uniform neighborhoods on the normalized density.
    - **Consistency term**: from per-scale ``gauss_s*`` gradient magnitudes, rewards
      agreement of structure across Gaussian scales (after per-map z-scoring).

    Final score is a weighted sum of the three terms; weights are renormalized to sum
    to 1. Requires ``gradient_magnitude``, ``local_variance``, and at least one
    ``gauss_s*_gradient_magnitude`` entry (multi-scale pipeline).
    """
    grad = np.asarray(features["gradient_magnitude"])
    var = np.asarray(features["local_variance"])
    dt = np.float32 if grad.dtype == np.float32 else np.float64

    g_n = _percentile_scale01(grad, eps=eps)
    v_n = _percentile_scale01(var, eps=eps)
    grad_term = (np.float32(1.0) if dt == np.float32 else 1.0) - g_n
    var_term = (np.float32(1.0) if dt == np.float32 else 1.0) - v_n

    gms = _per_scale_gradient_stack(features)
    if len(gms) >= 2:
        cons_term = _cross_scale_consistency(gms, eps=eps)
        # squash to [0, 1] for comparable weighting (theoretical max is 1 at sd=0)
        cons_term = np.clip(cons_term, 0.0, 1.0)
    else:
        cons_term = np.ones_like(grad_term, dtype=dt)

    wg = float(w_gradient)
    wv = float(w_variance)
    wc = float(w_consistency)
    ws = wg + wv + wc
    if ws <= 0.0:
        raise ValueError("rigidity weights must sum to a positive value")
    wg, wv, wc = wg / ws, wv / ws, wc / ws

    out = wg * grad_term + wv * var_term + wc * cons_term
    return out.astype(dt, copy=False)


def compute_rigidity_map_from_npz(
    npz_path: str | Path,
    *,
    mask: np.ndarray | None = None,
    **kwargs: float,
) -> np.ndarray:
    """
    Memory-bounded rigidity: load one feature array at a time from a compressed NPZ.

    When ``mask`` is set, robust percentiles use only in-mask voxels (recommended for
    large boxes where solvent would otherwise dominate the scaling).
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "gradient_magnitude" not in data.files or "local_variance" not in data.files:
            raise KeyError(f"{path.name} missing gradient_magnitude or local_variance")
        grad = np.asarray(data["gradient_magnitude"], dtype=np.float32)
        var = np.asarray(data["local_variance"], dtype=np.float32)
        gms_keys = sorted(
            k for k in data.files if re.fullmatch(r"gauss_s\d+_gradient_magnitude", k)
        )
        gms = [np.asarray(data[k], dtype=np.float32) for k in gms_keys]

    feats: dict[str, np.ndarray] = {
        "gradient_magnitude": grad,
        "local_variance": var,
    }
    for k, arr in zip(gms_keys, gms, strict=True):
        feats[k] = arr

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        return _compute_rigidity_masked_scaling(feats, mask=m, **kwargs)
    return compute_rigidity_map(feats, **kwargs)


def _percentile_scale01_masked(
    x: np.ndarray,
    mask: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    eps: float = 1e-12,
) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)[mask]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    pl, ph = np.percentile(v, (lo_pct, hi_pct))
    out = (np.asarray(x, dtype=np.float32) - np.float32(pl)) / np.float32(ph - pl + 1e-6)
    return np.clip(out, 0.0, 1.0)


def _compute_rigidity_masked_scaling(
    features: dict[str, np.ndarray],
    *,
    mask: np.ndarray,
    w_gradient: float = 1.0 / 3.0,
    w_variance: float = 1.0 / 3.0,
    w_consistency: float = 1.0 / 3.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Like :func:`compute_rigidity_map` but percentiles/z-scores use ``mask`` only."""
    m = np.asarray(mask, dtype=bool)
    grad = np.asarray(features["gradient_magnitude"], dtype=np.float32)
    var = np.asarray(features["local_variance"], dtype=np.float32)
    g_n = _percentile_scale01_masked(grad, m, eps=eps)
    v_n = _percentile_scale01_masked(var, m, eps=eps)
    grad_term = np.float32(1.0) - g_n
    var_term = np.float32(1.0) - v_n

    gms = _per_scale_gradient_stack(features)
    if len(gms) >= 2:
        teps = np.float32(1e-6)
        normed: list[np.ndarray] = []
        for arr in gms:
            a = np.asarray(arr, dtype=np.float32)
            v = a[m]
            s = float(v.std(dtype=np.float64))
            mu = float(v.mean(dtype=np.float64))
            normed.append(((a - np.float32(mu)) / np.float32(s + teps)).astype(np.float32))
        stack = np.stack(normed, axis=-1)
        sd = np.std(stack, axis=-1, dtype=np.float64).astype(np.float32)
        cons_term = np.clip(np.float32(1.0) / (np.float32(1.0) + sd), 0.0, 1.0)
    else:
        cons_term = np.ones_like(grad_term, dtype=np.float32)

    wg, wv, wc = w_gradient, w_variance, w_consistency
    ws = wg + wv + wc
    if ws <= 0.0:
        raise ValueError("rigidity weights must sum to a positive value")
    wg, wv, wc = wg / ws, wv / ws, wc / ws
    return (wg * grad_term + wv * var_term + wc * cons_term).astype(np.float32, copy=False)
