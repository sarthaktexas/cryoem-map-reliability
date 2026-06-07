"""Local mean, variance, higher moments, entropy, gradients, and structure-tensor anisotropy."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage


def _match_dtype(volume: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=volume.dtype)


def local_mean(volume: np.ndarray, size: int = 5) -> np.ndarray:
    """Voxel-wise local mean with a cubic uniform window (odd size recommended)."""
    r = ndimage.uniform_filter(volume, size=size, mode="nearest")
    return _match_dtype(volume, r)


def local_variance(volume: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Local variance via E[X^2] - E[X]^2 with the same window as local_mean.
    """
    mean = local_mean(volume, size=size)
    mean_sq = ndimage.uniform_filter(volume * volume, size=size, mode="nearest")
    var = mean_sq - mean * mean
    return _match_dtype(volume, np.maximum(var, 0.0))


def local_mean_and_variance(
    volume: np.ndarray, size: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local mean and variance together (one fewer pass over mean_sq)."""
    mean = local_mean(volume, size=size)
    mean_sq = ndimage.uniform_filter(volume * volume, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, _match_dtype(volume, var)


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude: sqrt(gz^2 + gy^2 + gx^2)."""
    gz, gy, gx = np.gradient(volume)
    mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    return _match_dtype(volume, mag)


def local_laplacian(volume: np.ndarray) -> np.ndarray:
    """Discrete Laplacian (sum of second central differences along Z, Y, X)."""
    lap = ndimage.laplace(volume, mode="nearest")
    return _match_dtype(volume, lap)


def local_skewness(volume: np.ndarray, size: int = 5, *, eps: float = 1e-12) -> np.ndarray:
    """Voxel-wise skewness from local raw moments (uniform window)."""
    v = volume.astype(np.float64, copy=False)
    m1 = ndimage.uniform_filter(v, size=size, mode="nearest")
    m2 = ndimage.uniform_filter(v * v, size=size, mode="nearest")
    m3 = ndimage.uniform_filter(v * v * v, size=size, mode="nearest")
    var = np.maximum(m2 - m1 * m1, 0.0)
    m3c = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    skew = m3c / (var**1.5 + eps)
    return _match_dtype(volume, skew)


def local_kurtosis_excess(
    volume: np.ndarray, size: int = 5, *, eps: float = 1e-12
) -> np.ndarray:
    """Excess kurtosis (Fisher) in each window: E[(X-mu)^4]/var^2 - 3."""
    v = volume.astype(np.float64, copy=False)
    m1 = ndimage.uniform_filter(v, size=size, mode="nearest")
    m2 = ndimage.uniform_filter(v * v, size=size, mode="nearest")
    m3 = ndimage.uniform_filter(v * v * v, size=size, mode="nearest")
    m4 = ndimage.uniform_filter(v * v * v * v, size=size, mode="nearest")
    var = np.maximum(m2 - m1 * m1, 0.0)
    m4c = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
    kurt = m4c / (var**2 + eps) - 3.0
    return _match_dtype(volume, kurt)


def local_entropy(
    volume: np.ndarray,
    size: int = 5,
    *,
    n_bins: int = 16,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Shannon entropy (nats) of the local histogram in a cubic window.

    The volume is globally clipped to ``[p_lo, p_hi]`` (percentiles), then quantized
    into ``n_bins`` equal-width bins so each window has a proper discrete distribution.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    v = np.asarray(volume, dtype=np.float64)
    lo, hi = np.percentile(v, (percentile_lo, percentile_hi))
    hi = max(hi, lo + eps)
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    idx = np.minimum((t * n_bins).astype(np.int32), n_bins - 1)
    ent = np.zeros_like(v)
    for k in range(n_bins):
        m = (idx == k).astype(np.float64)
        p = ndimage.uniform_filter(m, size=size, mode="nearest")
        ent -= p * np.log(p + eps)
    return _match_dtype(volume, ent)


def structure_tensor_fractional_anisotropy(
    volume: np.ndarray,
    smooth_size: int = 5,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Fractional anisotropy of the smoothed outer product of the intensity gradient.

    Builds the symmetric 3×3 structure tensor with entries ``uniform_filter(g_i g_j)``,
    then computes FA from its eigenvalues (largest to smallest).
    """
    v = volume.astype(np.float64, copy=False)
    gz, gy, gx = np.gradient(v)

    def uf(a: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(a, size=smooth_size, mode="nearest")

    jxx = uf(gx * gx)
    jyy = uf(gy * gy)
    jzz = uf(gz * gz)
    jxy = uf(gx * gy)
    jxz = uf(gx * gz)
    jyz = uf(gy * gz)
    j = np.stack(
        [
            np.stack([jxx, jxy, jxz], axis=-1),
            np.stack([jxy, jyy, jyz], axis=-1),
            np.stack([jxz, jyz, jzz], axis=-1),
        ],
        axis=-2,
    )
    evals = np.linalg.eigh(j)[0][..., ::-1]
    mean_l = np.mean(evals, axis=-1, keepdims=True)
    num = np.sum((evals - mean_l) ** 2, axis=-1)
    den = np.sum(evals**2, axis=-1) + eps
    fa = np.sqrt(1.5 * num / den)
    return _match_dtype(volume, fa)


def _pad_for_window(nz: int, z0: int, z1: int, pad: int) -> tuple[int, int, int]:
    za = max(0, z0 - pad)
    zb = min(nz, z1 + pad)
    return za, zb, z0 - za


def sliding_local_statistics_pipeline(
    volume: np.ndarray,
    window_sizes: Iterable[int] = (3, 5, 7, 9),
    *,
    entropy_bins: int = 16,
    chunk_z: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute sliding-window statistics for several cubic window widths.

    For each odd integer ``w`` in ``window_sizes``, adds keys (Z, Y, X arrays):

    - ``local_mean_w{w}``, ``local_variance_w{w}``
    - ``local_skewness_w{w}``, ``local_kurtosis_excess_w{w}``
    - ``local_entropy_w{w}``
    - ``structure_tensor_fa_w{w}`` (FA of gradients smoothed with the same ``w``)

    Always adds (window-independent):

    - ``gradient_magnitude``, ``local_laplacian``

    ``chunk_z``: if set, processes the volume in Z slabs with overlap so memory stays
    bounded (helpful for large boxes). ``None`` runs the full stack in one pass.
    """
    sizes = tuple(int(s) for s in window_sizes)
    for s in sizes:
        if s < 1:
            raise ValueError("window sizes must be positive")

    v = np.asarray(volume)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {v.shape}")

    max_w = max(sizes) if sizes else 1
    pad = (max_w // 2) + 2

    def compute_block(block: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "gradient_magnitude": gradient_magnitude(block),
            "local_laplacian": local_laplacian(block),
        }
        if not sizes:
            return out
        for w in sizes:
            mean, var = local_mean_and_variance(block, size=w)
            out[f"local_mean_w{w}"] = mean
            out[f"local_variance_w{w}"] = var
            out[f"local_skewness_w{w}"] = local_skewness(block, size=w)
            out[f"local_kurtosis_excess_w{w}"] = local_kurtosis_excess(block, size=w)
            out[f"local_entropy_w{w}"] = local_entropy(
                block, size=w, n_bins=entropy_bins
            )
            out[f"structure_tensor_fa_w{w}"] = structure_tensor_fractional_anisotropy(
                block, smooth_size=w
            )
        return out

    if chunk_z is None:
        return compute_block(v)

    nz, ny, nx = v.shape
    tmpl_z = min(nz, max(max_w + 4, 2 * pad + 4))
    template = compute_block(v[:tmpl_z, :, :])
    accum: dict[str, np.ndarray] = {
        k: np.empty((nz, ny, nx), dtype=np.asarray(template[k]).dtype) for k in template
    }

    z0 = 0
    while z0 < nz:
        z1 = min(nz, z0 + chunk_z)
        za, zb, z_off = _pad_for_window(nz, z0, z1, pad)
        slab = v[za:zb]
        part = compute_block(slab)
        take = z1 - z0
        for k in accum:
            accum[k][z0:z1] = part[k][z_off : z_off + take]
        z0 = z1

    return accum
