"""Hessian-derived scalar maps on cryo-EM density (local curvature / stiffness proxies).

The density ρ is treated as a scalar potential; at each voxel we form the 3×3
Hessian of second partial derivatives and summarize its eigenvalues.

See docs/THESIS_AND_PUBLICATION.md (gradient-constraint decomposition).
"""

from __future__ import annotations

import numpy as np


def _symmetric_hessian_from_gradients(
    gzz: np.ndarray,
    gzy: np.ndarray,
    gzx: np.ndarray,
    gyz: np.ndarray,
    gyy: np.ndarray,
    gyx: np.ndarray,
    gxz: np.ndarray,
    gxy: np.ndarray,
    gxx: np.ndarray,
) -> np.ndarray:
    """Stack a symmetric 3×3 Hessian for every voxel → shape (..., 3, 3)."""
    h00 = gzz
    h11 = gyy
    h22 = gxx
    h01 = 0.5 * (gzy + gyz)
    h02 = 0.5 * (gzx + gxz)
    h12 = 0.5 * (gyx + gxy)
    return np.stack(
        [
            np.stack([h00, h01, h02], axis=-1),
            np.stack([h01, h11, h12], axis=-1),
            np.stack([h02, h12, h22], axis=-1),
        ],
        axis=-2,
    )


def _scalar_summaries_from_hessian(h: np.ndarray, *, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Eigenvalue summaries from Hessian stack (..., 3, 3)."""
    evals = np.linalg.eigh(h)[0]
    lam1 = evals[..., 2]
    lam2 = evals[..., 1]
    lam3 = evals[..., 0]
    trace = lam1 + lam2 + lam3
    frob = np.sqrt(np.sum(h * h, axis=(-2, -1)))
    det = np.linalg.det(h)
    aniso = (lam1 - lam3) / (np.abs(lam1) + np.abs(lam3) + eps)
    return {
        "hessian_trace": trace,
        "hessian_frobenius": frob,
        "hessian_determinant": det,
        "hessian_eig_max": lam1,
        "hessian_eig_mid": lam2,
        "hessian_eig_min": lam3,
        "hessian_anisotropy": aniso,
    }


def density_hessian_scalar_maps(
    rho: np.ndarray,
    *,
    chunk_z: int | None = 64,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Scalar Hessian summaries of normalized density ρ.

    Returns maps aligned with ``rho`` (Z, Y, X): trace (≈ Laplacian), Frobenius
    norm, determinant, eigenvalues, and an anisotropy index (λ_max − λ_min).
    """
    v = np.asarray(rho, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {v.shape}")
    dt = rho.dtype

    def _summarize_block(block: np.ndarray) -> dict[str, np.ndarray]:
        gz, gy, gx = np.gradient(block)
        gzz, gzy, gzx = np.gradient(gz)
        gyz, gyy, gyx = np.gradient(gy)
        gxz, gxy, gxx = np.gradient(gx)
        h = _symmetric_hessian_from_gradients(gzz, gzy, gzx, gyz, gyy, gyx, gxz, gxy, gxx)
        return _scalar_summaries_from_hessian(h, eps=eps)

    if chunk_z is None:
        raw = _summarize_block(v)
        return {k: np.asarray(arr, dtype=dt) for k, arr in raw.items()}

    nz, ny, nx = v.shape
    pad = 3
    tmpl = _summarize_block(v[: min(nz, chunk_z + 2 * pad), :, :])
    out: dict[str, np.ndarray] = {
        k: np.empty((nz, ny, nx), dtype=dt) for k in tmpl
    }

    z0 = 0
    while z0 < nz:
        z1 = min(nz, z0 + chunk_z)
        za = max(0, z0 - pad)
        zb = min(nz, z1 + pad)
        part = _summarize_block(v[za:zb])
        off = z0 - za
        take = z1 - z0
        for k in out:
            out[k][z0:z1] = part[k][off : off + take].astype(dt, copy=False)
        z0 = z1

    return out
