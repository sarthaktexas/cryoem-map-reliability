"""Tests for cryoem_mrc.hessian."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.hessian import density_hessian_scalar_maps


def test_hessian_quadratic_peak_positive_curvature() -> None:
    n = 24
    z, y, x = np.ogrid[:n, :n, :n]
    c = (n - 1) / 2.0
    rho = -((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2).astype(np.float64)
    maps = density_hessian_scalar_maps(rho, chunk_z=12)
    center = maps["hessian_eig_max"][n // 2, n // 2, n // 2]
    assert center < 0.0


def test_hessian_keys_and_shape() -> None:
    rho = np.random.default_rng(0).standard_normal((20, 22, 18)).astype(np.float32)
    maps = density_hessian_scalar_maps(rho, chunk_z=8)
    assert maps["hessian_trace"].shape == rho.shape
    assert np.isfinite(maps["hessian_frobenius"]).all()
