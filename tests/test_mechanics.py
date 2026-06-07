"""Tests for cryoem_mrc.mechanics."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.mechanics import (
    compute_mechanics_maps,
    euler_lagrange_residual,
    fluctuation_constraint_decomposition,
    halfmap_hamiltonian,
    rigidity_like_from_balance,
    rigidity_like_from_energy,
)


def _smooth_blob(n: int = 32) -> np.ndarray:
    z, y, x = np.ogrid[:n, :n, :n]
    c = (n - 1) / 2.0
    r2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return np.exp(-r2 / (2 * 4.0**2)).astype(np.float32)


def test_fluctuation_constraint_L_equals_T_minus_V() -> None:
    rho = _smooth_blob()
    delta = 0.05 * np.random.default_rng(0).standard_normal(rho.shape).astype(np.float32)
    maps = fluctuation_constraint_decomposition(rho, delta, window=1)
    t = maps["fluctuation_T"].astype(np.float64)
    v = maps["constraint_V"].astype(np.float64)
    l = maps["L_balance"].astype(np.float64)
    np.testing.assert_allclose(l, t - v, rtol=1e-5, atol=1e-6)


def test_L_more_negative_when_halves_agree_on_structured_density() -> None:
    rho = _smooth_blob()
    delta_zero = np.zeros_like(rho)
    delta_noisy = np.random.default_rng(1).standard_normal(rho.shape).astype(np.float32) * 0.2
    l_agree = float(
        fluctuation_constraint_decomposition(rho, delta_zero, window=5)["L_balance"].mean()
    )
    l_disagree = float(
        fluctuation_constraint_decomposition(rho, delta_noisy, window=5)["L_balance"].mean()
    )
    assert l_agree < l_disagree


def test_halfmap_hamiltonian_exports_L_and_H() -> None:
    rho = _smooth_blob()
    delta = np.zeros_like(rho)
    h = halfmap_hamiltonian(rho, delta, window=5)
    assert "L_balance" in h
    assert "H_repro" in h
    np.testing.assert_allclose(
        h["L_balance"].astype(np.float64),
        h["fluctuation_T"].astype(np.float64) - h["constraint_V"].astype(np.float64),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        h["H_repro"].astype(np.float64),
        h["fluctuation_T"].astype(np.float64) + h["constraint_V"].astype(np.float64),
        rtol=1e-5,
        atol=1e-6,
    )


def test_rigidity_like_balance_high_when_V_dominates() -> None:
    l = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    r = rigidity_like_from_balance(l)
    assert r[0] > r[1] > r[2]


def test_rigidity_like_monotone_decreasing_in_energy() -> None:
    e = np.array([0.0, 1.0, 10.0], dtype=np.float32)
    r = rigidity_like_from_energy(e)
    assert r[0] > r[1] > r[2]


def test_compute_mechanics_maps_keys_and_shape() -> None:
    rho = _smooth_blob()
    delta = 0.01 * np.random.default_rng(0).standard_normal(rho.shape).astype(np.float32)
    maps = compute_mechanics_maps(rho, delta, window=5)
    assert maps["L_balance"].shape == rho.shape
    assert maps["fluctuation_T"].shape == rho.shape
    assert "H_repro" in maps
    assert "rigidity_like_L" in maps
    assert np.isfinite(maps["L_balance"]).all()


def test_el_residual_finite_on_blob() -> None:
    rho = _smooth_blob()
    r = euler_lagrange_residual(rho, window=5)
    assert np.isfinite(r).all()
