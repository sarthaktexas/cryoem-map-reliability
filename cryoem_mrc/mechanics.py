"""Lagrangian-inspired fluctuation-constraint balance on cryo-EM half-maps.

Primary thesis functional (see docs/THESIS_AND_PUBLICATION.md):

    T = (1/2) * delta_rho^2      fluctuation proxy (half-map disagreement)
    V = (1/2) * |grad rho|^2      constraint proxy (density gradient)
    L = T - V                     flexible-like when positive, rigid-like when negative

Optional window > 1 box-filters T and V before forming L (local summaries).

Legacy rho-only Lagrangian helpers remain for archive comparisons only.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .local_stats import gradient_magnitude, local_laplacian, local_mean


def _uf(x: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(np.asarray(x, dtype=np.float64), size=int(size), mode="nearest")


def _match_dtype(volume: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=volume.dtype)


def _maybe_window(arr: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1:
        return np.asarray(arr, dtype=np.float64)
    return _uf(arr, w)


def fluctuation_constraint_decomposition(
    rho: np.ndarray,
    delta_rho: np.ndarray,
    *,
    window: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Lagrangian-inspired fluctuation-constraint balance from averaged map and half difference.

    T = (alpha/2) * delta_rho^2
    V = (beta/2) * ||grad rho||^2
    L = T - V   (positive: fluctuation dominates; negative: gradient constraint dominates)
    H = T + V   (Hamiltonian sum; exploratory secondary scalar)

    When ``window`` > 1, T and V are box-filtered before L and H are formed.
    """
    if rho.shape != delta_rho.shape:
        raise ValueError(f"Shape mismatch: rho {rho.shape} vs delta_rho {delta_rho.shape}")
    v = np.asarray(rho)
    d = np.asarray(delta_rho, dtype=np.float64)
    a, b = float(alpha), float(beta)

    t_raw = 0.5 * a * (d * d)
    grad_sq = gradient_magnitude(v) ** 2
    v_raw = 0.5 * b * grad_sq

    t = _maybe_window(t_raw, window)
    pot = _maybe_window(v_raw, window)
    lagrangian = t - pot
    hamiltonian = t + pot

    return {
        "fluctuation_T": _match_dtype(v, t),
        "constraint_V": _match_dtype(v, pot),
        "L_balance": _match_dtype(v, lagrangian),
        "H_sum": _match_dtype(v, hamiltonian),
    }


def classify_tv_regime(
    fluctuation_t: np.ndarray,
    constraint_v: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Three-way (T, V) phase portrait relative to in-mask medians.

    Returns uint8 labels (0 outside mask):

    - 0 = featureless (both T and V below median)
    - 1 = flexible-like (T dominates: high T, low V, or T > V when both high)
    - 2 = rigid-like (V dominates: low T, high V, or V > T when both high)
    """
    t = np.asarray(fluctuation_t, dtype=np.float64)
    v = np.asarray(constraint_v, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    zones = np.zeros(t.shape, dtype=np.uint8)
    if not m.any():
        return zones

    t_med = float(np.median(t[m]))
    v_med = float(np.median(v[m]))
    t_hi = t > t_med
    v_hi = v > v_med

    featureless = m & (~t_hi) & (~v_hi)
    flexible = m & t_hi & (~v_hi)
    rigid = m & (~t_hi) & v_hi
    both_hi = m & t_hi & v_hi

    zones[featureless] = 0
    zones[flexible] = 1
    zones[rigid] = 2
    if both_hi.any():
        zones[both_hi] = np.where(t[both_hi] >= v[both_hi], 1, 2).astype(np.uint8)

    return zones


def halfmap_hamiltonian(
    rho: np.ndarray,
    delta_rho: np.ndarray,
    *,
    window: int = 5,
    sigma: float = 1.0,
    kappa: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    LH map bundle: T, V, L = T − V, H = T + V (alias: :func:`lh_map_metrics`).

    Uses :func:`fluctuation_constraint_decomposition` with
    ``alpha = 1/sigma^2``, ``beta = kappa``. Export keys retain ``H_repro`` /
    ``L_repro`` names for NPZ/MRC compatibility.
    """
    sig2 = float(sigma) ** 2 + eps
    decomp = fluctuation_constraint_decomposition(
        rho,
        delta_rho,
        window=window,
        alpha=1.0 / sig2,
        beta=float(kappa),
    )
    return {
        "H_repro_fluctuation": decomp["fluctuation_T"],
        "H_repro_smoothness": decomp["constraint_V"],
        "H_repro": decomp["H_sum"],
        "L_repro": decomp["L_balance"],
        "fluctuation_T": decomp["fluctuation_T"],
        "constraint_V": decomp["constraint_V"],
        "L_balance": decomp["L_balance"],
        "H_sum": decomp["H_sum"],
    }


# Preferred public name for the LH map bundle (T, V, L, H).
lh_map_metrics = halfmap_hamiltonian


def rigidity_like_from_balance(l_balance: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Map L = T - V to (0, 1]: higher = more rigid-like (V dominates over T)."""
    x = np.asarray(l_balance, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(np.clip(x, -50.0, 50.0)))
    return out.astype(np.asarray(l_balance).dtype, copy=False)


def rigidity_like_from_energy(energy: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Map non-negative energy to (0, 1]: higher = lower energy / more rigid-like."""
    e = np.asarray(energy, dtype=np.float64)
    out = 1.0 / (1.0 + np.maximum(e, 0.0))
    return out.astype(np.asarray(energy).dtype, copy=False)


def lagrangian_density(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
) -> dict[str, np.ndarray]:
    """
    Legacy rho-only smoothness functional (archive comparisons only).

    Not the thesis fluctuation-constraint balance; see
    :func:`fluctuation_constraint_decomposition`.
    """
    v = np.asarray(rho)
    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=window)
    dev_sq = (v - mean_w) ** 2
    a, b = float(alpha), float(beta)
    kinetic = _match_dtype(v, 0.5 * a * grad_sq)
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    total = _match_dtype(v, kinetic + potential)
    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "lagrangian_density": total,
    }


def euler_lagrange_residual(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
) -> np.ndarray:
    """
    Euler-Lagrange residual for the legacy rho-only Lagrangian (archive only).
    """
    v = np.asarray(rho)
    lap = local_laplacian(v)
    mean_w = local_mean(v, size=window)
    a, b = float(alpha), float(beta)
    return _match_dtype(v, -a * lap + b * (v - mean_w))


def hamiltonian_density(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Legacy rho-only Hamiltonian (archive comparisons only)."""
    v = np.asarray(rho)
    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=window)
    dev_sq = (v - mean_w) ** 2
    a, b = float(alpha), float(beta)
    kinetic = _match_dtype(v, 0.5 * grad_sq / (a + eps))
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    total = _match_dtype(v, kinetic + potential)
    return {
        "hamiltonian_kinetic": kinetic,
        "hamiltonian_potential": potential,
        "hamiltonian": total,
    }


def compute_mechanics_headlines(
    rho: np.ndarray,
    delta_rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    sigma: float = 1.0,
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Headline mechanics scores for rigidity comparison (minimal peak memory).

    Primary maps: fluctuation T, constraint V, balance L = T - V.
    Legacy rho-only L/H/EL retained for archive correlation tables.
    """
    v = np.asarray(rho)
    a, b = float(alpha), float(beta)
    w = int(window)
    eps = np.float32(1e-6) if v.dtype == np.float32 else 1e-12
    sig2 = float(sigma) ** 2 + eps
    kap = float(kappa)

    decomp = fluctuation_constraint_decomposition(
        v,
        delta_rho,
        window=w,
        alpha=1.0 / sig2,
        beta=kap,
    )
    l_balance = decomp["L_balance"]
    h_sum = decomp["H_sum"]

    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    del grad
    mean_w = local_mean(v, size=w)
    dev_sq = (v - mean_w) ** 2

    legacy_l = _match_dtype(v, 0.5 * a * grad_sq + 0.5 * b * dev_sq)
    legacy_h = _match_dtype(v, 0.5 * grad_sq / (a + eps) + 0.5 * b * dev_sq)
    lap = local_laplacian(v)
    el_norm = _match_dtype(v, np.abs(-a * lap + b * (v - mean_w)))
    del lap, mean_w, dev_sq, grad_sq

    return {
        "fluctuation_T": decomp["fluctuation_T"],
        "constraint_V": decomp["constraint_V"],
        "L_balance": l_balance,
        "H_sum": h_sum,
        "H_repro": h_sum,
        "L_repro": l_balance,
        "legacy_lagrangian_density": legacy_l,
        "legacy_hamiltonian": legacy_h,
        "el_residual_norm": el_norm,
        "rigidity_like_L": rigidity_like_from_balance(l_balance),
        "rigidity_like_H_repro": rigidity_like_from_balance(l_balance),
        "rigidity_like_legacy_L": rigidity_like_from_energy(legacy_l),
        "rigidity_like_legacy_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_el": rigidity_like_from_energy(el_norm),
        # Backward-compatible aliases for archive scripts
        "lagrangian_density": legacy_l,
        "hamiltonian": legacy_h,
        "rigidity_like_H": rigidity_like_from_energy(legacy_h),
    }


def compute_mechanics_maps(
    rho: np.ndarray,
    delta_rho: np.ndarray | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    sigma: float = 1.0,
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Full mechanics feature bundle for correlation analysis.

    When ``delta_rho`` is provided, returns the thesis fluctuation-constraint
    decomposition plus legacy rho-only maps for comparison.
    """
    v = np.asarray(rho)
    a, b = float(alpha), float(beta)
    w = int(window)
    eps = np.float32(1e-6) if v.dtype == np.float32 else 1e-12

    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=w)
    dev_sq = (v - mean_w) ** 2

    kinetic = _match_dtype(v, 0.5 * a * grad_sq)
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    legacy_l = _match_dtype(v, kinetic + potential)

    ham_kinetic = _match_dtype(v, 0.5 * grad_sq / (a + eps))
    ham_potential = potential
    legacy_h = _match_dtype(v, ham_kinetic + ham_potential)

    lap = local_laplacian(v)
    el = _match_dtype(v, -a * lap + b * (v - mean_w))
    el_norm = _match_dtype(v, np.abs(el.astype(np.float64)))

    out: dict[str, np.ndarray] = {
        "legacy_kinetic_energy": kinetic,
        "legacy_potential_energy": potential,
        "legacy_lagrangian_density": legacy_l,
        "legacy_hamiltonian_kinetic": ham_kinetic,
        "legacy_hamiltonian_potential": ham_potential,
        "legacy_hamiltonian": legacy_h,
        "el_residual": el,
        "el_residual_norm": el_norm,
        "rigidity_like_legacy_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_legacy_L": rigidity_like_from_energy(legacy_l),
        "rigidity_like_el": rigidity_like_from_energy(el_norm),
        "lagrangian_density": legacy_l,
        "hamiltonian": legacy_h,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "hamiltonian_kinetic": ham_kinetic,
        "hamiltonian_potential": ham_potential,
        "rigidity_like_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_L": rigidity_like_from_energy(legacy_l),
    }

    if delta_rho is not None:
        sig2 = float(sigma) ** 2 + eps
        kap = float(kappa)
        decomp = fluctuation_constraint_decomposition(
            v,
            delta_rho,
            window=w,
            alpha=1.0 / sig2,
            beta=kap,
        )
        l_balance = decomp["L_balance"]
        out["fluctuation_T"] = decomp["fluctuation_T"]
        out["constraint_V"] = decomp["constraint_V"]
        out["L_balance"] = l_balance
        out["L_repro"] = l_balance
        out["H_sum"] = decomp["H_sum"]
        out["H_repro"] = decomp["H_sum"]
        out["H_repro_fluctuation"] = decomp["fluctuation_T"]
        out["H_repro_smoothness"] = decomp["constraint_V"]
        out["rigidity_like_L"] = rigidity_like_from_balance(l_balance)
        out["rigidity_like_H_repro"] = rigidity_like_from_balance(l_balance)

    return out
