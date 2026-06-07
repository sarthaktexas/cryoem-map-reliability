"""Lightweight ML helpers (numpy-only ridge) for feature-vs-target comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score; returns (z, mean, std)."""
    xf = np.asarray(x, dtype=np.float64)
    mu = np.mean(xf, axis=0)
    sd = np.std(xf, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (xf - mu) / sd, mu, sd


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Ordinary ridge regression with intercept handled via column of ones."""
    xt = np.asarray(x_train, dtype=np.float64)
    yt = np.asarray(y_train, dtype=np.float64).reshape(-1)
    xe = np.asarray(x_test, dtype=np.float64)
    one_tr = np.ones((xt.shape[0], 1))
    one_te = np.ones((xe.shape[0], 1))
    xtr = np.hstack([one_tr, xt])
    xte = np.hstack([one_te, xe])
    p = xtr.shape[1]
    coef = np.linalg.solve(
        xtr.T @ xtr + alpha * np.eye(p),
        xtr.T @ yt,
    )
    return xte @ coef


def ridge_cv_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """
    K-fold ridge on rank-transformed target; returns best Spearman ρ on held-out folds.

    Features should already be finite and aligned row-wise with ``y``.
    """
    xf = np.asarray(x, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(xf).all(axis=1) & np.isfinite(yf)
    xf = xf[m]
    yf = yf[m]
    n = xf.shape[0]
    if n < 500:
        return {"spearman_rho": float("nan"), "best_alpha": float("nan"), "n_samples": float(n)}

    z, _, _ = standardize_features(xf)
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    folds = np.array_split(order, n_folds)

    best_rho = -np.inf
    best_alpha = alphas[0]
    for alpha in alphas:
        preds = np.empty(n, dtype=np.float64)
        for k in range(n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            preds[test_idx] = ridge_fit_predict(
                z[train_idx], yf[train_idx], z[test_idx], alpha=alpha
            )
        rho = float(stats.spearmanr(preds, yf).statistic)
        if rho > best_rho:
            best_rho = rho
            best_alpha = alpha

    return {
        "spearman_rho": float(best_rho),
        "best_alpha": float(best_alpha),
        "n_samples": float(n),
    }
