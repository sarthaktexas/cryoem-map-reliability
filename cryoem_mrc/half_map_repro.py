"""Local reproducibility metrics between two cryo-EM half-maps on the same grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .io import save_volume_like_reference


def _uf(x: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(x.astype(np.float64), size=size, mode="nearest")


def half_map_local_metrics(
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    window: int = 5,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Per-voxel neighborhood statistics (cubic window of side ``window``).

    Returns (Z, Y, X) arrays:

    - ``local_cross_correlation``: Pearson correlation between the two halves
    - ``local_mean_squared_difference``: mean of ``(h1 - h2)²``
    - ``local_variance_difference``: variance of ``(h1 - h2)``
    - ``local_reproducibility_snr``: ``0.5 * (|mean(h1)| + |mean(h2)|) / (std(h1-h2) + eps)``
      where mean/std are local to the same window (dimensionless, not dB).
    """
    if half1.shape != half2.shape:
        raise ValueError(f"Shape mismatch: {half1.shape} vs {half2.shape}")
    a = np.asarray(half1, dtype=np.float64)
    b = np.asarray(half2, dtype=np.float64)
    w = int(window)
    if w < 1:
        raise ValueError("window must be positive")

    m1 = _uf(a, w)
    m2 = _uf(b, w)
    m1sq = _uf(a * a, w)
    m2sq = _uf(b * b, w)
    mab = _uf(a * b, w)
    v1 = np.maximum(m1sq - m1 * m1, 0.0)
    v2 = np.maximum(m2sq - m2 * m2, 0.0)
    cov = mab - m1 * m2
    denom = np.sqrt(v1 * v2) + eps
    local_cc = cov / denom

    diff = a - b
    local_mse = _uf(diff * diff, w)
    md = _uf(diff, w)
    local_var_diff = np.maximum(_uf(diff * diff, w) - md * md, 0.0)
    std_d = np.sqrt(local_var_diff + eps)
    combined = 0.5 * (np.abs(m1) + np.abs(m2))
    snr_like = combined / (std_d + eps)

    dt = np.result_type(half1.dtype, half2.dtype)
    return {
        "local_cross_correlation": local_cc.astype(dt, copy=False),
        "local_mean_squared_difference": local_mse.astype(dt, copy=False),
        "local_variance_difference": local_var_diff.astype(dt, copy=False),
        "local_reproducibility_snr": snr_like.astype(dt, copy=False),
    }


def save_half_map_metrics_mrc(
    metrics: dict[str, np.ndarray],
    reference_mrc_path: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "half_repro_",
) -> dict[str, Path]:
    """
    Write each metric volume as MRC on the same grid as ``reference_mrc_path``.

    Returns a map from metric name to output path.
    """
    ref = Path(reference_mrc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, vol in metrics.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = out_dir / f"{prefix}{safe}.mrc"
        save_volume_like_reference(ref, vol, path, extra_label=name[:80])
        written[name] = path
    return written


def plot_half_map_metric_distributions(
    metrics: dict[str, np.ndarray],
    *,
    max_samples: int = 500_000,
    bins: int = 80,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Histogram each metric (uniform random subsample of voxels for large maps).
    """
    rng = np.random.default_rng(0)
    names = list(metrics.keys())
    n = len(names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_ax = axes.ravel()
    for i, name in enumerate(names):
        ax = flat_ax[i]
        v = np.asarray(metrics[name]).ravel()
        if v.size > max_samples:
            idx = rng.choice(v.size, size=max_samples, replace=False)
            v = v[idx]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, color="steelblue", alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel("density")
    for j in range(len(names), len(flat_ax)):
        flat_ax[j].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
