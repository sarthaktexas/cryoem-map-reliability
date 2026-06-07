"""Orchestrate loading, feature extraction, save/load, and optional visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import NormalizationMode, apply_start_threshold, load_mrc, normalize_density
from .local_stats import gradient_magnitude, local_mean_and_variance
from .multiscale import gaussian_multiscale_features
from .reliability import attach_reliability_to_features
from .rigidity import compute_rigidity_map
from .visualize import plot_feature_slices, plot_rigidity_inspection


def run_pipeline(
    mrc_path: str | Path,
    *,
    normalization: NormalizationMode = "zscore",
    start_threshold: float | None = None,
    local_window: int = 5,
    gaussian_sigmas: tuple[float, ...] | list[float] | None = None,
    use_float32: bool = False,
    compute_rigidity: bool = False,
    rigidity_weights: tuple[float, float, float] | None = None,
    half1_path: str | Path | None = None,
    half2_path: str | Path | None = None,
    reliability_mask: np.ndarray | None = None,
    compute_reliability: bool = True,
    plot: bool = False,
    plot_keys: list[str] | None = None,
    plot_save: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Load MRC, normalize, compute local stats, gradient, and multi-scale Gaussians.

    If ``start_threshold`` is set, voxels with raw intensity below it are set to 0
    before normalization and feature extraction (the saved ``density_raw`` is still
    the unmodified map from disk).

    Multi-scale: for each sigma, the map is Gaussian-smoothed, then local variance
    (same ``local_window`` as the base map) and gradient magnitude are computed on
    that smoothed volume so you can compare stability across scales. Flat arrays are
    keyed for ``np.savez``; use :func:`cryoem_mrc.multiscale.group_multiscale_features`
    for a nested view.

    Returns a dict of 3D float arrays (Z, Y, X), including:
    - density_raw, density_normalized
    - local_mean, local_variance, gradient_magnitude (on normalized density)
    - multiscale_sigmas (1D, length k)
    - gauss_s{i}, gauss_s{i}_local_variance, gauss_s{i}_gradient_magnitude
    - reliability_score, reliability_H_repro, build_zone (when half-maps supplied)
    - rigidity (optional legacy heuristic; off by default; enable with
      ``compute_rigidity=True``)

    When ``plot`` is True and ``plot_keys`` is None, an inspection figure is shown
    (density, local variance, gradient, one mid-scale gradient, rigidity).

    Set ``use_float32=True`` for large maps (e.g. 10⁸ voxels) to roughly halve memory
    and speed up filtering and disk IO versus float64.
    """
    dt = np.float32 if use_float32 else np.float64
    raw = load_mrc(mrc_path, dtype=dt)
    volume_for_features = apply_start_threshold(raw, start_threshold)
    normed = normalize_density(volume_for_features, mode=normalization)

    mean, var = local_mean_and_variance(normed, size=local_window)
    grad = gradient_magnitude(normed)
    multiscale = gaussian_multiscale_features(
        normed,
        sigmas=gaussian_sigmas,
        local_window=local_window,
    )

    features: dict[str, np.ndarray] = {
        "density_raw": raw,
        "density_normalized": normed,
        "local_mean": mean,
        "local_variance": var,
        "gradient_magnitude": grad,
        **multiscale,
    }

    if compute_rigidity:
        wg = wc = wv = 1.0 / 3.0
        if rigidity_weights is not None:
            wg, wv, wc = rigidity_weights
        features["rigidity"] = compute_rigidity_map(
            features,
            w_gradient=wg,
            w_variance=wv,
            w_consistency=wc,
        )

    if compute_reliability and half1_path is not None and half2_path is not None:
        h1 = load_mrc(half1_path, dtype=dt)
        h2 = load_mrc(half2_path, dtype=dt)
        attach_reliability_to_features(
            features,
            h1,
            h2,
            window=local_window,
            mask=reliability_mask,
            compute_zones=reliability_mask is not None,
        )

    if plot:
        if plot_keys is None:
            plot_rigidity_inspection(
                features,
                save_path=plot_save,
                show=plot_save is None,
            )
        else:
            cmap_overrides = (
                {"rigidity": "viridis"} if "rigidity" in plot_keys else None
            )
            plot_feature_slices(
                features,
                keys=plot_keys,
                cmap_overrides=cmap_overrides,
                save_path=plot_save,
                show=plot_save is None,
            )

    return features


def save_feature_maps(
    features: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    compressed: bool = True,
) -> None:
    """Save all feature arrays to a single .npz file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {k: np.asarray(v) for k, v in features.items()}
    if compressed:
        np.savez_compressed(out_path, **kwargs)
    else:
        np.savez(out_path, **kwargs)


def load_feature_maps(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load feature maps from .npz written by save_feature_maps."""
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_feature_maps_npy(features: dict[str, np.ndarray], out_dir: str | Path) -> None:
    """Save each feature as ``<key>.npy`` under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in features.items():
        safe = key.replace("/", "_")
        np.save(out_dir / f"{safe}.npy", np.asarray(arr))
