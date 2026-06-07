"""Multi-scale Gaussian smoothing of 3D volumes and per-scale features."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import ndimage

from .local_stats import gradient_magnitude, local_variance


def _default_sigmas() -> tuple[float, float, float, float, float]:
    return (0.5, 1.0, 2.0, 4.0, 8.0)


def _match_dtype(volume: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=volume.dtype)


def gaussian_multiscale_features(
    volume: np.ndarray,
    sigmas: tuple[float, ...] | list[float] | None = None,
    *,
    local_window: int = 5,
) -> dict[str, np.ndarray]:
    """
    Gaussian-smooth ``volume`` at each sigma (voxel units), then for each smoothed map:

    - ``gauss_s{i}``: smoothed density
    - ``gauss_s{i}_local_variance``: local variance on the smoothed map
    - ``gauss_s{i}_gradient_magnitude``: gradient magnitude on the smoothed map

    Also stores ``multiscale_sigmas`` (shape ``(k,)``) so bundles can be reconstructed
    after ``np.savez`` / ``np.load``.

    Default uses five scales; pass 3–5 sigmas to match your experiment.
    """
    if sigmas is None:
        sigmas = _default_sigmas()
    if len(sigmas) < 3 or len(sigmas) > 5:
        raise ValueError("Provide between 3 and 5 Gaussian sigma values.")
    sig = np.asarray(sigmas, dtype=np.float64)
    out: dict[str, np.ndarray] = {"multiscale_sigmas": sig}
    v = np.asarray(volume)
    for i, s in enumerate(sigmas):
        smoothed = ndimage.gaussian_filter(v, sigma=float(s), mode="nearest")
        smoothed = _match_dtype(v, smoothed)
        out[f"gauss_s{i}"] = smoothed
        out[f"gauss_s{i}_local_variance"] = local_variance(smoothed, size=local_window)
        out[f"gauss_s{i}_gradient_magnitude"] = gradient_magnitude(smoothed)
    return out


def group_multiscale_features(features: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """
    Re-group flat pipeline / ``.npz`` keys into a nested structure for scale-wise work::

        {
            "sigmas": np.ndarray,  # shape (k,)
            "scales": {
                0: {"sigma": float, "smoothed", "local_variance", "gradient_magnitude"},
                ...
            },
        }

    Returns ``None`` if no multi-scale Gaussian keys are present. Older bundles that
    only have ``gauss_s{i}`` (no ``multiscale_sigmas`` or per-scale variance/gradient)
    are grouped with ``sigma`` set to ``nan`` where unknown, and missing arrays omitted.
    """
    keys = list(features.keys())
    gauss_idx = sorted(
        {int(m.group(1)) for k in keys if (m := re.fullmatch(r"gauss_s(\d+)", k))}
    )
    if not gauss_idx:
        return None

    if "multiscale_sigmas" in features:
        sigmas = np.asarray(features["multiscale_sigmas"], dtype=np.float64).ravel()
    else:
        sigmas = np.full(len(gauss_idx), np.nan, dtype=np.float64)

    scales: dict[int, dict[str, Any]] = {}
    for i in gauss_idx:
        prefix = f"gauss_s{i}"
        block: dict[str, Any] = {}
        if i < len(sigmas) and np.isfinite(sigmas[i]):
            block["sigma"] = float(sigmas[i])
        else:
            block["sigma"] = float("nan")
        if prefix in features:
            block["smoothed"] = features[prefix]
        lv_key = f"{prefix}_local_variance"
        gm_key = f"{prefix}_gradient_magnitude"
        if lv_key in features:
            block["local_variance"] = features[lv_key]
        if gm_key in features:
            block["gradient_magnitude"] = features[gm_key]
        scales[i] = block

    return {"sigmas": sigmas, "scales": scales}


def gaussian_scales(
    volume: np.ndarray,
    sigmas: tuple[float, ...] | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Apply Gaussian filtering at several scales (voxel units).

    Returns only ``gauss_s{i}`` smoothed maps. For variance and gradient at each
    scale, use :func:`gaussian_multiscale_features`.
    """
    if sigmas is None:
        sigmas = _default_sigmas()
    if len(sigmas) < 3 or len(sigmas) > 5:
        raise ValueError("Provide between 3 and 5 Gaussian sigma values.")
    v = np.asarray(volume)
    out: dict[str, np.ndarray] = {}
    for i, s in enumerate(sigmas):
        sm = ndimage.gaussian_filter(v, sigma=float(s), mode="nearest")
        out[f"gauss_s{i}"] = _match_dtype(v, sm)
    return out
