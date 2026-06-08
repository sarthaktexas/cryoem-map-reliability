"""Matplotlib slice views of 3D feature maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from style.nature import apply, label_panel, savefig as save_nature


def rigidity_inspection_keys(features: dict[str, np.ndarray]) -> list[str]:
    """
    Curated keys for comparing inputs to the rigidity map on the same slice.

    Order: normalized density, local variance, base gradient, one mid-scale
    gradient, rigidity (when present).
    """
    keys: list[str] = []
    for k in ("density_normalized", "local_variance", "gradient_magnitude"):
        if k in features:
            keys.append(k)
    gm_scales: list[tuple[int, str]] = []
    for name in features:
        if name.startswith("gauss_s") and name.endswith("_gradient_magnitude"):
            parts = name.removeprefix("gauss_s").removesuffix("_gradient_magnitude")
            if parts.isdigit():
                gm_scales.append((int(parts), name))
    if gm_scales:
        gm_scales.sort(key=lambda t: t[0])
        mid = gm_scales[len(gm_scales) // 2][1]
        keys.append(mid)
    if "rigidity" in features:
        keys.append("rigidity")
    return keys if keys else list(features.keys())[: min(6, len(features))]


def plot_rigidity_inspection(
    feature_maps: dict[str, np.ndarray],
    *,
    slice_axis: int = 0,
    slice_index: int | None = None,
    figsize_per: tuple[float, float] = (4.0, 3.5),
    cmap: str = "gray",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Same slice across density, local stats, one multi-scale gradient, and ``rigidity``."""
    keys = rigidity_inspection_keys(feature_maps)
    return plot_feature_slices(
        feature_maps,
        keys=keys,
        slice_axis=slice_axis,
        slice_index=slice_index,
        figsize_per=figsize_per,
        cmap=cmap,
        cmap_overrides={"rigidity": "viridis"},
        save_path=save_path,
        show=show,
    )


def plot_feature_slices(
    feature_maps: dict[str, np.ndarray],
    keys: Sequence[str] | None = None,
    slice_axis: int = 0,
    slice_index: int | None = None,
    figsize_per: tuple[float, float] = (4.0, 3.5),
    cmap: str = "gray",
    cmap_overrides: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a single 2D slice (same index) for each selected 3D feature map.

    slice_axis: 0=z, 1=y, 2=x in (Z, Y, X) convention.
    slice_index: defaults to center along that axis.
    cmap_overrides: optional per-key colormap names (e.g. ``{"rigidity": "viridis"}``).
    """
    if keys is None:
        keys = rigidity_inspection_keys(feature_maps)
        if not keys:
            keys = list(feature_maps.keys())
    keys = list(keys)
    if not keys:
        raise ValueError("No keys to plot.")

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = np.array([axes])

    overrides = cmap_overrides or {}
    for i, (ax, name) in enumerate(zip(axes, keys)):
        apply(ax)
        vol = np.asarray(feature_maps[name])
        if vol.ndim != 3:
            raise ValueError(f"{name}: expected 3D array, got {vol.shape}")
        idx = slice_index if slice_index is not None else vol.shape[slice_axis] // 2
        if slice_axis == 0:
            sl = vol[idx, :, :]
        elif slice_axis == 1:
            sl = vol[:, idx, :]
        else:
            sl = vol[:, :, idx]
        cm = overrides.get(name, cmap)
        im = ax.imshow(sl, cmap=cm, origin="lower")
        ax.set_title(name)
        if slice_axis == 0:
            ax.set_xlabel("X (columns)")
            ax.set_ylabel("Y (rows)")
        elif slice_axis == 1:
            ax.set_xlabel("X (columns)")
            ax.set_ylabel("Z (rows)")
        else:
            ax.set_xlabel("Y (columns)")
            ax.set_ylabel("Z (rows)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, ax in enumerate(axes):
        label_panel(ax, chr(ord("a") + i))

    fig.tight_layout()
    if save_path is not None:
        save_nature(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_central_orthogonal_slices(
    volume: np.ndarray,
    *,
    cmap: str = "gray",
    figsize: tuple[float, float] = (9.0, 3.2),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Three orthogonal slices through the volume center (Z, Y, X indexing)."""
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {vol.shape}")
    nz, ny, nx = vol.shape
    iz, iy, ix = nz // 2, ny // 2, nx // 2
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    im0 = axes[0].imshow(vol[iz], cmap=cmap, origin="lower")
    apply(axes[0])
    axes[0].set_title(f"XY plane (z = {iz})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    im1 = axes[1].imshow(vol[:, iy, :], cmap=cmap, origin="lower")
    apply(axes[1])
    axes[1].set_title(f"XZ plane (y = {iy})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
    im2 = axes[2].imshow(vol[:, :, ix], cmap=cmap, origin="lower")
    apply(axes[2])
    axes[2].set_title(f"YZ plane (x = {ix})")
    axes[2].set_xlabel("y"); axes[2].set_ylabel("z")
    for letter, ax, im in zip("abc", axes, (im0, im1, im2)):
        label_panel(ax, letter)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        save_nature(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_volume_histogram(
    volume: np.ndarray,
    *,
    bins: int = 120,
    max_samples: int = 2_000_000,
    percentile_clip: tuple[float, float] | None = (0.05, 99.95),
    density: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Histogram of voxel intensities with optional robust axis clipping."""
    rng = np.random.default_rng(0)
    v = np.asarray(volume).ravel()
    if v.size > max_samples:
        v = v[rng.choice(v.size, size=max_samples, replace=False)]
    v = v[np.isfinite(v)]
    if percentile_clip is not None:
        lo, hi = np.percentile(v, percentile_clip)
        v = v[(v >= lo) & (v <= hi)]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    apply(ax)
    ax.hist(v, bins=bins, density=density, color="gray", edgecolor="none", alpha=0.9)
    ax.set_xlabel("intensity")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title or "Voxel intensity histogram")
    fig.tight_layout()
    if save_path is not None:
        save_nature(fig, save_path)
    if show:
        plt.show()
    return fig
