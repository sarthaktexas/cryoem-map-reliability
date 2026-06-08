"""Tight bounding-box crop around a contour mask to avoid full-grid compute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VolumeBbox:
    """Half-open index ranges ``[z0:z1, y0:y1, x0:x1]`` on a (Z, Y, X) volume."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.z0, self.z1), slice(self.y0, self.y1), slice(self.x0, self.x1))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    @property
    def n_voxels(self) -> int:
        nz, ny, nx = self.shape
        return int(nz * ny * nx)


def pad_voxels_for_filters(
    *,
    window: int = 5,
    gaussian_sigmas: Sequence[float] | None = None,
    gradient_pad: int = 1,
) -> int:
    """
    Halo width (voxels) so windowed / Gaussian filters match a full-grid pass.

    Uses ``3 * max(sigma)`` for Gaussians (scipy ``mode='nearest'``) and
    ``window // 2`` for uniform windows.
    """
    pad = max(int(window) // 2, int(gradient_pad))
    if gaussian_sigmas:
        pad = max(pad, int(np.ceil(3.0 * float(max(gaussian_sigmas)))))
    return pad


def bbox_from_mask(mask: np.ndarray, *, pad: int = 0) -> VolumeBbox:
    """Tight bbox of ``True`` voxels, expanded by ``pad`` and clipped to the volume."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {m.shape}")
    if not m.any():
        raise ValueError("Cannot build bbox from an empty mask")
    coords = np.argwhere(m)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    nz, ny, nx = m.shape
    p = int(pad)
    return VolumeBbox(
        z0=max(0, int(z0) - p),
        z1=min(nz, int(z1) + p),
        y0=max(0, int(y0) - p),
        y1=min(ny, int(y1) + p),
        x0=max(0, int(x0) - p),
        x1=min(nx, int(x1) + p),
    )


def bbox_from_contour(
    density: np.ndarray,
    contour: float,
    *,
    pad: int = 0,
) -> VolumeBbox:
    """Build a crop box from ``density >= contour``."""
    from .analysis import build_contour_mask

    return bbox_from_mask(build_contour_mask(density, contour), pad=pad)


def crop_array(arr: np.ndarray, bbox: VolumeBbox) -> np.ndarray:
    """Return ``arr[bbox]`` as a view or copy."""
    return np.asarray(arr)[bbox.slices]


def embed_array(
    full_shape: tuple[int, int, int],
    bbox: VolumeBbox,
    cropped: np.ndarray,
    *,
    fill: float | int = 0,
    dtype: type[np.float32] | type[np.float64] | type[np.uint8] | None = None,
) -> np.ndarray:
    """Scatter ``cropped`` into a full-sized volume; outside the bbox stays ``fill``."""
    dt = dtype or cropped.dtype
    out = np.full(full_shape, fill, dtype=dt)
    out[bbox.slices] = np.asarray(cropped, dtype=dt)
    return out


def format_bbox_log(bbox: VolumeBbox, full_shape: tuple[int, int, int], *, pad: int) -> str:
    """One-line summary for pipeline logs."""
    full_n = int(np.prod(full_shape))
    frac = 100.0 * bbox.n_voxels / max(1, full_n)
    return (
        f"bbox {bbox.shape[0]}×{bbox.shape[1]}×{bbox.shape[2]} "
        f"of {full_shape[0]}×{full_shape[1]}×{full_shape[2]} "
        f"({bbox.n_voxels:,}/{full_n:,} voxels, {frac:.1f}%, pad={pad})"
    )
