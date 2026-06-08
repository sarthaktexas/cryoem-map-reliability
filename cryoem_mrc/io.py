"""Load, normalize, and persist MRC volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mrcfile
import numpy as np


NormalizationMode = Literal["zscore", "minmax", "percentile"]


def load_mrc(
    path: str | Path,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> np.ndarray:
    """Load an MRC file and return a 3D float array (Z, Y, X).

    Use ``dtype=np.float32`` for large boxes (~10⁸ voxels) to halve RAM and speed IO.
    """
    path = Path(path)
    mrc = mrcfile.open(path)
    if dtype not in (np.float32, np.float64):
        raise TypeError("dtype must be np.float32 or np.float64")
    data = np.asarray(mrc.data, dtype=dtype)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume, got shape {data.shape}")
    return data


def apply_start_threshold(volume: np.ndarray, threshold: float | None) -> np.ndarray:
    """
    Optionally zero voxels below a raw-intensity threshold before further processing.

    When ``threshold`` is None, returns ``volume`` unchanged (same dtype/shape semantics
    as ``np.asarray``). Otherwise returns a new array where values strictly below
    ``threshold`` are set to 0.
    """
    v = np.asarray(volume)
    if threshold is None:
        return v
    z = np.zeros_like(v)
    return np.where(v >= threshold, v, z)


def normalize_density(
    volume: np.ndarray,
    mode: NormalizationMode = "zscore",
    eps: float = 1e-12,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
) -> np.ndarray:
    """
    Normalize voxel intensities in place on a copy.

    - zscore: (x - mean) / std
    - minmax: linear scale to [0, 1] using min/max
    - percentile: clip to [p_lo, p_hi] then minmax to [0, 1]
    """
    v = np.asarray(volume)
    dt: type[np.float32] | type[np.float64] = (
        np.float64 if v.dtype == np.float64 else np.float32
    )
    v = v.astype(dt, copy=True)
    teps = np.float32(1e-6) if dt == np.float32 else eps
    if mode == "zscore":
        m = float(np.mean(v))
        s = float(np.std(v))
        return ((v - m) / (s + teps)).astype(dt, copy=False)
    if mode == "minmax":
        lo, hi = float(np.min(v)), float(np.max(v))
        return ((v - lo) / (hi - lo + teps)).astype(dt, copy=False)
    if mode == "percentile":
        pl = np.percentile(v, percentile_lo)
        ph = np.percentile(v, percentile_hi)
        v = np.clip(v, pl, ph)
        return ((v - pl) / (ph - pl + teps)).astype(dt, copy=False)
    raise ValueError(f"Unknown normalization mode: {mode}")


def save_volume_like_reference(
    reference_path: str | Path,
    volume: np.ndarray,
    out_path: str | Path,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float32,
    extra_label: str | None = None,
) -> None:
    """
    Write a 3D array as MRC using the reference file's header (cell, origin, sym, …).

    Reads only the reference header (not the full map), then writes ``out_path``
    with the same grid as ``reference_path`` so volume overlays superpose
    the volume on the original map. ``volume`` must be shaped ``(Z, Y, X)`` like
    :func:`load_mrc`.

    ``dtype`` defaults to float32, which is typical for derived maps and keeps
    files half the size of float64.
    """
    if dtype not in (np.float32, np.float64):
        raise TypeError("dtype must be np.float32 or np.float64")
    reference_path = Path(reference_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vol = np.asarray(volume)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume, got shape {vol.shape}")

    with mrcfile.open(reference_path, header_only=True) as ref:
        hdr = ref.header.copy()
        ext = np.copy(ref.extended_header)

    expected = (int(hdr.nz), int(hdr.ny), int(hdr.nx))
    if vol.shape != expected:
        raise ValueError(
            f"volume shape {vol.shape} (Z,Y,X) does not match reference "
            f"{reference_path} dimensions {expected}"
        )

    to_write = np.ascontiguousarray(vol, dtype=dtype)
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.header[()] = hdr[()]
        mrc.set_extended_header(ext)
        mrc.set_data(to_write)
        if extra_label:
            try:
                mrc.add_label(extra_label[:80])
            except IndexError:
                # Reference already has 10 labels; keep file valid without appending
                pass


def save_rigidity_mrc(
    reference_path: str | Path,
    rigidity: np.ndarray,
    out_path: str | Path | None = None,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float32,
    extra_label: str | None = "rigidity (cryoem_mrc)",
) -> Path:
    """
    Write the rigidity map to ``<reference_stem>_rigidity.mrc`` (or ``out_path``).

    Same grid and header geometry as ``reference_path``.
    """
    reference_path = Path(reference_path)
    if out_path is None:
        out_path = reference_path.with_name(f"{reference_path.stem}_rigidity.mrc")
    else:
        out_path = Path(out_path)
    save_volume_like_reference(
        reference_path,
        rigidity,
        out_path,
        dtype=dtype,
        extra_label=extra_label,
    )
    return out_path
