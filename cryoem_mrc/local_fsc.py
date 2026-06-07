"""Windowed local half-map FSC -> per-voxel resolution in Å.

Implements Cardone et al. 2013-style windowed FSC (threshold-based resolution
assignment) in pure Python. It is a **map reliability** estimator, not molecular
flexibility.

Caveats (methods section)
-------------------------
- No noise-substitution correction (Cardone et al. mask-aware FSC refinement).
- Mask awareness is approximate: only patch centers inside ``mask`` are computed;
  patches with too few masked voxels are skipped.
- ``fsc_threshold=0.143`` is the standard half-map criterion; ``0.5`` is the
  conservative half-bit criterion.
- Resolution is clipped to ``[2 * voxel_size_a, patch_size * voxel_size_a]``.
  Voxels at the upper clip mean "no measurable resolution at this patch size."
- ``patch_size`` trades localization vs. FSC stability; sensitivity at 13, 17, 25
  belongs in the thesis.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import fft
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

logger = logging.getLogger(__name__)

# Above this size (bytes per half-map), ProcessPool workers would pickle full volumes
# and can OOM the machine — use threads (shared memory) or n_jobs=1.
_LARGE_VOLUME_BYTES = 64 * 1024 * 1024

from .io import save_volume_like_reference

_FSC_EPS = 1e-20


def _build_radial_shell_indices(patch_size: int) -> tuple[np.ndarray, int]:
    """
    Precompute shell index per Fourier voxel for ``rfftn`` output and ``n_shells``.

    Shell index = ``floor(sqrt(kz^2 + ky^2 + kx^2))`` in native ``rfftn`` frequency
    layout (unshifted ``fftfreq`` on Z/Y; ``rfftfreq`` on X).
    """
    p = int(patch_size)
    if p < 3 or p % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    # Match ``rfftn`` layout: Z/Y axes are unshifted ``fftfreq`` order; X is rfft half-spectrum.
    kz_1d = np.fft.fftfreq(p) * p
    ky_1d = np.fft.fftfreq(p) * p
    kx_1d = np.fft.rfftfreq(p) * p
    kz, ky, kx = np.meshgrid(kz_1d, ky_1d, kx_1d, indexing="ij")
    r = np.floor(np.sqrt(kz * kz + ky * ky + kx * kx)).astype(np.int32)
    shell_idx = r.ravel()
    n_shells = int(shell_idx.max()) + 1
    return shell_idx, n_shells


def _build_window(patch_size: int, kind: str) -> np.ndarray:
    """3D separable window (Hann / cosine / none). Shape ``(P, P, P)``."""
    p = int(patch_size)
    kind_l = str(kind).lower()
    if kind_l == "none":
        return np.ones((p, p, p), dtype=np.float64)
    if kind_l == "hann":
        w1d = np.hanning(p)
    elif kind_l == "cosine":
        x = np.linspace(-np.pi / 2, np.pi / 2, p)
        w1d = np.cos(x) ** 2
    else:
        raise ValueError(f"Unknown window kind: {kind!r}; use 'hann', 'cosine', or 'none'")
    w = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]
    return w.astype(np.float64, copy=False)


def _fsc_curve_from_patches(
    p1: np.ndarray,
    p2: np.ndarray,
    shell_idx: np.ndarray,
    n_shells: int,
) -> np.ndarray:
    """
    1D FSC per shell from windowed real-space patches.

    ``FSC[k] = Re(sum(F1·conj(F2))) / sqrt(sum(|F1|^2) * sum(|F2|^2))`` within shell ``k``.
    """
    f1 = fft.rfftn(np.asarray(p1, dtype=np.float64))
    f2 = fft.rfftn(np.asarray(p2, dtype=np.float64))
    a = f1.ravel()
    b = f2.ravel()
    cross = np.real(a * np.conj(b))
    a2 = np.abs(a) ** 2
    b2 = np.abs(b) ** 2
    num = np.bincount(shell_idx, weights=cross, minlength=n_shells)
    s1 = np.bincount(shell_idx, weights=a2, minlength=n_shells)
    s2 = np.bincount(shell_idx, weights=b2, minlength=n_shells)
    denom = np.sqrt(s1 * s2 + _FSC_EPS)
    fsc = num / denom
    if n_shells > 0:
        fsc[0] = 1.0
    return fsc


def _resolution_from_fsc(
    fsc: np.ndarray,
    threshold: float,
    patch_size: int,
    voxel_size_a: float,
) -> float:
    """
    Smallest shell index where FSC drops below ``threshold``, with linear sub-shell
    interpolation; convert to Å via ``resolution = (patch_size * voxel_size_a) / k``.
    """
    p = int(patch_size)
    vox = float(voxel_size_a)
    worst = float(p * vox)
    fsc = np.asarray(fsc, dtype=np.float64)
    if fsc.size < 2:
        return worst
    # Skip DC shell (index 0); search for first crossing below threshold.
    for k in range(1, fsc.size):
        if fsc[k] < threshold:
            f_lo = float(fsc[k - 1])
            f_hi = float(fsc[k])
            if f_lo >= threshold and f_hi < threshold and f_lo != f_hi:
                frac = (threshold - f_lo) / (f_hi - f_lo)
                k_eff = (k - 1) + frac
            else:
                k_eff = float(k)
            k_eff = max(k_eff, 1e-6)
            res = (p * vox) / k_eff
            lo = 2.0 * vox
            hi = worst
            return float(np.clip(res, lo, hi))
    return worst


def _clip_resolution(
    res_a: float,
    *,
    voxel_size_a: float,
    patch_size: int,
) -> float:
    lo = 2.0 * float(voxel_size_a)
    hi = float(patch_size) * float(voxel_size_a)
    return float(np.clip(res_a, lo, hi))


def _patch_resolution_at_center(
    half1: np.ndarray,
    half2: np.ndarray,
    cz: int,
    cy: int,
    cx: int,
    *,
    half: int,
    window3d: np.ndarray,
    shell_idx: np.ndarray,
    n_shells: int,
    fsc_threshold: float,
    patch_size: int,
    voxel_size_a: float,
    mask: np.ndarray | None,
    min_voxels_for_fsc: int,
) -> float | None:
    """Return resolution in Å at one patch center, or None to skip."""
    z0, z1 = cz - half, cz + half + 1
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    p1 = half1[z0:z1, y0:y1, x0:x1]
    p2 = half2[z0:z1, y0:y1, x0:x1]
    if mask is not None:
        mp = mask[z0:z1, y0:y1, x0:x1]
        if int(mp.sum()) < min_voxels_for_fsc:
            return None
    w = window3d
    p1w = p1.astype(np.float64) * w
    p2w = p2.astype(np.float64) * w
    fsc = _fsc_curve_from_patches(p1w, p2w, shell_idx, n_shells)
    res = _resolution_from_fsc(fsc, fsc_threshold, patch_size, voxel_size_a)
    return _clip_resolution(res, voxel_size_a=voxel_size_a, patch_size=patch_size)


def _worker_patch_batch(
    half1: np.ndarray,
    half2: np.ndarray,
    centers: list[tuple[int, int, int]],
    *,
    half: int,
    window3d: np.ndarray,
    shell_idx: np.ndarray,
    n_shells: int,
    fsc_threshold: float,
    patch_size: int,
    voxel_size_a: float,
    mask: np.ndarray | None,
    min_voxels_for_fsc: int,
) -> list[tuple[int, int, int, float]]:
    """Process a batch of centers (thread worker; shares ``half1``/``half2`` in memory)."""
    out: list[tuple[int, int, int, float]] = []
    for cz, cy, cx in centers:
        r = _patch_resolution_at_center(
            half1,
            half2,
            cz,
            cy,
            cx,
            half=half,
            window3d=window3d,
            shell_idx=shell_idx,
            n_shells=n_shells,
            fsc_threshold=fsc_threshold,
            patch_size=patch_size,
            voxel_size_a=voxel_size_a,
            mask=mask,
            min_voxels_for_fsc=min_voxels_for_fsc,
        )
        if r is not None:
            out.append((cz, cy, cx, r))
    return out


def _centers_on_stride_grid(
    shape: tuple[int, int, int],
    *,
    half: int,
    stride: int,
    mask_b: np.ndarray | None,
) -> list[tuple[int, int, int]]:
    """Patch centers on the stride grid, optionally restricted to ``mask_b``."""
    centers: list[tuple[int, int, int]] = []
    for cz in range(half, shape[0] - half, stride):
        for cy in range(half, shape[1] - half, stride):
            for cx in range(half, shape[2] - half, stride):
                if mask_b is not None and not mask_b[cz, cy, cx]:
                    continue
                centers.append((cz, cy, cx))
    return centers


def _spread_patch_resolutions_to_mask(
    patch_values: list[tuple[int, int, int, float]],
    full_shape: tuple[int, int, int],
    mask_b: np.ndarray,
    *,
    fallback: float,
) -> np.ndarray:
    """
    Assign Å resolution to masked voxels from sparse patch centers.

    Uses linear interpolation inside the convex hull of patch centers, then
    nearest-patch values elsewhere. Does **not** fill the full map box with a
    global median (that produced a misleading rectangular field in ChimeraX).
    """
    fill = fallback if np.isfinite(fallback) else np.nan
    full = np.full(full_shape, fill, dtype=np.float32)
    if not patch_values:
        return full
    idx = np.argwhere(mask_b)
    if idx.size == 0:
        return full

    pts = np.array([(z, y, x) for z, y, x, _ in patch_values], dtype=np.float64)
    res = np.array([r for _, _, _, r in patch_values], dtype=np.float64)
    query = idx.astype(np.float64)

    linear = LinearNDInterpolator(pts, res, fill_value=np.nan)
    nearest = NearestNDInterpolator(pts, res)
    vals = linear(query)
    nan_sel = ~np.isfinite(vals)
    if np.any(nan_sel):
        vals[nan_sel] = nearest(query[nan_sel])

    full[idx[:, 0], idx[:, 1], idx[:, 2]] = vals.astype(np.float32, copy=False)
    return full


def compute_local_fsc_resolution(
    half1: np.ndarray,
    half2: np.ndarray,
    voxel_size_a: float,
    *,
    patch_size: int = 17,
    stride: int = 4,
    fsc_threshold: float = 0.143,
    window: str = "hann",
    mask: np.ndarray | None = None,
    min_voxels_for_fsc: int = 64,
    fallback_resolution_a: float = np.nan,
    n_jobs: int = 1,
    require_mask: bool = True,
) -> np.ndarray:
    """
    Windowed local FSC -> Å-valued per-voxel resolution map ``(Z, Y, X)``.

    For each patch center on a stride grid, extract cubic patches from each
    half-map, apply a soft window, compute shell-averaged 1D FSC, and assign
    resolution from the smallest spatial frequency where FSC drops below
    ``fsc_threshold``. Values are trilinearly interpolated onto masked voxels
    (not the full box — avoids multi-GB ``ndimage.zoom`` on large EMDB entries).

    **Memory:** use a contour mask (``require_mask=True``, default) so only
    macromolecule patch centers are evaluated. ``n_jobs>1`` uses threads, not
    processes, so half-maps are not copied per worker.

    See module docstring for mask, noise-substitution, and clipping caveats.
    """
    h1 = np.asarray(half1)
    h2 = np.asarray(half2)
    if h1.shape != h2.shape or h1.ndim != 3:
        raise ValueError(f"half1 and half2 must be 3D with the same shape; got {h1.shape} vs {h2.shape}")
    if patch_size % 2 == 0 or patch_size < 3:
        raise ValueError("patch_size must be odd and >= 3")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    vox = float(voxel_size_a)
    if vox <= 0:
        raise ValueError("voxel_size_a must be positive")

    shape = h1.shape
    half = patch_size // 2
    shell_idx, n_shells = _build_radial_shell_indices(patch_size)
    window3d = _build_window(patch_size, window)

    mask_b: np.ndarray | None = None
    if mask is not None:
        mask_b = np.asarray(mask).astype(bool)
        if mask_b.shape != shape:
            raise ValueError(f"mask shape {mask_b.shape} != volume shape {shape}")
    elif require_mask:
        raise ValueError(
            "mask is required for large-box local FSC (contour mask from the "
            "reference map). Pass mask=build_contour_mask(...) or require_mask=False "
            "only for small synthetic tests."
        )

    z_centers = list(range(half, shape[0] - half, stride))
    y_centers = list(range(half, shape[1] - half, stride))
    x_centers = list(range(half, shape[2] - half, stride))
    if not z_centers or not y_centers or not x_centers:
        raise ValueError(f"Volume {shape} too small for patch_size={patch_size}")

    centers = _centers_on_stride_grid(shape, half=half, stride=stride, mask_b=mask_b)
    if not centers:
        raise ValueError("No patch centers inside mask; check contour / mask coverage.")
    logger.info("local_fsc: %d patch centers (stride=%d, mask=%s)", len(centers), stride, mask_b is not None)

    vol_bytes = int(h1.nbytes)
    if n_jobs > 1 and vol_bytes > _LARGE_VOLUME_BYTES:
        logger.warning(
            "Large volume (%d MB per half); using thread pool (n_jobs=%d), not processes.",
            vol_bytes // (1024 * 1024),
            n_jobs,
        )

    patch_values: list[tuple[int, int, int, float]] = []

    if n_jobs <= 1:
        for cz, cy, cx in centers:
            r = _patch_resolution_at_center(
                h1,
                h2,
                cz,
                cy,
                cx,
                half=half,
                window3d=window3d,
                shell_idx=shell_idx,
                n_shells=n_shells,
                fsc_threshold=fsc_threshold,
                patch_size=patch_size,
                voxel_size_a=vox,
                mask=mask_b,
                min_voxels_for_fsc=min_voxels_for_fsc,
            )
            if r is not None:
                patch_values.append((cz, cy, cx, r))
    else:
        batch_size = max(1, len(centers) // (int(n_jobs) * 4))
        batches = [centers[i : i + batch_size] for i in range(0, len(centers), batch_size)]
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            futures = [
                ex.submit(
                    _worker_patch_batch,
                    h1,
                    h2,
                    batch,
                    half=half,
                    window3d=window3d,
                    shell_idx=shell_idx,
                    n_shells=n_shells,
                    fsc_threshold=fsc_threshold,
                    patch_size=patch_size,
                    voxel_size_a=vox,
                    mask=mask_b,
                    min_voxels_for_fsc=min_voxels_for_fsc,
                )
                for batch in batches
            ]
            for fut in as_completed(futures):
                patch_values.extend(fut.result())

    spread_mask = mask_b if mask_b is not None else np.ones(shape, dtype=bool)
    full = _spread_patch_resolutions_to_mask(
        patch_values, shape, spread_mask, fallback=fallback_resolution_a,
    )
    full = np.where(
        np.isfinite(full),
        np.clip(full, 2.0 * vox, float(patch_size) * vox),
        fallback_resolution_a,
    )
    return full.astype(np.float32, copy=False)


def save_local_fsc_resolution_mrc(
    resolution_volume: np.ndarray,
    reference_mrc_path: str | Path,
    out_path: str | Path,
    *,
    fsc_threshold: float,
    patch_size: int,
    stride: int,
    mask: np.ndarray | None = None,
    solvent_value: float = 0.0,
) -> Path:
    """
    Save resolution map on the reference grid with MRC labels for ``local_fsc`` inference.

    Voxels outside ``mask`` (or non-finite values) are set to ``solvent_value`` (default 0)
    so ChimeraX does not draw a misleading NaN bounding cube around the map.
    """
    reference_mrc_path = Path(reference_mrc_path)
    out_path = Path(out_path)
    vol = np.asarray(resolution_volume, dtype=np.float32)
    if mask is not None:
        mb = np.asarray(mask).astype(bool)
        if mb.shape != vol.shape:
            raise ValueError(f"mask shape {mb.shape} != volume shape {vol.shape}")
        outside = ~mb
        vol = vol.copy()
        vol[outside] = solvent_value
        vol[~np.isfinite(vol)] = solvent_value
    else:
        vol = np.where(np.isfinite(vol), vol, solvent_value).astype(np.float32, copy=False)
    label = (
        f"local_fsc t={fsc_threshold:g} P={patch_size} s={stride}; solvent={solvent_value:g}"
    )[:80]
    save_volume_like_reference(
        reference_mrc_path,
        vol,
        out_path,
        dtype=np.float32,
        extra_label=label,
    )
    return out_path
