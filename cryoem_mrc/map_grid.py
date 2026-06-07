"""Load cryo-EM MRC maps with grid metadata, verify alignment, and resample onto a common grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy import ndimage

from .io import NormalizationMode, normalize_density


def _xyz_tuple(rec) -> tuple[float, float, float]:
    return (float(rec.x), float(rec.y), float(rec.z))


def _permute_xyz_to_zyx(
    vec_xyz: tuple[float, float, float],
    maps: int,
    mapr: int,
    mapc: int,
) -> tuple[float, float, float]:
    """Map MRC (x, y, z) order to array axis order (section, row, col) = (Z, Y, X)."""
    x, y, z = vec_xyz
    lut = {1: x, 2: y, 3: z}
    return (lut[int(maps)], lut[int(mapr)], lut[int(mapc)])


@dataclass
class MapGrid:
    """
    One density map on a regular grid (Z, Y, X) with physical metadata.

    ``voxel_size_zyx`` and ``origin_zyx`` follow the same axis order as ``data``.
    ``origin_zyx`` is the physical coordinate (Å) of the **origin** of voxel index
    (0, 0, 0) using the MRC ``origin`` and ``nstart`` fields (see :func:`load_map_grid`).
    """

    data: np.ndarray
    voxel_size_zyx: tuple[float, float, float]
    origin_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    mapc: int
    mapr: int
    maps: int
    path: Path | None = None
    normalization: NormalizationMode | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, ...]
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D data, got shape {self.data.shape}")
        self.shape_zyx = (int(self.data.shape[0]), int(self.data.shape[1]), int(self.data.shape[2]))


def load_map_grid(
    path: str | Path,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    normalize: NormalizationMode | None = None,
) -> MapGrid:
    """
    Load an MRC / MAP file and return a :class:`MapGrid` with (Z, Y, X) ``data``.

    Voxel sizes (Å) and origin are taken from the MRC header and permuted from the
    file's (x, y, z) convention to match ``data`` axes using ``mapc``, ``mapr``,
    ``maps`` (CCP4 / MRC2000).

    ``origin_zyx`` includes the ``nstart`` grid offset in Å so that two maps from the
    same reconstruction should match when grids align.
    """
    import mrcfile

    if dtype not in (np.float32, np.float64):
        raise TypeError("dtype must be np.float32 or np.float64")
    path = Path(path)
    with mrcfile.open(path) as mrc:
        arr = np.asarray(mrc.data, dtype=dtype)
        h = mrc.header
        vs = mrc.voxel_size
        ns = mrc.nstart
        maps, mapr, mapc = int(h.maps), int(h.mapr), int(h.mapc)
        voxel_zyx = _permute_xyz_to_zyx(_xyz_tuple(vs), maps, mapr, mapc)
        origin_xyz = _xyz_tuple(h.origin)
        nstart_xyz = (float(h.nxstart), float(h.nystart), float(h.nzstart))
        origin_off_xyz = (
            origin_xyz[0] + nstart_xyz[0] * float(vs.x),
            origin_xyz[1] + nstart_xyz[1] * float(vs.y),
            origin_xyz[2] + nstart_xyz[2] * float(vs.z),
        )
        origin_zyx = _permute_xyz_to_zyx(origin_off_xyz, maps, mapr, mapc)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume in {path}, got shape {arr.shape}")

    mg = MapGrid(
        data=arr,
        voxel_size_zyx=voxel_zyx,
        origin_zyx=origin_zyx,
        shape_zyx=(arr.shape[0], arr.shape[1], arr.shape[2]),
        mapc=mapc,
        mapr=mapr,
        maps=maps,
        path=path,
    )
    if normalize is not None:
        mg = MapGrid(
            data=normalize_density(mg.data, mode=normalize),
            voxel_size_zyx=mg.voxel_size_zyx,
            origin_zyx=mg.origin_zyx,
            shape_zyx=mg.shape_zyx,
            mapc=mg.mapc,
            mapr=mg.mapr,
            maps=mg.maps,
            path=mg.path,
            normalization=normalize,
        )
    return mg


@dataclass
class GridAlignmentReport:
    """Human-readable result of comparing two :class:`MapGrid` headers and shapes."""

    shape_match: bool
    voxel_match: bool
    origin_match: bool
    axis_order_match: bool
    messages: list[str]

    @property
    def ok(self) -> bool:
        return all(
            (
                self.shape_match,
                self.voxel_match,
                self.origin_match,
                self.axis_order_match,
            )
        )


def verify_grid_alignment(
    a: MapGrid,
    b: MapGrid,
    *,
    voxel_rtol: float = 1e-3,
    voxel_atol: float = 1e-4,
    origin_atol: float = 1e-2,
) -> GridAlignmentReport:
    """
    Compare shape, voxel spacing, origin, and (mapc, mapr, maps) between two maps.

    ``origin_atol`` is in Å and applies to each axis after subtracting origins.
    """
    msgs: list[str] = []
    shape_match = a.shape_zyx == b.shape_zyx
    if not shape_match:
        msgs.append(f"Shape mismatch: {a.shape_zyx} vs {b.shape_zyx}")

    va = np.asarray(a.voxel_size_zyx, dtype=np.float64)
    vb = np.asarray(b.voxel_size_zyx, dtype=np.float64)
    voxel_match = bool(np.allclose(va, vb, rtol=voxel_rtol, atol=voxel_atol))
    if not voxel_match:
        msgs.append(f"Voxel size mismatch (Z,Y,X): {a.voxel_size_zyx} vs {b.voxel_size_zyx}")

    oa = np.asarray(a.origin_zyx, dtype=np.float64)
    ob = np.asarray(b.origin_zyx, dtype=np.float64)
    origin_match = bool(np.allclose(oa, ob, rtol=0.0, atol=origin_atol))
    if not origin_match:
        diff = tuple(float(x) for x in (oa - ob))
        msgs.append(f"Origin mismatch (Z,Y,X) diff Å: {diff}")

    axis_order_match = (a.mapc, a.mapr, a.maps) == (b.mapc, b.mapr, b.maps)
    if not axis_order_match:
        msgs.append(
            f"Axis order (mapc,mapr,maps): ({a.mapc},{a.mapr},{a.maps}) vs "
            f"({b.mapc},{b.mapr},{b.maps})"
        )

    return GridAlignmentReport(
        shape_match=shape_match,
        voxel_match=voxel_match,
        origin_match=origin_match,
        axis_order_match=axis_order_match,
        messages=msgs,
    )


def verify_same_grid_as_reference(
    volume: MapGrid,
    reference: MapGrid,
    *,
    voxel_rtol: float = 1e-3,
    voxel_atol: float = 1e-4,
    origin_atol: float = 1e-2,
) -> GridAlignmentReport:
    """
    Same checks as :func:`verify_grid_alignment` for a derived volume (e.g. local
    resolution) against a reference map grid.
    """
    return verify_grid_alignment(
        volume,
        reference,
        voxel_rtol=voxel_rtol,
        voxel_atol=voxel_atol,
        origin_atol=origin_atol,
    )


def resample_volume_onto_grid(
    source: MapGrid,
    target: MapGrid,
    *,
    order: int = 1,
    chunk_z: int = 32,
    cval: float = 0.0,
) -> np.ndarray:
    """
    Interpolate ``source.data`` onto the grid of ``target`` (same shape as ``target.data``).

    Uses physical coordinates (Å): voxel corner ``origin + index * voxel_size`` for
    each axis. Values outside ``source`` are filled with ``cval``.

    Processes Z slabs of thickness ``chunk_z`` to limit peak memory from coordinate arrays.
    """
    if order not in (0, 1, 2, 3, 4, 5):
        raise ValueError("order must be 0–5 for map_coordinates")
    src = np.asarray(source.data, dtype=np.float64)
    nz_t, ny_t, nx_t = target.shape_zyx
    out = np.empty((nz_t, ny_t, nx_t), dtype=source.data.dtype)
    v_s = np.asarray(source.voxel_size_zyx, dtype=np.float64)
    o_s = np.asarray(source.origin_zyx, dtype=np.float64)
    v_t = np.asarray(target.voxel_size_zyx, dtype=np.float64)
    o_t = np.asarray(target.origin_zyx, dtype=np.float64)

    y_coords = np.arange(ny_t, dtype=np.float64)
    x_coords = np.arange(nx_t, dtype=np.float64)

    for z0 in range(0, nz_t, chunk_z):
        z1 = min(nz_t, z0 + chunk_z)
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        wy = o_t[1] + y_coords[None, :, None] * v_t[1]
        wx = o_t[2] + x_coords[None, None, :] * v_t[2]
        wz = o_t[0] + zz * v_t[0]

        coords = np.empty((3, z1 - z0, ny_t, nx_t), dtype=np.float64)
        coords[0] = (wz - o_s[0]) / v_s[0]
        coords[1] = (wy - o_s[1]) / v_s[1]
        coords[2] = (wx - o_s[2]) / v_s[2]

        slab = ndimage.map_coordinates(
            src,
            coords,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=order > 1,
        )
        out[z0:z1] = slab.astype(out.dtype, copy=False)

    return out


@dataclass
class FullHalfMapBundle:
    """Aligned full map and two half-maps plus per-file alignment vs. the chosen reference."""

    full: MapGrid
    half1: MapGrid
    half2: MapGrid
    reference: Literal["full", "half1", "half2"]
    reports: dict[str, GridAlignmentReport]


def load_full_and_half_maps(
    full_path: str | Path,
    half1_path: str | Path,
    half2_path: str | Path,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    reference: Literal["full", "half1", "half2"] = "full",
    normalize: NormalizationMode | None = None,
    resample_if_needed: bool = True,
    resample_order: int = 1,
    chunk_z: int = 32,
) -> FullHalfMapBundle:
    """
    Load full map and two half-maps, verify each against the reference grid, optionally resample.

    When ``resample_if_needed`` is True, any map that does not match the reference is
    interpolated in memory onto the reference grid. :attr:`FullHalfMapBundle.reports`
    always compares the **original** on-disk maps to the reference so you can see whether
    resampling was required.
    """
    full = load_map_grid(full_path, dtype=dtype, normalize=normalize)
    h1 = load_map_grid(half1_path, dtype=dtype, normalize=normalize)
    h2 = load_map_grid(half2_path, dtype=dtype, normalize=normalize)

    ref = {"full": full, "half1": h1, "half2": h2}[reference]
    originals = {"full": full, "half1": h1, "half2": h2}
    reports = {k: verify_grid_alignment(v, ref) for k, v in originals.items()}

    def _aligned(mg: MapGrid, rep: GridAlignmentReport) -> MapGrid:
        if rep.ok or not resample_if_needed:
            return mg
        data = resample_volume_onto_grid(mg, ref, order=resample_order, chunk_z=chunk_z)
        return MapGrid(
            data=data.astype(dtype, copy=False),
            voxel_size_zyx=ref.voxel_size_zyx,
            origin_zyx=ref.origin_zyx,
            shape_zyx=ref.shape_zyx,
            mapc=ref.mapc,
            mapr=ref.mapr,
            maps=ref.maps,
            path=mg.path,
            normalization=mg.normalization,
        )

    return FullHalfMapBundle(
        full=_aligned(full, reports["full"]),
        half1=_aligned(h1, reports["half1"]),
        half2=_aligned(h2, reports["half2"]),
        reference=reference,
        reports=reports,
    )


def ensure_same_grid(
    maps: Iterable[MapGrid],
    *,
    reference_index: int = 0,
    resample_order: int = 1,
    chunk_z: int = 32,
) -> tuple[list[MapGrid], list[GridAlignmentReport]]:
    """
    Given several :class:`MapGrid` instances, resample any that differ from ``maps[reference_index]``.

    Returns ``(aligned_maps, reports_vs_reference)`` where each report compares the
    **original** map to the reference before resampling.
    """
    mlist = list(maps)
    if not mlist:
        return [], []
    ref = mlist[reference_index]
    reports: list[GridAlignmentReport] = []
    out: list[MapGrid] = []
    for mg in mlist:
        rep = verify_grid_alignment(mg, ref)
        reports.append(rep)
        if rep.ok:
            out.append(mg)
        else:
            data = resample_volume_onto_grid(mg, ref, order=resample_order, chunk_z=chunk_z)
            out.append(
                MapGrid(
                    data=data.astype(mg.data.dtype, copy=False),
                    voxel_size_zyx=ref.voxel_size_zyx,
                    origin_zyx=ref.origin_zyx,
                    shape_zyx=ref.shape_zyx,
                    mapc=ref.mapc,
                    mapr=ref.mapr,
                    maps=ref.maps,
                    path=mg.path,
                    normalization=mg.normalization,
                )
            )
    return out, reports
