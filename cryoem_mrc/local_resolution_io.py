"""Load home-rolled local-FSC resolution maps, align to reference, export datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

import mrcfile

from .map_grid import (
    GridAlignmentReport,
    MapGrid,
    load_map_grid,
    resample_volume_onto_grid,
    verify_same_grid_as_reference,
)
from .pipeline import load_feature_maps, run_pipeline

logger = logging.getLogger(__name__)

LocalResolutionSource = Literal["local_fsc", "blocres"]

LocalResolutionGridReport = GridAlignmentReport


def _header_label_strings(path: Path) -> list[str]:
    with mrcfile.open(path) as mrc:
        labels = mrc.get_labels()
    return [str(lab).lower() for lab in labels]


def _infer_local_fsc_from_path(path: Path) -> bool:
    """True if filename stem or MRC labels indicate a ``local_fsc`` map."""
    stem = path.stem.lower()
    if "local_fsc" in stem or "localfsc" in stem:
        return True
    try:
        for lab in _header_label_strings(path):
            if "local_fsc" in lab or "local fsc" in lab or "localfsc" in lab:
                return True
    except OSError as e:
        logger.debug("Could not read MRC labels for %s: %s", path, e)
    return False


def _infer_blocres_from_path(path: Path) -> bool:
    """True if filename stem or MRC labels indicate a BlocRes local-resolution map."""
    stem = path.stem.lower()
    if "locres_blocres" in stem or stem.endswith("_blocres") or stem == "blocres":
        return True
    try:
        for lab in _header_label_strings(path):
            if "blocres" in lab:
                return True
    except OSError as e:
        logger.debug("Could not read MRC labels for %s: %s", path, e)
    return False


def load_local_resolution_map(
    path: str | Path,
    *,
    source: LocalResolutionSource | None = None,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> MapGrid:
    """
    Load an Å-valued local-resolution volume (home-rolled FSC or BlocRes).

    ``source`` may be ``"local_fsc"`` or ``"blocres"`` when set explicitly. When
    ``source`` is None, the path stem or MRC labels must identify the format.
    """
    path = Path(path)
    if source == "local_fsc":
        if not _infer_local_fsc_from_path(path):
            raise ValueError(f"Could not identify {path.name} as a local_fsc map.")
        resolved = "local_fsc"
    elif source == "blocres":
        if not _infer_blocres_from_path(path):
            raise ValueError(f"Could not identify {path.name} as a BlocRes map.")
        resolved = "blocres"
    elif source is None:
        if _infer_blocres_from_path(path):
            resolved = "blocres"
        elif _infer_local_fsc_from_path(path):
            resolved = "local_fsc"
        else:
            raise ValueError(
                f"Could not identify {path.name} as local_fsc or BlocRes. "
                "Use a filename like 'locres_blocres.mrc' or '*_local_fsc_*.mrc'."
            )
    else:
        raise ValueError(
            f"Unsupported local-resolution source {source!r}; "
            "use 'local_fsc' or 'blocres'."
        )
    logger.debug("load_local_resolution_map: source=%s path=%s", resolved, path)
    return load_map_grid(path, dtype=dtype, normalize=None)


def resample_local_resolution_onto_reference(
    local: MapGrid,
    reference: MapGrid,
    *,
    order: int = 1,
    chunk_z: int = 32,
) -> MapGrid:
    """
    Interpolate local-resolution values onto ``reference``'s grid (Å), same convention
    as :func:`map_grid.resample_volume_onto_grid`.
    """
    data = resample_volume_onto_grid(local, reference, order=order, chunk_z=chunk_z)
    return MapGrid(
        data=data.astype(local.data.dtype, copy=False),
        voxel_size_zyx=reference.voxel_size_zyx,
        origin_zyx=reference.origin_zyx,
        shape_zyx=reference.shape_zyx,
        mapc=reference.mapc,
        mapr=reference.mapr,
        maps=reference.maps,
        path=local.path,
        normalization=None,
    )


def verify_local_resolution_matches_reference(
    local_aligned: MapGrid,
    reference: MapGrid,
    *,
    voxel_rtol: float = 1e-3,
    voxel_atol: float = 1e-4,
    origin_atol: float = 1e-2,
) -> LocalResolutionGridReport:
    """Same checks as :func:`map_grid.verify_same_grid_as_reference` for aligned local-res."""
    return verify_same_grid_as_reference(
        local_aligned,
        reference,
        voxel_rtol=voxel_rtol,
        voxel_atol=voxel_atol,
        origin_atol=origin_atol,
    )


def _mask_bool(mask: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} != reference {shape}")
    if m.dtype == bool:
        return m
    return m > 0


def export_masked_feature_dataset(
    reference_mrc_path: str | Path,
    features: dict[str, np.ndarray],
    local_resolution_angstrom: np.ndarray,
    mask: np.ndarray | None,
    *,
    out_path: Path,
    format: Literal["npz", "zarr"] = "npz",
    metadata_extra: dict[str, Any] | None = None,
    zarr_chunks: tuple[int, int] | None = None,
) -> Path:
    """
    Export a masked paired dataset for ML / analysis.

    **Mask:** boolean or float ``(Z,Y,X)``. Float masks use ``> 0`` as inclusion.
    If ``mask`` is None, **all voxels** are exported (high RAM for large boxes).

    **Features:** only keys whose values are ndarray, ndim==3, and whose shape
    matches the reference map are included. Other keys (e.g. ``multiscale_sigmas``)
    are skipped with a log line; a JSON sidecar
    ``<stem>_skipped_features.json`` lists them.

    **Metadata** is stored as JSON inside the NPZ under ``metadata_json`` (UTF-8 string
    array), or as Zarr group attributes when ``format='zarr'``.
    """
    reference_mrc_path = Path(reference_mrc_path)
    out_path = Path(out_path)
    ref_mg = load_map_grid(reference_mrc_path, normalize=None)
    shape = ref_mg.shape_zyx
    lr = np.asarray(local_resolution_angstrom)
    if lr.shape != shape:
        raise ValueError(f"local_resolution_angstrom shape {lr.shape} != reference {shape}")

    mb = _mask_bool(mask, shape)
    if mb is None:
        logger.warning(
            "Exporting full box %s with no mask (very large arrays possible).",
            shape,
        )
        zz, yy, xx = np.meshgrid(
            np.arange(shape[0], dtype=np.int32),
            np.arange(shape[1], dtype=np.int32),
            np.arange(shape[2], dtype=np.int32),
            indexing="ij",
        )
        indices_zyx = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
    else:
        indices_zyx = np.argwhere(mb).astype(np.int32, copy=False)

    n = indices_zyx.shape[0]
    loc_flat = lr[indices_zyx[:, 0], indices_zyx[:, 1], indices_zyx[:, 2]]

    export_feats: dict[str, np.ndarray] = {
        "indices_zyx": indices_zyx,
        "local_resolution_A": loc_flat.astype(np.float32, copy=False),
    }

    skipped: dict[str, str] = {}
    for key, arr in features.items():
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape != shape:
            skipped[key] = f"ndim={a.ndim}, shape={getattr(a, 'shape', None)}"
            continue
        export_feats[key] = a[indices_zyx[:, 0], indices_zyx[:, 1], indices_zyx[:, 2]].astype(
            np.float32,
            copy=False,
        )

    if skipped:
        sidecar = out_path.parent / f"{out_path.stem}_skipped_features.json"
        sidecar.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        logger.info("Skipped %d non-3D or mis-shaped feature keys; wrote %s", len(skipped), sidecar)

    meta: dict[str, Any] = {
        "reference_mrc_path": str(reference_mrc_path.resolve()),
        "voxel_size_zyx": [float(x) for x in ref_mg.voxel_size_zyx],
        "origin_zyx": [float(x) for x in ref_mg.origin_zyx],
        "shape_zyx": [int(x) for x in shape],
        "n_voxels_exported": int(n),
        "masked": mask is not None,
    }
    if metadata_extra:
        meta.update(metadata_extra)

    meta_json = json.dumps(meta, indent=2)
    export_feats["metadata_json"] = np.asarray(meta_json)

    fmt = format
    if fmt == "npz":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **export_feats)
        return out_path

    if fmt == "zarr":
        try:
            import zarr
        except ImportError as e:
            raise ImportError(
                "format='zarr' requires the optional 'zarr' package. "
                "Install with `pip install zarr` or use format='npz'."
            ) from e
        out_path.parent.mkdir(parents=True, exist_ok=True)
        zpath = out_path
        if zpath.suffix != ".zarr":
            zpath = out_path.with_suffix(".zarr")
        store = zarr.DirectoryStore(str(zpath))
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update({k: meta[k] for k in meta})  # type: ignore[arg-type]
        chunk_rows, chunk_cols = zarr_chunks or (min(65536, max(1, n)), 3)
        for k, v in export_feats.items():
            if k == "metadata_json":
                continue
            arr = np.asarray(v)
            chunks = (chunk_rows,) if arr.ndim == 1 else (chunk_rows, chunk_cols)
            root.create_dataset(k, data=arr, chunks=chunks, overwrite=True)
        return Path(zpath)

    raise ValueError(f"Unknown format: {fmt!r}")


def build_dataset_from_pipeline(
    reference_mrc_path: str | Path,
    *,
    local_res_path: str | Path,
    local_res_source: LocalResolutionSource | None = None,
    features_npz: str | Path | None = None,
    mask: np.ndarray | None = None,
    out_path: str | Path,
    export_format: Literal["npz", "zarr"] = "npz",
    pipeline_kwargs: dict[str, Any] | None = None,
    voxel_rtol: float = 1e-3,
    origin_atol: float = 1e-2,
) -> Path:
    """
    Build a masked feature + local-resolution export.

    ``local_res_path`` must point to an Å-valued MRC from
    :mod:`cryoem_mrc.local_fsc` (``scripts/run_local_fsc.py``).
    Features come from ``features_npz`` if set, else :func:`pipeline.run_pipeline`.
    """
    reference_mrc_path = Path(reference_mrc_path)
    out_path = Path(out_path)

    if local_res_path is None:
        raise ValueError("local_res_path is required.")
    local_path = Path(local_res_path)

    ref_mg = load_map_grid(reference_mrc_path, normalize=None)
    local_mg = load_local_resolution_map(local_path, source=local_res_source)
    rep0 = verify_same_grid_as_reference(local_mg, ref_mg, voxel_rtol=voxel_rtol, origin_atol=origin_atol)
    if rep0.ok:
        aligned_mg = MapGrid(
            data=np.asarray(local_mg.data, dtype=np.float64),
            voxel_size_zyx=ref_mg.voxel_size_zyx,
            origin_zyx=ref_mg.origin_zyx,
            shape_zyx=ref_mg.shape_zyx,
            mapc=ref_mg.mapc,
            mapr=ref_mg.mapr,
            maps=ref_mg.maps,
            path=local_mg.path,
            normalization=None,
        )
    else:
        aligned_mg = resample_local_resolution_onto_reference(
            local_mg,
            ref_mg,
            order=1,
            chunk_z=32,
        )

    vrep = verify_local_resolution_matches_reference(
        aligned_mg,
        ref_mg,
        voxel_rtol=voxel_rtol,
        origin_atol=origin_atol,
    )
    if not vrep.ok:
        raise ValueError(f"Local resolution grid mismatch after resample: {vrep.messages}")

    if features_npz is not None:
        feats = load_feature_maps(features_npz)
        norm_flag = "from_npz"
    else:
        pk = dict(pipeline_kwargs or {})
        feats = run_pipeline(reference_mrc_path, **pk)
        norm_flag = str(pk.get("normalization", "zscore"))

    meta_extra: dict[str, Any] = {
        "normalization": norm_flag,
        "local_resolution_source": "local_fsc",
        "local_resolution_path": str(local_path.resolve()),
    }

    return export_masked_feature_dataset(
        reference_mrc_path,
        feats,
        np.asarray(aligned_mg.data),
        mask,
        out_path=out_path,
        format=export_format,
        metadata_extra=meta_extra,
    )
