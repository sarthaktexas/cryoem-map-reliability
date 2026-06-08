"""Tests for contour-mask bounding-box cropping."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.analysis import build_contour_mask, half_map_local_metrics_chunked, half_map_local_metrics_chunked_bbox
from cryoem_mrc.mask_bbox import (
    VolumeBbox,
    bbox_from_mask,
    crop_array,
    embed_array,
    pad_voxels_for_filters,
)


def test_bbox_crop_roundtrip() -> None:
    vol = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 3:7, 4:6] = True
    bbox = bbox_from_mask(mask, pad=1)
    assert bbox == VolumeBbox(1, 9, 2, 8, 3, 7)
    cropped = crop_array(vol, bbox)
    back = embed_array(vol.shape, bbox, cropped, dtype=vol.dtype)
    assert np.array_equal(back[bbox.slices], cropped)
    assert (back == 0).sum() == full_n - bbox.n_voxels if (full_n := vol.size) else True


def test_half_map_metrics_bbox_matches_full_in_mask_interior() -> None:
    rng = np.random.default_rng(0)
    shape = (48, 48, 48)
    half1 = rng.standard_normal(shape, dtype=np.float32)
    half2 = rng.standard_normal(shape, dtype=np.float32)
    density = np.abs(half1 + half2)
    mask = build_contour_mask(density, contour=float(np.percentile(density, 60)))
    pad = pad_voxels_for_filters(window=5)
    bbox = bbox_from_mask(mask, pad=pad)

    full = half_map_local_metrics_chunked(half1, half2, window=5, chunk_z=16)
    cropped = half_map_local_metrics_chunked_bbox(half1, half2, mask, window=5, chunk_z=16, pad=pad)

    # Interior in-mask voxels (two voxels from bbox edge) should match closely.
    inner = np.zeros(shape, dtype=bool)
    inner[bbox.z0 + 2 : bbox.z1 - 2, bbox.y0 + 2 : bbox.y1 - 2, bbox.x0 + 2 : bbox.x1 - 2] = True
    check = mask & inner
    assert check.sum() > 100
    for key in full:
        a = full[key][check]
        b = cropped[key][check]
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    outside = ~mask
    assert np.all(cropped["local_cross_correlation"][outside] == 0)
