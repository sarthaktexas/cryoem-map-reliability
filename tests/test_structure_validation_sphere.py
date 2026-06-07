"""Sphere sampling and B-factor correlation helpers."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.map_grid import MapGrid
from cryoem_mrc.structure_validation import (
    aggregate_sphere_samples,
    build_ca_sphere_index_caches,
    compute_bfactor_score_correlation_rows,
    summarize_b_iso_distribution,
)
from cryoem_mrc.structure_validation import CaResidue


def _grid() -> MapGrid:
    data = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    return MapGrid(
        data=data,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        origin_zyx=(0.0, 0.0, 0.0),
        shape_zyx=(2, 2, 2),
        mapc=1,
        mapr=2,
        maps=3,
    )


def test_aggregate_sphere_ignores_nan() -> None:
    vol = np.ones((3, 3, 3), dtype=np.float64)
    vol[1, 1, 1] = np.nan
    idx = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.int32)
    mean, n_fin, n_sph = aggregate_sphere_samples(vol, idx, agg="mean")
    assert n_sph == 3
    assert n_fin == 2
    assert mean == 1.0


def test_bfactor_correlation_row_count() -> None:
    grid = _grid()
    res = [
        CaResidue("A", 1, "", "ALA", 0.5, 0.5, 0.5, 10.0),
        CaResidue("A", 2, "", "ALA", 1.5, 0.5, 0.5, 20.0),
        CaResidue("A", 3, "", "ALA", 0.5, 1.5, 0.5, 30.0),
    ]
    caches = build_ca_sphere_index_caches(res, grid, 1.0)
    vol = np.ones((2, 2, 2), dtype=np.float32)
    scores = {"local_variance": vol, "local_cross_correlation": vol * 2}
    b = np.array([r.b_iso for r in res], dtype=np.float64)
    rows = compute_bfactor_score_correlation_rows(
        scores,
        res,
        b,
        sphere_caches_by_radius={1.0: caches},
        radii_a=(1.0,),
        aggregations=("mean",),
        residue_mask=np.ones(3, dtype=bool),
    )
    assert len(rows) == 2


def test_summarize_b_iso_flags_constant() -> None:
    s = summarize_b_iso_distribution(np.full(50, 50.0))
    assert "near zero" in s.notes.lower() or s.std < 1e-3
