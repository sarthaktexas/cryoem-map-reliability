"""Tests for LH phase portrait classification."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.mechanics import classify_tv_regime


def test_tv_regime_labels_inside_mask() -> None:
    t = np.array([[[0.1, 2.0], [0.1, 0.1]], [[2.0, 2.0], [0.1, 2.0]]], dtype=np.float32)
    v = np.array([[[0.1, 0.1], [2.0, 0.1]], [[0.1, 2.0], [2.0, 0.1]]], dtype=np.float32)
    mask = np.ones_like(t, dtype=bool)
    z = classify_tv_regime(t, v, mask)
    assert set(np.unique(z[mask])) <= {0, 1, 2}
