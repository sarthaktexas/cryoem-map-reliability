"""Unit tests for placement utility analyses."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cryoem_mrc.placement_utility import (
    balanced_accuracy,
    compute_calibration_bins,
    compute_low_q_enrichment_row,
    compute_misranking_row,
    rank_auc,
    _predictor_flags,
)


def _synthetic_df(*, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0, 1, size=n)
    q = 0.3 + 0.5 * rel + rng.normal(0, 0.1, size=n)
    cc = 0.2 + 0.6 * rel + rng.normal(0, 0.08, size=n)
    loc = rng.uniform(2.5, 5.0, size=n)
    var = rng.uniform(0.5, 5.0, size=n)
    zone = np.digitize(rel, [1 / 3, 2 / 3]) - 1
    return pd.DataFrame(
        {
            "reliability_score": rel,
            "q_score": q,
            "windowed_halfmap_correlation": cc,
            "local_resolution": loc,
            "local_variance": var,
            "build_zone": zone,
            "in_contour_mask": True,
        }
    )


class TestPlacementUtilityMetrics(unittest.TestCase):
    def test_rank_auc_perfect_separation(self) -> None:
        y = np.array([1, 1, 0, 0])
        scores = np.array([1.0, 0.9, 0.1, 0.0])
        self.assertAlmostEqual(rank_auc(y, scores), 1.0)

    def test_balanced_accuracy(self) -> None:
        y = np.array([1, 1, 0, 0])
        pred = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(balanced_accuracy(y, pred), 0.75)


class TestLowQEnrichment(unittest.TestCase):
    def test_enrichment_tracks_correlated_q(self) -> None:
        df = _synthetic_df()
        row = compute_low_q_enrichment_row(df, emdb_id="test", q_threshold=0.5)
        assert row is not None
        self.assertGreater(row.frac_low_q_in_omit_zone, row.omit_zone_baseline)
        self.assertGreater(row.frac_low_q_reliability_below, 0.2)

    def test_predictor_flags_shapes(self) -> None:
        df = _synthetic_df(n=50)
        flags = _predictor_flags(df)
        self.assertEqual(len(flags["omit_zone"]), 50)


class TestMisranking(unittest.TestCase):
    def test_misranking_row_finite(self) -> None:
        df = _synthetic_df()
        row = compute_misranking_row(df, emdb_id="test")
        assert row is not None
        self.assertTrue(np.isfinite(row.frac_omit_zone_low_q_tercile))


class TestCalibration(unittest.TestCase):
    def test_calibration_bins_monotone_trend(self) -> None:
        df = _synthetic_df()
        bins = compute_calibration_bins([df], n_bins=5)
        self.assertGreater(len(bins), 2)
        means = [b.mean_q for b in bins]
        self.assertGreater(means[-1], means[0])


if __name__ == "__main__":
    unittest.main()
