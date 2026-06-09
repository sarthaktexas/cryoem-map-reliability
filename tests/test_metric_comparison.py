"""Tests for cross-metric loading and correlations."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from cryoem_mrc.metric_comparison import compute_cross_metric_correlations, load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST


class TestMetricComparison(unittest.TestCase):
    def test_load_all_metrics_local_resolution_nan_without_blocres(self) -> None:
        try:
            df = load_all_metrics("49450", manifest=COHORT_MANIFEST)
        except FileNotFoundError:
            self.skipTest("EMD-49450 pipeline outputs not local")
        self.assertIn("local_resolution", df.columns)
        self.assertIn("v_metric", df.columns)
        locres_path = Path("outputs/emd_49450/locres_blocres.mrc")
        if locres_path.is_file():
            self.assertTrue(df["local_resolution"].notna().any())
        else:
            self.assertTrue(df["local_resolution"].isna().all())

    def test_compute_cross_metric_correlations_shape(self) -> None:
        try:
            df = load_all_metrics("49450", manifest=COHORT_MANIFEST)
        except FileNotFoundError:
            self.skipTest("EMD-49450 pipeline outputs not local")
        corr = compute_cross_metric_correlations(df)
        self.assertEqual(corr.shape[0], corr.shape[1])
        self.assertIn("v_metric", corr.index)
        self.assertIn("local_resolution", corr.columns)
        v_loc = corr.loc["v_metric", "local_resolution"]
        if df["local_resolution"].notna().sum() >= 30:
            self.assertTrue(np.isfinite(v_loc))


if __name__ == "__main__":
    unittest.main()
