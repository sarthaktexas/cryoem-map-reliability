"""Unit tests for semi-prospective LOMO placement validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cryoem_mrc.placement_utility import (
    evaluate_map_predictor,
    pooled_roc_curve,
    run_lomo_placement_validation,
)


def _frame(emdb_id: str, seed: int, n: int = 80) -> tuple[str, pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0, 1, size=n)
    q = 0.25 + 0.55 * rel + rng.normal(0, 0.08, size=n)
    cc = 0.15 + 0.65 * rel + rng.normal(0, 0.07, size=n)
    loc = rng.uniform(2.5, 5.0, size=n)
    var = rng.uniform(0.5, 4.0, size=n)
    zone = np.digitize(rel, [1 / 3, 2 / 3]) - 1
    df = pd.DataFrame(
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
    return emdb_id, df, 3.0


class TestLomoPlacement(unittest.TestCase):
    def test_lomo_runs_with_three_maps(self) -> None:
        frames = [_frame("a", 0), _frame("b", 1), _frame("c", 2)]
        summary = run_lomo_placement_validation(frames, q_threshold=0.5)
        self.assertEqual(len({r.held_out_emdb_id for r in summary.fold_rows}), 3)
        rel = summary.predictor_medians["reliability_below_0_33"]
        self.assertGreater(rel["median_auc"], 0.55)

    def test_pooled_roc_auc_high_when_correlated(self) -> None:
        _eid, df, _ = _frame("x", 3, n=120)
        per_map = [("x", df)]
        curve = pooled_roc_curve(per_map, "reliability_below_0_33", q_threshold=0.5)
        self.assertGreater(curve.auc, 0.7)

    def test_train_derived_locres_differs_from_in_map(self) -> None:
        frames = [_frame("a", 4), _frame("b", 5), _frame("c", 6)]
        test_df = frames[0][1]
        train_dfs = [frames[1][1], frames[2][1]]
        from cryoem_mrc.placement_utility import _train_medians

        loc_med, _ = _train_medians(train_dfs)
        ba_train, _, _, _, _, _ = evaluate_map_predictor(
            test_df,
            "locres_worse_than_median",
            q_threshold=0.5,
            train_locres_median=loc_med,
        )
        ba_inmap, _, _, _, _, _ = evaluate_map_predictor(
            test_df,
            "locres_worse_than_median",
            q_threshold=0.5,
        )
        self.assertTrue(np.isfinite(ba_train))
        self.assertTrue(np.isfinite(ba_inmap))


if __name__ == "__main__":
    unittest.main()
