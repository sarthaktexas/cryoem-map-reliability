"""Unit tests for Q-score validation helpers (no qscore runtime required)."""

from __future__ import annotations

import unittest

import numpy as np

from cryoem_mrc.qscore_validation import QscoreResidueRow, compute_qscore_validation_stats


def _row(q: float, v: float, *, in_mask: bool = True, b: float = 50.0) -> QscoreResidueRow:
    return QscoreResidueRow(
        chain="A",
        seq_num=1,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=b,
        q_score=q,
        reliability_constraint_V=v,
        reliability_constraint_V_rank=v,
        in_contour_mask=in_mask,
    )


class TestQscoreValidationStats(unittest.TestCase):
    def test_positive_correlation(self):
        rows = [_row(float(i) / 20.0, float(i), b=float(i)) for i in range(20)]
        stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
        self.assertEqual(stats.n_in_mask, 20)
        self.assertGreater(stats.spearman_q_vs_V, 0.95)

    def test_respects_mask(self):
        rows = [_row(0.5, 1.0, in_mask=False) for _ in range(20)]
        stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
        self.assertTrue(np.isnan(stats.spearman_q_vs_V))


if __name__ == "__main__":
    unittest.main()
