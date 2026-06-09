"""Tests for BlocRes status tracking."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_blocres_local_resolution import (
    STATUS_NAME,
    _format_status_line,
    _parse_contour,
    _reconcile_status,
    _write_contour_mask,
    _write_status,
)


class TestBlocresStatus(unittest.TestCase):
    def test_write_and_format_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            emd = "99999"
            out_dir = Path(tmp) / f"emd_{emd}"
            out_dir.mkdir()
            out_mrc = out_dir / "locres_blocres.mrc"
            out_mrc.write_bytes(b"\x00" * 128)

            with mock.patch(
                "scripts.run_blocres_local_resolution.emd_output_dir",
                return_value=out_dir,
            ):
                _write_status(
                    emd,
                    {
                        "status": "completed",
                        "started_at": "2026-06-08T12:00:00Z",
                        "finished_at": "2026-06-08T12:05:00Z",
                        "output_path": str(out_mrc),
                        "output_bytes": 128,
                    },
                )
                status = _reconcile_status(emd)
                line = _format_status_line(status)

            self.assertEqual(status["status"], "completed")
            self.assertIn("EMD-99999", line)
            self.assertIn("128 B", line)
            saved = json.loads((out_dir / STATUS_NAME).read_text())
            self.assertEqual(saved["status"], "completed")


    def test_parse_contour_rejects_tbd(self) -> None:
        with self.assertRaises(ValueError):
            _parse_contour({"contour": "TBD"}, override=None)

    def test_write_contour_mask_11638(self) -> None:
        ref = Path("data/emd_11638-atomic_apoferritin/emd_11638.map")
        if not ref.is_file():
            self.skipTest("EMD-11638 reference not local")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mask.mrc"
            n = _write_contour_mask(ref, 0.116, out)
            self.assertGreater(n, 10_000)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
