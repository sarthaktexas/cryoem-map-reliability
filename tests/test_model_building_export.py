"""Unit tests for model-building export helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cryoem_mrc.model_building_export import (
    write_chimerax_build_zone_script,
    write_coot_zone_python,
)
from cryoem_mrc.structure_validation import ResidueValidationRow


def _row(chain: str, seq: int, zone: int) -> ResidueValidationRow:
    return ResidueValidationRow(
        chain=chain,
        seq_num=seq,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=20.0,
        reliability_score=0.5,
        reliability_H_repro=0.5,
        build_zone=zone,
        in_contour_mask=True,
        auth_chain=chain,
        auth_seq_num=seq,
    )


class TestModelBuildingExport(unittest.TestCase):
    def test_chimerax_script_contains_zone_colors(self) -> None:
        rows = [_row("A", 10, 0), _row("A", 11, 2)]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "zones.cxc"
            write_chimerax_build_zone_script(
                structure_path=Path("model.cif"),
                reference_mrc=Path("map.mrc"),
                rows=rows,
                out_script=script,
                contour=0.1,
            )
            text = script.read_text()
            self.assertIn("#e74c3c", text)
            self.assertIn("#27ae60", text)
            self.assertIn("/A:10", text)

    def test_coot_script_calls_color(self) -> None:
        rows = [_row("B", 5, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coot.py"
            write_coot_zone_python(rows, path)
            text = path.read_text()
            self.assertIn("set_residue_colour", text)
            self.assertIn("'B'", text)


if __name__ == "__main__":
    unittest.main()
