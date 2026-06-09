"""Tests for BlocRes Cα aggregation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mrcfile
import numpy as np

from cryoem_mrc.local_resolution import aggregate_locres_to_ca
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.structure_validation import iter_ca_residues


class TestAggregateLocresToCa(unittest.TestCase):
    def _write_constant_locres(self, path: Path, value: float, *, n: int = 16, vs: float = 1.0) -> None:
        data = np.full((n, n, n), value, dtype=np.float32)
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = vs
            mrc.header.origin.x = 0.0
            mrc.header.origin.y = 0.0
            mrc.header.origin.z = 0.0

    def test_aggregate_returns_columns(self) -> None:
        pdb = Path("pdb/7a4m.cif")
        if not pdb.is_file():
            self.skipTest("pdb/7a4m.cif not in workspace")
        with tempfile.TemporaryDirectory() as tmp:
            locres = Path(tmp) / "locres.mrc"
            self._write_constant_locres(locres, 2.5, vs=0.5332)
            df = aggregate_locres_to_ca(locres, pdb, radius_angstrom=2.0)
        self.assertIn("chain", df.columns)
        self.assertIn("seq_num", df.columns)
        self.assertIn("local_resolution_mean", df.columns)
        self.assertIn("n_voxels", df.columns)
        self.assertEqual(len(df), len(iter_ca_residues(pdb)))
        finite = df["local_resolution_mean"].dropna()
        if finite.empty:
            self.skipTest("no Cα fell inside synthetic locres grid")
        self.assertTrue((finite > 0).all())

    def test_aggregate_returns_schema_when_all_ca_off_grid(self) -> None:
        """Regression (EMD-33736): an empty result must still carry the documented
        columns so callers selecting them do not hit a KeyError."""
        pdb = Path("pdb/7a4m.cif")
        if not pdb.is_file():
            self.skipTest("pdb/7a4m.cif not in workspace")
        with tempfile.TemporaryDirectory() as tmp:
            locres = Path(tmp) / "locres.mrc"
            # Tiny grid far from the model origin -> every sphere cache is empty.
            self._write_constant_locres(locres, 2.5, n=4, vs=0.5)
            df = aggregate_locres_to_ca(
                locres, pdb, radius_angstrom=2.0, value_column="custom_value"
            )
        for col in ("chain", "seq_num", "custom_value", "n_voxels"):
            self.assertIn(col, df.columns)
        # The selection pattern that previously raised KeyError must now work.
        _ = df[["chain", "seq_num", "custom_value"]]


class TestBlocresHeaderCheck(unittest.TestCase):
    def test_half_maps_match_reference_49450(self) -> None:
        from scripts.run_blocres_local_resolution import _assert_half_maps_match_reference

        ref = Path("data/emd_49450-mgtA_e2p+e1/emd_49450.map")
        h1 = Path("data/emd_49450-mgtA_e2p+e1/emd_49450_half_map_1.map")
        h2 = Path("data/emd_49450-mgtA_e2p+e1/emd_49450_half_map_2.map")
        if not all(p.is_file() for p in (ref, h1, h2)):
            self.skipTest("EMD-49450 maps not local")
        vs = _assert_half_maps_match_reference(ref, h1, h2)
        self.assertAlmostEqual(vs, 0.93, places=2)


if __name__ == "__main__":
    unittest.main()
