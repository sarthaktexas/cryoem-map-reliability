"""Smoke tests for local-resolution export (synthetic small volumes)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mrcfile

from cryoem_mrc.io import save_volume_like_reference
from cryoem_mrc.local_resolution_io import (
    export_masked_feature_dataset,
    load_local_resolution_map,
)
from cryoem_mrc.map_grid import load_map_grid, verify_same_grid_as_reference


class TestMaskedLocalResolutionExport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_masked_npz_roundtrip_shapes(self) -> None:
        shape = (32, 32, 32)
        ref = np.random.randn(*shape).astype(np.float32) * 0.1
        loc = np.abs(np.random.randn(*shape).astype(np.float32)) + 2.0
        mask = np.zeros(shape, dtype=bool)
        mask[4:20, 4:20, 4:20] = True

        ref_path = self.root / "reference.mrc"
        loc_path = self.root / "local_fsc_like.mrc"
        with mrcfile.new(ref_path, overwrite=True) as m:
            m.set_data(ref)

        save_volume_like_reference(ref_path, loc, loc_path, extra_label="synthetic locres")

        feats = {
            "density_normalized": ref.copy(),
            "multiscale_sigmas": np.array([0.5, 1.0], dtype=np.float32),
        }
        out_npz = self.root / "dataset.npz"
        export_masked_feature_dataset(
            ref_path,
            feats,
            loc,
            mask,
            out_path=out_npz,
            format="npz",
            metadata_extra={"source_tool": "test"},
        )

        data = np.load(out_npz, allow_pickle=False)
        idx = data["indices_zyx"]
        lr = data["local_resolution_A"]
        dens = data["density_normalized"]
        meta = json.loads(str(data["metadata_json"]))
        self.assertEqual(idx.shape[1], 3)
        self.assertEqual(idx.shape[0], lr.shape[0])
        self.assertEqual(idx.shape[0], dens.shape[0])
        self.assertEqual(idx.shape[0], int(mask.sum()))
        self.assertEqual(meta["shape_zyx"], [32, 32, 32])

        side = self.root / "dataset_skipped_features.json"
        self.assertTrue(side.is_file())
        skipped = json.loads(side.read_text(encoding="utf-8"))
        self.assertIn("multiscale_sigmas", skipped)

    def test_load_local_resolution_infer_local_fsc(self) -> None:
        shape = (8, 8, 8)
        p = self.root / "foo_local_fsc_output.mrc"
        vol = np.ones(shape, dtype=np.float32) * 3.5
        with mrcfile.new(p, overwrite=True) as m:
            m.set_data(vol)
        mg = load_local_resolution_map(p, source=None)
        self.assertEqual(mg.shape_zyx, shape)

    def test_load_local_resolution_requires_explicit_when_ambiguous(self) -> None:
        shape = (8, 8, 8)
        p = self.root / "ambiguous.mrc"
        with mrcfile.new(p, overwrite=True) as m:
            m.set_data(np.ones(shape, dtype=np.float32))
        with self.assertRaises(ValueError):
            load_local_resolution_map(p, source=None)

    def test_verify_aligned_local_matches_reference_grid(self) -> None:
        shape = (16, 16, 16)
        ref_path = self.root / "ref2.mrc"
        with mrcfile.new(ref_path, overwrite=True) as m:
            m.set_data(np.random.randn(*shape).astype(np.float32))
        ref_mg = load_map_grid(ref_path, normalize=None)
        loc = np.linspace(1, 4, np.prod(shape), dtype=np.float32).reshape(shape)
        loc_path = self.root / "local_fsc.mrc"
        save_volume_like_reference(ref_path, loc, loc_path)
        loc_mg = load_local_resolution_map(loc_path, source="local_fsc")
        r = verify_same_grid_as_reference(loc_mg, ref_mg)
        self.assertTrue(r.ok)


if __name__ == "__main__":
    unittest.main()
