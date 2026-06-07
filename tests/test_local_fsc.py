"""Tests for windowed local FSC resolution estimation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import mrcfile

from cryoem_mrc.local_fsc import (
    _build_radial_shell_indices,
    _build_window,
    _fsc_curve_from_patches,
    _resolution_from_fsc,
    compute_local_fsc_resolution,
    save_local_fsc_resolution_mrc,
)


def _gaussian_blob(shape: tuple[int, int, int], sigma_vox: float, center: tuple[int, int, int]) -> np.ndarray:
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    cz, cy, cx = center
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-r2 / (2.0 * sigma_vox**2)).astype(np.float32)


class TestLocalFscHelpers(unittest.TestCase):
    def test_shell_indices_length_matches_rfft(self) -> None:
        p = 17
        shell_idx, n_shells = _build_radial_shell_indices(p)
        n_fft = p * p * (p // 2 + 1)
        self.assertEqual(shell_idx.shape[0], n_fft)
        self.assertGreater(n_shells, 0)

    def test_identical_patches_high_fsc(self) -> None:
        p = 17
        shell_idx, n_shells = _build_radial_shell_indices(p)
        blob = _gaussian_blob((p, p, p), 2.0, (p // 2, p // 2, p // 2))
        w3 = _build_window(p, "hann")
        fsc = _fsc_curve_from_patches(blob * w3, blob * w3, shell_idx, n_shells)
        self.assertGreater(float(fsc[1:5].min()), 0.9)

    def test_resolution_from_fsc_monotone_drop(self) -> None:
        fsc = np.array([1.0, 0.9, 0.5, 0.1, 0.05])
        res = _resolution_from_fsc(fsc, 0.143, patch_size=17, voxel_size_a=1.0)
        self.assertGreater(res, 2.0)
        self.assertLessEqual(res, 17.0)


class TestLocalFscSynthetic(unittest.TestCase):
    def test_two_noisy_halves_blob_near_nyquist(self) -> None:
        """Two noisy copies of the same blob -> resolution near 2 * voxel_size."""
        shape = (64, 64, 64)
        vox = 1.0
        center = (32, 32, 32)
        signal = _gaussian_blob(shape, sigma_vox=2.0, center=center)
        rng = np.random.default_rng(42)
        half1 = signal + rng.normal(0, 0.05, size=shape).astype(np.float32)
        half2 = signal + rng.normal(0, 0.05, size=shape).astype(np.float32)
        mask = np.zeros(shape, dtype=bool)
        mask[20:44, 20:44, 20:44] = True

        res_map = compute_local_fsc_resolution(
            half1,
            half2,
            vox,
            patch_size=17,
            stride=4,
            fsc_threshold=0.143,
            window="hann",
            mask=mask,
            min_voxels_for_fsc=32,
            n_jobs=1,
            require_mask=True,
        )
        inside = res_map[mask]
        inside = inside[np.isfinite(inside)]
        self.assertGreater(inside.size, 10)
        nyquist = 2.0 * vox
        patch_hi = 17.0 * vox
        self.assertTrue(np.all(inside >= nyquist - 1e-6))
        self.assertTrue(np.all(inside <= patch_hi + 1e-6))
        # Tight blob + correlated halves -> near Nyquist (allow modest slack)
        # Correlated halves on a compact blob: finer than patch Nyquist, coarser than 2 Å
        self.assertLess(float(np.median(inside)), 10.0 * vox)

    def test_clip_bounds(self) -> None:
        shape = (48, 48, 48)
        vox = 0.93
        rng = np.random.default_rng(0)
        h1 = rng.standard_normal(shape).astype(np.float32)
        h2 = rng.standard_normal(shape).astype(np.float32)
        res = compute_local_fsc_resolution(
            h1, h2, vox, patch_size=13, stride=8, mask=None, n_jobs=1,
            require_mask=False,
        )
        finite = res[np.isfinite(res)]
        self.assertGreater(finite.size, 0)
        self.assertGreaterEqual(float(finite.min()), 2.0 * vox - 1e-5)
        self.assertLessEqual(float(finite.max()), 13.0 * vox + 1e-5)


class TestSaveLocalFscMrc(unittest.TestCase):
    def test_save_and_label(self) -> None:
        shape = (16, 16, 16)
        ref = np.zeros(shape, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.mrc"
            out_path = Path(tmp) / "out_local_fsc.mrc"
            with mrcfile.new(ref_path, overwrite=True) as m:
                m.set_data(ref)
            vol = np.full(shape, 4.5, dtype=np.float32)
            save_local_fsc_resolution_mrc(
                vol, ref_path, out_path,
                fsc_threshold=0.143, patch_size=17, stride=4,
            )
            self.assertTrue(out_path.is_file())
            with mrcfile.open(out_path) as m:
                labels = [str(l).lower() for l in m.header.label]
            self.assertTrue(any("local_fsc" in lab for lab in labels))


if __name__ == "__main__":
    unittest.main()
