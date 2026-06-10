"""Generate model density maps on a cryo-EM reference grid.

FSC-Q normally uses Xmipp ``xmipp_volume_from_pdb`` (electron form factors). When Xmipp
is unavailable locally, :func:`generate_gaussian_model_map` splats isotropic Gaussians at
non-hydrogen atom positions — sufficient for BlocRes local-resolution subtraction when
both FSC arms use the same map generator and mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cryoem_mrc.io import save_volume_like_reference
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.structure_validation import physical_xyz_to_voxel_indices


def _atomic_weight(element: str) -> float:
    """Crude electron-count proxy for Gaussian peak height."""
    el = (element or "C").strip().upper()
    z = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "FE": 26, "ZN": 30, "MG": 12}
    if el in z:
        return float(z[el])
    return float(z.get(el[0], 6))


def generate_gaussian_model_map(
    structure_path: str | Path,
    reference_path: str | Path,
    out_path: str | Path,
    *,
    sigma_a: float = 1.5,
    sigma_scale_resolution: float | None = None,
    global_resolution_a: float | None = None,
) -> Path:
    """
    Splat Gaussians at non-hydrogen atom sites onto ``reference_path``'s grid.

    ``sigma_a`` is the base Gaussian width (Å). When ``sigma_scale_resolution`` and
    ``global_resolution_a`` are set, ``sigma_a`` is overridden by
    ``sigma_scale_resolution * global_resolution_a``.
    """
    import gemmi

    structure_path = Path(structure_path)
    reference_path = Path(reference_path)
    out_path = Path(out_path)

    if sigma_scale_resolution is not None and global_resolution_a is not None:
        sigma_a = float(sigma_scale_resolution * global_resolution_a)

    grid = load_map_grid(reference_path, dtype=np.float32)
    vol = np.zeros(grid.shape_zyx, dtype=np.float32)
    vsz, vsy, vsx = grid.voxel_size_zyx
    rad_a = 3.0 * sigma_a
    rz = max(1, int(np.ceil(rad_a / vsz)))
    ry = max(1, int(np.ceil(rad_a / vsy)))
    rx = max(1, int(np.ceil(rad_a / vsx)))
    inv2s2 = 1.0 / (2.0 * sigma_a * sigma_a)
    nz, ny, nx = vol.shape

    st = gemmi.read_structure(str(structure_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    for model in st:
        for chain in model:
            for residue in chain:
                if residue.entity_type == gemmi.EntityType.Water:
                    continue
                for atom in residue:
                    w = _atomic_weight(atom.element.name)
                    iz, iy, ix = physical_xyz_to_voxel_indices(
                        float(atom.pos.x), float(atom.pos.y), float(atom.pos.z), grid
                    )
                    z0, z1 = max(0, iz - rz), min(nz, iz + rz + 1)
                    y0, y1 = max(0, iy - ry), min(ny, iy + ry + 1)
                    x0, x1 = max(0, ix - rx), min(nx, ix + rx + 1)
                    zz = (np.arange(z0, z1, dtype=np.float64) - iz) * vsz
                    yy = (np.arange(y0, y1, dtype=np.float64) - iy) * vsy
                    xx = (np.arange(x0, x1, dtype=np.float64) - ix) * vsx
                    dz2 = zz[:, None, None] ** 2
                    dy2 = yy[None, :, None] ** 2
                    dx2 = xx[None, None, :] ** 2
                    kern = w * np.exp(-(dz2 + dy2 + dx2) * inv2s2).astype(np.float32)
                    vol[z0:z1, y0:y1, x0:x1] += kern
        break

    save_volume_like_reference(
        reference_path,
        vol,
        out_path,
        extra_label="Gaussian atom model map (gemmi fallback)",
    )
    return out_path
