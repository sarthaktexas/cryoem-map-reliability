"""Repository layout conventions: ``data/``, ``outputs/emd_<ID>/``, ``cohort/manifest.csv``."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data")
OUTPUTS_ROOT = Path("outputs")
PDB_ROOT = Path("pdb")
COHORT_MANIFEST = Path("cohort/manifest.csv")


def emd_output_dir(emdb_id: str | int) -> Path:
    return OUTPUTS_ROOT / f"emd_{str(emdb_id).strip()}"


def analysis_dir(emdb_id: str | int) -> Path:
    return emd_output_dir(emdb_id) / "analysis"


def analysis_localres_dir(emdb_id: str | int) -> Path:
    return emd_output_dir(emdb_id) / "analysis_localres"


def lh_map_reliability_dir(emdb_id: str | int) -> Path:
    """Per-map LH reliability bundle: ``outputs/emd_<ID>/lh_map_reliability/``."""
    return emd_output_dir(emdb_id) / "lh_map_reliability"


def thesis_overview_dir(emdb_id: str | int = "49450") -> Path:
    return emd_output_dir(emdb_id) / "thesis_overview"


def halfmap_metrics_npz(emdb_id: str | int) -> Path:
    return analysis_dir(emdb_id) / "halfmap_metrics.npz"


def sensitivity_local_fsc_dir() -> Path:
    return OUTPUTS_ROOT / "sensitivity" / "local_fsc"


def archive_rigidity_vs_mechanics_dir() -> Path:
    return OUTPUTS_ROOT / "archive" / "rigidity_vs_mechanics"


def bfactor_conformation_pairs_dir() -> Path:
    return OUTPUTS_ROOT / "bfactor_conformation_pairs"


def find_features_npz(data_dir: Path, emdb_id: str | int, contour: float) -> Path | None:
    """
    Locate averaged-map feature NPZ for one EMDB entry.

    Prefer the depositor-contour tag, then ``t0000`` (unsharpened avg maps whose
    in-mask max sits below the depositor contour — see cohort pipeline), then any match.
    """
    emd = f"emd_{str(emdb_id).strip()}"
    tag = f"t{int(round(float(contour) * 1000)):04d}"
    candidates = [
        data_dir / f"{emd}_avg_features_{tag}.npz",
        data_dir / f"{emd}_avg_features_t{int(round(contour * 1000)):04d}.npz",
        data_dir / f"{emd}_avg_features_t0000.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_dir.glob(f"{emd}_avg_features*.npz"))
    return matches[0] if matches else None
