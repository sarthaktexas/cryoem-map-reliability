"""Run BlocRes local-resolution maps for cohort entries with deposited half-maps.

Writes ``outputs/emd_<ID>/locres_blocres.mrc`` per entry, limited to the depositor
contour mask (``contour_mask.mrc`` + blocres ``-Mask``). After BlocRes finishes,
the map is reheadered onto the deposited reference grid (same origin as half-maps)
via :func:`cryoem_mrc.io.save_volume_like_reference`. Does not modify existing
pipeline outputs (``lh_map_reliability/``, ``local_fsc``, etc.).

Progress is tracked in ``outputs/emd_<ID>/blocres_status.json``.

Example::

    source .venv/bin/activate
    python scripts/run_blocres_local_resolution.py --emd-id 49450
    python scripts/run_blocres_local_resolution.py --all
    python scripts/run_blocres_local_resolution.py --status --emd-id 49450
    python scripts/run_blocres_local_resolution.py --status --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mrcfile
import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc, save_volume_like_reference
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.repo_paths import COHORT_MANIFEST, emd_output_dir

BLOCRES_BIN = Path("/usr/local/bsoft/bin/blocres")
STATUS_NAME = "blocres_status.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_path(emdb_id: str) -> Path:
    return emd_output_dir(emdb_id) / STATUS_NAME


def _locres_path(emdb_id: str) -> Path:
    return emd_output_dir(emdb_id) / "locres_blocres.mrc"


def _mask_path(emdb_id: str) -> Path:
    return emd_output_dir(emdb_id) / "contour_mask.mrc"


def _parse_contour(row: dict[str, str], *, override: float | None) -> float:
    if override is not None:
        return float(override)
    raw = row.get("contour", "").strip()
    if not raw or raw.upper() == "TBD":
        raise ValueError(f"contour not set in manifest ({raw!r}); pass --contour")
    return float(raw)


def _write_contour_mask(reference: Path, contour: float, out_path: Path) -> int:
    """Write 0/1 MRC mask (1 = inside depositor contour) for blocres ``-Mask``."""
    rho = load_mrc(reference, dtype=np.float32)
    mask = build_contour_mask(rho, contour).astype(np.float32)
    n_in = int(mask.sum())
    if n_in < 1000:
        raise ValueError(
            f"contour mask too small ({n_in} voxels at contour={contour}); "
            "check manifest contour"
        )
    save_volume_like_reference(
        reference,
        mask,
        out_path,
        extra_label=f"contour mask >= {contour:g}",
    )
    return n_in


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_status(emdb_id: str, payload: dict) -> Path:
    path = _status_path(emdb_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"emdb_id": str(emdb_id).strip(), "updated_at": _utc_now(), **payload}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _load_status(emdb_id: str) -> dict | None:
    path = _status_path(emdb_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _find_blocres_pid_for_output(out_mrc: Path) -> int | None:
    """Best-effort: locate a live blocres process writing ``out_mrc``."""
    try:
        proc = subprocess.run(
            ["pgrep", "-fl", "blocres"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    candidates = {
        str(out_mrc),
        str(out_mrc.resolve()),
    }
    for line in proc.stdout.splitlines():
        if not any(c in line for c in candidates):
            continue
        pid_s = line.strip().split(maxsplit=1)[0]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if _pid_alive(pid):
            return pid
    return None


def _reconcile_status(emdb_id: str) -> dict:
    """Refresh on-disk status from output file and live blocres PID."""
    out_mrc = _locres_path(emdb_id)
    status = _load_status(emdb_id) or {
        "emdb_id": str(emdb_id).strip(),
        "status": "unknown",
        "output_path": str(out_mrc),
    }

    blocres_pid = status.get("blocres_pid")
    if out_mrc.is_file() and out_mrc.stat().st_size > 0:
        status["status"] = "completed"
        status["output_bytes"] = out_mrc.stat().st_size
        status["blocres_pid"] = None
        status["process_alive"] = False
    else:
        live_pid = _find_blocres_pid_for_output(out_mrc)
        if live_pid is not None:
            status["status"] = "running"
            status["blocres_pid"] = live_pid
            status["process_alive"] = True
        elif status.get("status") == "running" and _pid_alive(blocres_pid):
            status["process_alive"] = True
        elif status.get("status") == "running":
            status["status"] = "failed"
            status["process_alive"] = False
            status.setdefault("error", "blocres process exited before output was written")
            status["blocres_pid"] = None
        else:
            status["process_alive"] = False

    status["updated_at"] = _utc_now()
    _write_status(emdb_id, status)
    return status


def _format_status_line(status: dict) -> str:
    emdb_id = status.get("emdb_id", "?")
    state = status.get("status", "unknown")
    parts = [f"EMD-{emdb_id}", state]
    if status.get("process_alive"):
        parts.append(f"pid={status.get('blocres_pid')}")
    if status.get("started_at"):
        parts.append(f"started={status['started_at']}")
    if status.get("finished_at"):
        parts.append(f"finished={status['finished_at']}")
    out = status.get("output_path")
    if out:
        p = Path(out)
        if p.is_file():
            parts.append(f"out={p.name} ({p.stat().st_size:,} B)")
        elif state == "running":
            parts.append(f"out={p.name} (not written yet)")
    if status.get("error"):
        parts.append(f"error={status['error']}")
    return "  ".join(parts)


def _print_status(emdb_id: str) -> int:
    status = _reconcile_status(emdb_id)
    print(_format_status_line(status), flush=True)
    if status.get("status") == "running" and status.get("process_alive"):
        return 0
    if status.get("status") == "completed":
        return 0
    if status.get("status") in ("failed", "unknown"):
        return 1
    return 0


def _print_status_all(manifest: Path) -> int:
    rows = _manifest_rows_with_halves(manifest)
    rc = 0
    any_line = False
    for row in rows:
        emdb_id = str(row["emdb_id"]).strip()
        status_path = _status_path(emdb_id)
        out_mrc = _locres_path(emdb_id)
        if not status_path.is_file() and not out_mrc.is_file():
            continue
        any_line = True
        code = _print_status(emdb_id)
        rc = max(rc, code)
    if not any_line:
        print("[blocres] no status files or completed outputs found", flush=True)
    return rc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument("--all", action="store_true", help="Process all manifest rows with half-map paths")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--force", action="store_true", help="Re-run even if locres_blocres.mrc exists")
    p.add_argument(
        "--contour",
        type=float,
        default=None,
        help="Override depositor contour from manifest (default: manifest contour column)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print blocres_status.json (use with --emd-id or --all); does not run BlocRes",
    )
    p.add_argument(
        "--realign-only",
        action="store_true",
        help="Reheader existing locres_blocres.mrc onto reference (no BlocRes run)",
    )
    return p.parse_args(argv)


def _require_blocres() -> Path:
    if not BLOCRES_BIN.is_file():
        raise FileNotFoundError(
            f"BlocRes binary not found at {BLOCRES_BIN}. "
            "Install Bsoft and ensure blocres is on PATH at that location."
        )
    return BLOCRES_BIN


def _voxel_size_angstrom_from_mrc(path: Path) -> float:
    """Read isotropic voxel size (Å) from an MRC header."""
    with mrcfile.open(path, permissive=True) as mrc:
        vs = mrc.voxel_size
        sx, sy, sz = float(vs.x), float(vs.y), float(vs.z)
    if not (abs(sx - sy) < 1e-3 and abs(sx - sz) < 1e-3):
        raise ValueError(
            f"{path.name}: expected isotropic voxels for BlocRes, got "
            f"({sx}, {sy}, {sz}) Å"
        )
    return sx


def _mrc_grid_signature(path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Shape (nx, ny, nz) and voxel size (x, y, z) in Å from the MRC header only."""
    with mrcfile.open(path, permissive=True) as mrc:
        h = mrc.header
        vs = mrc.voxel_size
        shape = (int(h.nx), int(h.ny), int(h.nz))
        voxel = (float(vs.x), float(vs.y), float(vs.z))
    return shape, voxel


def _assert_half_maps_match_reference(
    reference: Path,
    half1: Path,
    half2: Path,
) -> float:
    """
    Verify half-maps share grid metadata with the primary map.

    Returns isotropic voxel size (Å) from the reference header.
    """
    ref_shape, ref_voxel = _mrc_grid_signature(reference)
    for label, path in (("half1", half1), ("half2", half2)):
        shape, voxel = _mrc_grid_signature(path)
        if shape != ref_shape:
            raise AssertionError(
                f"{reference.name} vs {label} ({path.name}): "
                f"shape {ref_shape} != {shape}"
            )
        if any(abs(a - b) > 1e-3 for a, b in zip(voxel, ref_voxel)):
            raise AssertionError(
                f"{reference.name} vs {label} ({path.name}): "
                f"voxel size {ref_voxel} != {voxel} Å"
            )
    return _voxel_size_angstrom_from_mrc(reference)


def _prepare_blocres_half_maps(
    reference: Path,
    half1: Path,
    half2: Path,
    out_dir: Path,
) -> tuple[Path, Path, float, list[str]]:
    """
    Return half-map paths on the deposited reference grid for BlocRes.

    When on-disk half-maps differ in shape/voxel/origin (common on EMDB), resample
    onto the reference grid—the same approach as :func:`load_full_and_half_maps`.
    """
    notes: list[str] = []
    try:
        voxel_a = _assert_half_maps_match_reference(reference, half1, half2)
        return half1, half2, voxel_a, notes
    except (AssertionError, ValueError) as exc:
        notes.append(f"on-disk grid mismatch: {exc}")

    bundle = load_full_and_half_maps(
        reference,
        half1,
        half2,
        dtype=np.float32,
        resample_if_needed=True,
    )
    for label in ("half1", "half2"):
        rep = bundle.reports[label]
        if not rep.ok:
            notes.extend(rep.messages)

    out_h1 = out_dir / "aligned_half_map_1.mrc"
    out_h2 = out_dir / "aligned_half_map_2.mrc"
    save_volume_like_reference(
        reference,
        bundle.half1.data,
        out_h1,
        extra_label="aligned half-map 1 for BlocRes",
    )
    save_volume_like_reference(
        reference,
        bundle.half2.data,
        out_h2,
        extra_label="aligned half-map 2 for BlocRes",
    )
    voxel_a = _voxel_size_angstrom_from_mrc(reference)
    notes.append(f"resampled halves -> {out_h1.name}, {out_h2.name}")
    return out_h1, out_h2, voxel_a, notes


def _align_locres_to_reference(reference: Path, locres_path: Path) -> None:
    """
    Copy BlocRes voxel data onto the deposited reference MRC header.

    BlocRes often writes the same (Z, Y, X) array with a centered origin while
    EMDB maps use origin (0, 0, 0). Index-space overlays are correct; only the
    header origin must match for Cα sampling and other physical-coordinate code.
    """
    data = load_mrc(locres_path, dtype=np.float32)
    save_volume_like_reference(
        reference,
        data,
        locres_path,
        extra_label="BlocRes local resolution (aligned to reference)",
    )


def _realign_one(row: dict[str, str]) -> int:
    emdb_id = str(row["emdb_id"]).strip()
    reference = Path(row["reference_mrc"])
    out_mrc = _locres_path(emdb_id)
    if not reference.is_file():
        print(f"[blocres] skip EMD-{emdb_id}: missing reference {reference}", flush=True)
        return 1
    if not out_mrc.is_file():
        print(f"[blocres] skip EMD-{emdb_id}: missing {out_mrc}", flush=True)
        return 1
    _align_locres_to_reference(reference, out_mrc)
    print(f"[blocres] realigned EMD-{emdb_id} -> {out_mrc}", flush=True)
    return 0


def _manifest_rows_with_halves(manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            h1 = row.get("half1_path", "").strip()
            h2 = row.get("half2_path", "").strip()
            if h1 and h2:
                rows.append(row)
    return rows


def _run_one(
    row: dict[str, str],
    *,
    blocres_bin: Path,
    force: bool,
    contour_override: float | None = None,
) -> int:
    emdb_id = str(row["emdb_id"]).strip()
    half1 = Path(row["half1_path"])
    half2 = Path(row["half2_path"])
    reference = Path(row["reference_mrc"])
    out_dir = emd_output_dir(emdb_id)
    out_mrc = out_dir / "locres_blocres.mrc"

    if not half1.is_file() or not half2.is_file():
        print(
            f"[blocres] skip EMD-{emdb_id}: missing half-map(s) "
            f"({half1 if not half1.is_file() else ''} {half2 if not half2.is_file() else ''})",
            flush=True,
        )
        return 0

    if not reference.is_file():
        print(f"[blocres] skip EMD-{emdb_id}: missing reference {reference}", flush=True)
        return 0

    try:
        contour = _parse_contour(row, override=contour_override)
    except ValueError as exc:
        print(f"[blocres] skip EMD-{emdb_id}: {exc}", flush=True)
        return 0

    if out_mrc.is_file() and not force:
        print(f"[blocres] skip EMD-{emdb_id}: exists {out_mrc}", flush=True)
        _write_status(
            emdb_id,
            {
                "status": "completed",
                "started_at": None,
                "finished_at": _utc_now(),
                "output_path": str(out_mrc),
                "output_bytes": out_mrc.stat().st_size,
                "blocres_pid": None,
                "process_alive": False,
                "note": "skipped; output already present",
            },
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        blocres_h1, blocres_h2, voxel_a, grid_notes = _prepare_blocres_half_maps(
            reference, half1, half2, out_dir
        )
    except Exception as exc:
        print(f"[blocres] FAIL EMD-{emdb_id}: align halves — {exc}", file=sys.stderr, flush=True)
        _write_status(
            emdb_id,
            {
                "status": "failed",
                "started_at": _utc_now(),
                "finished_at": _utc_now(),
                "output_path": str(out_mrc),
                "blocres_pid": None,
                "process_alive": False,
                "error": str(exc),
            },
        )
        return 1
    for note in grid_notes:
        print(f"[blocres] EMD-{emdb_id}: {note}", flush=True)
    mask_mrc = _mask_path(emdb_id)
    try:
        n_mask = _write_contour_mask(reference, contour, mask_mrc)
    except ValueError as exc:
        print(f"[blocres] FAIL EMD-{emdb_id}: mask — {exc}", file=sys.stderr, flush=True)
        _write_status(
            emdb_id,
            {
                "status": "failed",
                "started_at": _utc_now(),
                "finished_at": _utc_now(),
                "output_path": str(out_mrc),
                "blocres_pid": None,
                "process_alive": False,
                "error": str(exc),
            },
        )
        return 1

    sampling = f"{voxel_a:g},{voxel_a:g},{voxel_a:g}"
    cmd = [
        str(blocres_bin),
        "-sampling",
        sampling,
        "-box",
        "15",
        "-cutoff",
        "0.143",
        "-Mask",
        f"{mask_mrc},0.5",
        str(blocres_h1),
        str(blocres_h2),
        str(out_mrc),
    ]
    started = _utc_now()
    print(
        f"[blocres] EMD-{emdb_id}: mask={n_mask:,} voxels at contour={contour:g}",
        flush=True,
    )
    print(f"[blocres] EMD-{emdb_id}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _write_status(
        emdb_id,
        {
            "status": "running",
            "started_at": started,
            "finished_at": None,
            "output_path": str(out_mrc),
            "blocres_pid": proc.pid,
            "process_alive": True,
            "command": cmd,
            "reference_mrc": str(reference),
            "half1": str(blocres_h1),
            "half2": str(blocres_h2),
            "half1_source": str(half1),
            "half2_source": str(half2),
            "voxel_size_A": voxel_a,
            "contour": contour,
            "mask_path": str(mask_mrc),
            "n_mask_voxels": n_mask,
            "grid_notes": grid_notes,
        },
    )
    print(f"[blocres] status -> {_status_path(emdb_id)}", flush=True)
    stdout, stderr = proc.communicate()
    finished = _utc_now()
    if proc.returncode != 0:
        print(f"[blocres] FAIL EMD-{emdb_id}: exit {proc.returncode}", file=sys.stderr, flush=True)
        if stderr:
            print(stderr, file=sys.stderr, flush=True)
        _write_status(
            emdb_id,
            {
                "status": "failed",
                "started_at": started,
                "finished_at": finished,
                "output_path": str(out_mrc),
                "blocres_pid": None,
                "process_alive": False,
                "error": stderr.strip() or f"exit code {proc.returncode}",
                "returncode": proc.returncode,
            },
        )
        return 1

    if not out_mrc.is_file():
        print(f"[blocres] FAIL EMD-{emdb_id}: missing output {out_mrc}", file=sys.stderr, flush=True)
        _write_status(
            emdb_id,
            {
                "status": "failed",
                "started_at": started,
                "finished_at": finished,
                "output_path": str(out_mrc),
                "blocres_pid": None,
                "process_alive": False,
                "error": "blocres exited 0 but output MRC missing",
                "returncode": proc.returncode,
            },
        )
        return 1

    _align_locres_to_reference(reference, out_mrc)

    _write_status(
        emdb_id,
        {
            "status": "completed",
            "started_at": started,
            "finished_at": finished,
            "output_path": str(out_mrc),
            "output_bytes": out_mrc.stat().st_size if out_mrc.is_file() else 0,
            "aligned_to_reference": str(reference),
            "blocres_pid": None,
            "process_alive": False,
            "returncode": 0,
        },
    )
    print(f"[blocres] ok EMD-{emdb_id} -> {out_mrc} (aligned to {reference.name})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.status:
        if args.emd_id:
            return _print_status(args.emd_id.strip())
        if args.all:
            return _print_status_all(args.manifest)
        print("Specify --status with --emd-id or --all", file=sys.stderr)
        return 2

    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    rows = _manifest_rows_with_halves(args.manifest)
    if args.emd_id:
        target = args.emd_id.strip()
        rows = [r for r in rows if str(r["emdb_id"]).strip() == target]
        if not rows:
            print(f"[blocres] ERROR: EMD-{target} not in manifest or lacks half-map paths", file=sys.stderr)
            return 2

    if args.realign_only:
        rc = 0
        for row in rows:
            rc = max(rc, _realign_one(row))
        return rc

    try:
        blocres_bin = _require_blocres()
    except FileNotFoundError as exc:
        print(f"[blocres] ERROR: {exc}", file=sys.stderr)
        return 2

    rc = 0
    for row in rows:
        rc = max(rc, _run_one(
            row,
            blocres_bin=blocres_bin,
            force=args.force,
            contour_override=args.contour,
        ))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
