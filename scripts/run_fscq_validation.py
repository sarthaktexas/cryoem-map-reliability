"""FSC-Q validation: model-free constraint V vs the model-dependent FSC-Q metric.

FSC-Q (Ramírez-Aportela et al., *Nat Commun* 2021) is computed **with the published
binaries**, not reimplemented:

    FSC-Q = V_FSC(map, model_map) − V_FSC(half1, half2)        [Eq. 1, in Å]

where each ``V_FSC`` local-resolution volume comes from **BlocRes** (Bsoft) at the
gold-standard FSC = 0.143 threshold, the model density map is generated with
**xmipp_volume_from_pdb** (electron scattering form factors), and both BlocRes runs
use the *same* window/cutoff/mask. The only non-binary step is the Eq. 1 subtraction
itself (the metric's definition) and aggregation of the result to Cα — neither of
which re-derives local resolution. This mirrors how the cohort local-resolution
benchmark uses real BlocRes rather than the in-repo FSC estimator.

Binaries are resolved from the environment so this runs on an HPC module stack:

    export XMIPP_RUN="scipion3 run"          # or e.g. "" if xmipp_* are on PATH
    export BLOCRES_BIN=/path/to/bsoft/bin/blocres
    python scripts/run_fscq_validation.py --emd-id 49450
    python scripts/run_fscq_validation.py --anchors
    python scripts/run_fscq_validation.py --emd-id 33734   # discordant map (see §4.4)

Outputs under ``outputs/emd_<ID>/fscq/``:
    model_map.mrc, v_fsc_map_model.mrc, v_fsc_half.mrc, fscq.mrc,
    residue_fscq.csv, fscq_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc, save_volume_like_reference
from cryoem_mrc.local_resolution import aggregate_locres_to_ca
from cryoem_mrc.metric_comparison import load_all_metrics
from cryoem_mrc.model_map import generate_gaussian_model_map
from cryoem_mrc.repo_paths import (
    ANCHOR_EMDB_ID,
    BFACTOR_VALIDATION_EMDB_IDS,
    COHORT_MANIFEST,
    emd_output_dir,
)
from cryoem_mrc.structure_validation import load_cohort_manifest_row

# Default to the same Bsoft binary the cohort BlocRes benchmark already uses.
BLOCRES_BIN = Path(os.environ.get("BLOCRES_BIN", "/usr/local/bsoft/bin/blocres"))
# Prefix to launch Xmipp programs (e.g. "scipion3 run"); empty if xmipp_* are on PATH.
XMIPP_RUN = os.environ.get("XMIPP_RUN", "").strip()

FSC_CUTOFF = 0.143
# Below this model-vs-map correlation, treat the generated model map as misaligned
# and refuse to compute FSC-Q (the result would be meaningless).
MIN_MODEL_MAP_CC = 0.30


def _xmipp_cmd(program: str, args: list[str]) -> list[str]:
    """Build a command list for an Xmipp program honouring ``XMIPP_RUN``."""
    prefix = shlex.split(XMIPP_RUN) if XMIPP_RUN else []
    return [*prefix, program, *args]


def _xmipp_available() -> bool:
    """True when ``xmipp_volume_from_pdb`` can be launched."""
    import shutil

    if XMIPP_RUN:
        return shutil.which(shlex.split(XMIPP_RUN)[0]) is not None
    return shutil.which("xmipp_volume_from_pdb") is not None


def _run(cmd: list[str], *, label: str) -> None:
    print(f"[fscq] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")


def _voxel_size_a(path: Path) -> float:
    with mrcfile.open(path, permissive=True) as mrc:
        vs = mrc.voxel_size
        sx, sy, sz = float(vs.x), float(vs.y), float(vs.z)
    if not (abs(sx - sy) < 1e-3 and abs(sx - sz) < 1e-3):
        raise ValueError(f"{path.name}: anisotropic voxels ({sx},{sy},{sz}); FSC-Q expects isotropic")
    return sx


def _box_voxels(resolution_a: float, voxel_a: float) -> int:
    """FSC-Q window: 5× reported resolution, or 11 voxels for sub-2 Å maps.

    Follows Ramírez-Aportela et al. (2021): a sliding window of five times the
    reported resolution, fixed at 11 voxels below 2 Å to avoid instability.
    """
    if resolution_a < 2.0:
        return 11
    return max(11, int(round(5.0 * resolution_a / voxel_a)))


def _shape(path: Path) -> tuple[int, int, int]:
    with mrcfile.open(path, permissive=True) as mrc:
        h = mrc.header
        return int(h.nz), int(h.ny), int(h.nx)


def _generate_model_map(
    pdb_path: Path,
    reference: Path,
    out_mrc: Path,
    *,
    voxel_a: float,
    resolution_a: float,
    backend: str,
) -> Path:
    """Render the fitted model to a density map on the reference grid via Xmipp.

    Uses electron scattering form factors (``xmipp_volume_from_pdb`` default). The PDB
    is taken in the deposited (map) frame — no re-centering — then the result is
    reheadered onto the reference grid so it index-aligns with the experimental map.
    """
    if backend == "gemmi":
        return generate_gaussian_model_map(
            pdb_path,
            reference,
            out_mrc,
            sigma_scale_resolution=0.25,
            global_resolution_a=resolution_a,
        )

    out_dir = out_mrc.parent
    nz, ny, nx = _shape(reference)
    vol_root = out_dir / "model_vol"
    # xmipp_volume_from_pdb writes <root>.vol; --size takes a single cubic dim, so use
    # the max box edge and crop/reheader to the reference afterwards.
    box = max(nz, ny, nx)
    _run(
        _xmipp_cmd(
            "xmipp_volume_from_pdb",
            ["-i", str(pdb_path), "-o", str(vol_root), "--sampling", f"{voxel_a:g}", "--size", str(box)],
        ),
        label="volume_from_pdb",
    )
    raw_vol = vol_root.with_suffix(".vol")
    tmp_mrc = out_dir / "model_vol.mrc"
    _run(
        _xmipp_cmd("xmipp_image_convert", ["-i", str(raw_vol), "-o", f"{tmp_mrc}:mrc"]),
        label="image_convert",
    )
    data = load_mrc(tmp_mrc, dtype=np.float32)
    # Center-crop the cubic model volume back to the reference shape if needed.
    if data.shape != (nz, ny, nx):
        cz, cy, cx = data.shape
        sz, sy, sx = (cz - nz) // 2, (cy - ny) // 2, (cx - nx) // 2
        data = data[sz : sz + nz, sy : sy + ny, sx : sx + nx]
    save_volume_like_reference(reference, data, out_mrc, extra_label="model density (xmipp_volume_from_pdb)")
    for tmp in (raw_vol, tmp_mrc, vol_root.with_suffix(".xmd")):
        if tmp.is_file():
            tmp.unlink()
    return out_mrc


def _masked_cc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0
    x, y = a[sel].ravel(), b[sel].ravel()
    if x.size < 100 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _run_blocres(map1: Path, map2: Path, mask: Path, out_mrc: Path, *, voxel_a: float, box: int) -> Path:
    sampling = f"{voxel_a:g},{voxel_a:g},{voxel_a:g}"
    cmd = [
        str(BLOCRES_BIN),
        "-sampling", sampling,
        "-box", str(box),
        "-cutoff", str(FSC_CUTOFF),
        "-Mask", f"{mask},0.5",
        str(map1),
        str(map2),
        str(out_mrc),
    ]
    _run(cmd, label=f"blocres box={box}")
    return out_mrc


def _aggregate_signed_to_ca(
    fscq_mrc: Path, pdb_path: Path, reference: Path, *, radius_a: float
) -> pd.DataFrame:
    """Mean FSC-Q (signed Å) in a sphere around each Cα. FSC-Q can be negative."""
    df = aggregate_locres_to_ca(
        fscq_mrc,
        pdb_path,
        radius_angstrom=radius_a,
        reference_path=reference,
        positive_only=False,
        value_column="fscq_mean",
    )
    return df[["chain", "seq_num", "fscq_mean"]]


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan"), int(m.sum())
    r = stats.spearmanr(a[m], b[m]).statistic
    return float(r), int(m.sum())


def run_one(
    emd_id: str,
    *,
    manifest: Path,
    radius_a: float,
    box_override: int | None,
    force: bool,
    compute_only: bool = False,
    model_backend: str = "auto",
) -> tuple[int, dict | None]:
    emd_id = str(emd_id).strip()
    row = load_cohort_manifest_row(manifest, emd_id)
    reference = Path(row["reference_mrc"])
    half1 = Path(row["half1_path"])
    half2 = Path(row["half2_path"])
    pdb = Path(row.get("flexibility_path_or_pdb", "").strip())
    try:
        contour = float(row["contour"])
        resolution = float(row["global_resolution_a"])
    except (KeyError, ValueError) as exc:
        print(f"[fscq] skip EMD-{emd_id}: missing contour/resolution ({exc})", file=sys.stderr)
        return 1, None

    for label, p in (("reference", reference), ("half1", half1), ("half2", half2), ("model", pdb)):
        if not p.is_file():
            print(f"[fscq] skip EMD-{emd_id}: missing {label} ({p})", file=sys.stderr)
            return 1, None

    out_dir = emd_output_dir(emd_id) / "fscq"
    out_dir.mkdir(parents=True, exist_ok=True)
    fscq_mrc = out_dir / "fscq.mrc"
    if fscq_mrc.is_file() and not force:
        print(f"[fscq] EMD-{emd_id}: fscq.mrc exists (use --force to recompute); correlating only", flush=True)
    else:
        voxel_a = _voxel_size_a(reference)
        box = box_override if box_override is not None else _box_voxels(resolution, voxel_a)

        # Mask enclosing the macromolecule (depositor contour) — identical mask for
        # both BlocRes runs, as FSC-Q requires.
        ref_rho = load_mrc(reference, dtype=np.float32)
        mask_arr = build_contour_mask(ref_rho, contour).astype(np.float32)
        mask_mrc = out_dir / "fscq_mask.mrc"
        save_volume_like_reference(reference, mask_arr, mask_mrc, extra_label=f"contour mask >= {contour:g}")

        backend = model_backend
        if backend == "auto":
            backend = "xmipp" if _xmipp_available() else "gemmi"
        print(f"[fscq] EMD-{emd_id}: model-map backend = {backend}", flush=True)
        model_mrc = _generate_model_map(
            pdb,
            reference,
            out_dir / "model_map.mrc",
            voxel_a=voxel_a,
            resolution_a=resolution,
            backend=backend,
        )

        cc = _masked_cc(ref_rho, load_mrc(model_mrc, dtype=np.float32), mask_arr)
        print(f"[fscq] EMD-{emd_id}: model-vs-map CC in mask = {cc:.3f} (box={box}, res={resolution} Å)", flush=True)
        if not np.isfinite(cc) or cc < MIN_MODEL_MAP_CC:
            print(
                f"[fscq] ABORT EMD-{emd_id}: model map misaligned (CC={cc:.3f} < {MIN_MODEL_MAP_CC}). "
                "Grid/origin handling failed — run the canonical Scipion 'validate fsc-q' "
                "protocol for this entry instead (see docs).",
                file=sys.stderr,
            )
            return 2, None

        v_map_model = _run_blocres(reference, model_mrc, mask_mrc, out_dir / "v_fsc_map_model.mrc", voxel_a=voxel_a, box=box)
        v_half = _run_blocres(half1, half2, mask_mrc, out_dir / "v_fsc_half.mrc", voxel_a=voxel_a, box=box)

        a = load_mrc(v_map_model, dtype=np.float32)
        b = load_mrc(v_half, dtype=np.float32)
        fscq = np.where(mask_arr > 0, a - b, np.nan).astype(np.float32)
        save_volume_like_reference(
            reference,
            np.nan_to_num(fscq, nan=0.0),
            fscq_mrc,
            extra_label="FSC-Q = V_map_model - V_half (A)",
        )

    if compute_only:
        # Cluster path: BlocRes volume written; defer Cα/V correlation to a local run
        # (where reliability.npz / half-map metrics live). Re-running locally finds
        # the existing fscq.mrc and only correlates.
        print(f"[fscq] EMD-{emd_id}: compute-only — wrote {fscq_mrc}; skip correlation", flush=True)
        return 0, None

    # Correlate FSC-Q against the model-free constraint V (and B-factor) at Cα.
    metrics = load_all_metrics(emd_id, manifest=manifest)
    fscq_ca = _aggregate_signed_to_ca(fscq_mrc, pdb, reference, radius_a=radius_a)
    merged = metrics.merge(fscq_ca, on=["chain", "seq_num"], how="left")
    in_mask = merged["in_contour_mask"].astype(bool) if "in_contour_mask" in merged else np.ones(len(merged), bool)
    use = merged[in_mask]

    rho_v, n_v = _spearman(use["fscq_mean"].to_numpy(float), use["v_metric"].to_numpy(float))
    rho_b, n_b = _spearman(use["fscq_mean"].to_numpy(float), use["b_factor"].to_numpy(float))
    rho_var, n_var = float("nan"), 0
    if "local_variance" in use.columns:
        rho_var, n_var = _spearman(
            use["fscq_mean"].to_numpy(float), use["local_variance"].to_numpy(float)
        )

    merged.to_csv(out_dir / "residue_fscq.csv", index=False)
    stats_payload = {
        "emdb_id": emd_id,
        "n_in_mask": int(in_mask.sum()),
        "spearman_fscq_vs_V": rho_v,
        "n_fscq_vs_V": n_v,
        "spearman_fscq_vs_b": rho_b,
        "n_fscq_vs_b": n_b,
        "spearman_fscq_vs_variance": rho_var,
        "n_fscq_vs_variance": n_var,
        "fscq_median_A": float(np.nanmedian(use["fscq_mean"].to_numpy(float))),
    }
    (out_dir / "fscq_stats.json").write_text(json.dumps(stats_payload, indent=2) + "\n")
    print(
        f"[fscq] EMD-{emd_id}: ρ(FSC-Q, V)={rho_v:+.3f} (n={n_v}), "
        f"ρ(FSC-Q, B)={rho_b:+.3f}, ρ(FSC-Q, var)={rho_var:+.3f} "
        f"→ {out_dir/'fscq_stats.json'}",
        flush=True,
    )
    return 0, stats_payload


def _build_cohort_figure(manifest: Path, dpi: int) -> Path | None:
    """ρ(FSC-Q, V) per-structure ranking + ρ vs resolution (mirrors qscore_vs_V_cohort)."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from style.nature import PALETTES, apply, label_panel, savefig as save_nature
    from cryoem_mrc.repo_paths import OUTPUTS_ROOT

    csv_path = OUTPUTS_ROOT / "cohort_summary" / "fscq_correlations.csv"
    if not csv_path.is_file():
        print(f"[fscq] no cohort CSV at {csv_path}", file=sys.stderr)
        return None

    res_by_id, name_by_id = {}, {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            name_by_id[eid] = row.get("display_name", "").strip()
            try:
                res_by_id[eid] = float(row["global_resolution_a"])
            except (KeyError, ValueError):
                pass

    recs = []
    for r in csv.DictReader(csv_path.open()):
        raw = str(r.get("spearman_fscq_vs_V", ""))
        if raw in ("", "nan"):
            continue
        rho = float(raw)
        if not np.isfinite(rho):
            continue
        eid = str(r["emdb_id"]).strip()
        recs.append({"emdb_id": eid, "rho": rho, "res": res_by_id.get(eid, float("nan"))})
    if not recs:
        print("[fscq] no finite ρ rows for figure", file=sys.stderr)
        return None

    recs.sort(key=lambda d: d["rho"])
    rhos = np.array([d["rho"] for d in recs])
    res = np.array([d["res"] for d in recs])
    labels = [f"EMD-{d['emdb_id']}" for d in recs]
    median_rho = float(np.median(rhos))

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(11.0, 6.5))
    apply(ax_bar)
    ypos = np.arange(len(recs))
    ax_bar.barh(ypos, rhos, color=PALETTES["categorical"][0], edgecolor="0.2", linewidth=0.4)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=6)
    ax_bar.axvline(0.0, color="0.3", linewidth=0.6)
    ax_bar.axvline(median_rho, color=PALETTES["categorical"][1], linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Spearman ρ(FSC-Q, V), in-mask Cα")
    ax_bar.set_title(f"Per-structure FSC-Q vs V (median ρ={median_rho:+.2f})")
    label_panel(ax_bar, "a")

    apply(ax_sc)
    m = np.isfinite(res)
    ax_sc.scatter(res[m], rhos[m], s=24, c=PALETTES["categorical"][0], edgecolors="0.2", linewidths=0.4)
    ax_sc.axhline(0.0, color="0.3", linewidth=0.6)
    ax_sc.set_xlabel("Global resolution (Å)")
    ax_sc.set_ylabel("Spearman ρ(FSC-Q, V)")
    ax_sc.set_title("ρ(FSC-Q, V) vs resolution")
    label_panel(ax_sc, "b")

    fig.suptitle("FSC-Q vs constraint V — cohort summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTPUTS_ROOT / "cohort_summary" / "fscq_vs_V_cohort"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    print(f"[fscq] cohort figure → {out}.png", flush=True)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument("--anchors", action="store_true", help=f"Run thesis anchors: {', '.join(BFACTOR_VALIDATION_EMDB_IDS)}")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--radius", type=float, default=2.0, help="Cα sphere radius (Å) for aggregation")
    p.add_argument("--box", type=int, default=None, help="Override BlocRes window (voxels); default 5×res or 11 (<2 Å)")
    p.add_argument("--force", action="store_true", help="Recompute fscq.mrc even if present")
    p.add_argument(
        "--compute-only",
        action="store_true",
        help="Cluster mode: write fscq.mrc via Xmipp/BlocRes and stop (defer Cα/V correlation to a local run)",
    )
    p.add_argument("--cohort-summary", action="store_true", help="Write outputs/cohort_summary/fscq_correlations.csv")
    p.add_argument("--cohort-figure", action="store_true", help="Build fscq_vs_V_cohort.png from fscq_correlations.csv")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--model-backend",
        choices=("auto", "xmipp", "gemmi"),
        default="auto",
        help="Model-map generator (default auto: xmipp if on PATH, else gemmi Gaussians)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.cohort_figure and not (args.emd_id or args.anchors):
        out = _build_cohort_figure(args.manifest, args.dpi)
        return 0 if out is not None else 2

    if not args.emd_id and not args.anchors:
        print("Specify --emd-id, --anchors, or --cohort-figure", file=sys.stderr)
        return 2
    if not BLOCRES_BIN.is_file():
        print(f"[fscq] ERROR: BlocRes not found at {BLOCRES_BIN}; set BLOCRES_BIN", file=sys.stderr)
        return 2

    ids = list(BFACTOR_VALIDATION_EMDB_IDS) if args.anchors else [args.emd_id.strip()]
    rc = 0
    records: list[dict] = []
    for emd_id in ids:
        try:
            code, rec = run_one(
                emd_id,
                manifest=args.manifest,
                radius_a=args.radius,
                box_override=args.box,
                force=args.force,
                compute_only=args.compute_only,
                model_backend=args.model_backend,
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"[fscq] FAIL EMD-{emd_id}: {exc}", file=sys.stderr)
            rc = max(rc, 1)
            continue
        rc = max(rc, code)
        if rec is not None:
            records.append(rec)

    if args.cohort_summary and records:
        from cryoem_mrc.repo_paths import OUTPUTS_ROOT

        out = OUTPUTS_ROOT / "cohort_summary" / "fscq_correlations.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"[fscq] cohort summary → {out}", flush=True)

    if args.cohort_figure:
        _build_cohort_figure(args.manifest, args.dpi)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
