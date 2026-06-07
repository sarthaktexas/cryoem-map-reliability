"""Driver: aligned half-maps -> windowed local FSC resolution MRC (Å).

Example (EMD-49450):

    python scripts/run_local_fsc.py \\
        --half1 emd_49450/emd_49450_half_map_1.map \\
        --half2 emd_49450/emd_49450_half_map_2.map \\
        --reference emd_49450/emd_49450.map \\
        --contour 0.116 \\
        --patch-size 17 \\
        --stride 4 \\
        --fsc-threshold 0.143 \\
        --out emd_49450/emd_49450_local_fsc_t0143_P17_s4.mrc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.local_fsc import compute_local_fsc_resolution, save_local_fsc_resolution_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--half1", required=True, type=Path)
    p.add_argument("--half2", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path,
                   help="Reference MRC for grid alignment and output header")
    p.add_argument("--contour", type=float, default=0.116,
                   help="Density contour for patch-center mask (default 0.116)")
    p.add_argument(
        "--no-mask",
        action="store_true",
        help="DANGEROUS: full 430³ box — can exhaust RAM. Default uses contour mask.",
    )
    p.add_argument("--patch-size", type=int, default=17)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--fsc-threshold", type=float, default=0.143)
    p.add_argument("--window", choices=("hann", "cosine", "none"), default="hann")
    p.add_argument("--min-voxels-for-fsc", type=int, default=64)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Thread workers over patch centers (default 1; safe for large maps)",
    )
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args(argv)


def _write_summary(
    path: Path,
    *,
    res: np.ndarray,
    mask: np.ndarray | None,
    voxel_size_a: float,
    patch_size: int,
    args: argparse.Namespace,
) -> None:
    if mask is not None:
        vals = res[mask.astype(bool)]
        vals = vals[np.isfinite(vals)]
        n_mask = int(mask.sum())
        frac_nan = float(np.mean(~np.isfinite(res[mask]))) if n_mask else float("nan")
    else:
        vals = res[np.isfinite(res.ravel())]
        frac_nan = float(np.mean(~np.isfinite(res)))
        n_mask = int(res.size)
    lines = [
        "local_fsc summary",
        f"out: {args.out}",
        f"reference: {args.reference}",
        f"half1: {args.half1}",
        f"half2: {args.half2}",
        f"voxel_size_A: {voxel_size_a}",
        f"patch_size: {patch_size}",
        f"stride: {args.stride}",
        f"fsc_threshold: {args.fsc_threshold}",
        f"window: {args.window}",
        f"contour: {args.contour}",
        f"masked: {not args.no_mask}",
        f"n_voxels_in_mask: {n_mask}",
        f"fraction_nan_inside_mask: {frac_nan:.6f}",
    ]
    if vals.size:
        lines.extend([
            f"min_A_inside_mask: {float(np.min(vals)):.4f}",
            f"median_A_inside_mask: {float(np.median(vals)):.4f}",
            f"max_A_inside_mask: {float(np.max(vals)):.4f}",
        ])
    else:
        lines.append("no finite values inside mask")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"[run_local_fsc] loading halves vs reference {args.reference}")
    bundle = load_full_and_half_maps(
        args.reference,
        args.half1,
        args.half2,
        reference="full",
        dtype=np.float32,
        resample_if_needed=True,
    )
    for name, rep in bundle.reports.items():
        if not rep.ok:
            print(f"[run_local_fsc] WARNING: {name}: {rep.messages}", file=sys.stderr)

    ref_mg = load_map_grid(args.reference, normalize=None)
    vs = ref_mg.voxel_size_zyx
    if not (abs(vs[0] - vs[1]) < 1e-3 and abs(vs[1] - vs[2]) < 1e-3):
        print(f"[run_local_fsc] ERROR: anisotropic voxel {vs}; need isotropic", file=sys.stderr)
        return 2
    voxel_size_a = float(vs[0])

    if args.no_mask:
        print(
            "[run_local_fsc] WARNING: --no-mask evaluates the full box and may OOM on "
            "large maps. Use contour masking (default).",
            file=sys.stderr,
        )
        mask = None
        require_mask = False
    else:
        print(f"[run_local_fsc] contour mask >= {args.contour}")
        mask = build_contour_mask(bundle.full.data, args.contour)
        n_in = int(mask.sum())
        print(f"[run_local_fsc] mask: {n_in:,}/{mask.size:,} voxels ({100.0 * n_in / mask.size:.2f}%)")
        if n_in < 1000:
            print(
                "[run_local_fsc] ERROR: mask too small; check contour / reference scale",
                file=sys.stderr,
            )
            return 2
        require_mask = True
        if args.n_jobs > 4:
            print(
                f"[run_local_fsc] capping n_jobs {args.n_jobs} -> 4 for large masked volumes",
                file=sys.stderr,
            )
            args.n_jobs = 4

    print(
        f"[run_local_fsc] computing local FSC P={args.patch_size} s={args.stride} "
        f"t={args.fsc_threshold} n_jobs={args.n_jobs} (threads, masked={mask is not None})"
    )
    res = compute_local_fsc_resolution(
        bundle.half1.data,
        bundle.half2.data,
        voxel_size_a,
        patch_size=args.patch_size,
        stride=args.stride,
        fsc_threshold=args.fsc_threshold,
        window=args.window,
        mask=mask,
        min_voxels_for_fsc=args.min_voxels_for_fsc,
        n_jobs=args.n_jobs,
        require_mask=require_mask,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_local_fsc_resolution_mrc(
        res,
        args.reference,
        args.out,
        fsc_threshold=args.fsc_threshold,
        patch_size=args.patch_size,
        stride=args.stride,
        mask=mask,
        solvent_value=0.0,
    )
    summary_path = Path(str(args.out) + ".summary.txt")
    _write_summary(
        summary_path,
        res=res,
        mask=mask,
        voxel_size_a=voxel_size_a,
        patch_size=args.patch_size,
        args=args,
    )

    if mask is not None:
        vals = res[mask]
        vals = vals[np.isfinite(vals)]
        frac_nan = float(np.mean(~np.isfinite(res[mask])))
        if vals.size:
            print(
                f"[run_local_fsc] inside mask: min={vals.min():.3f} Å "
                f"median={np.median(vals):.3f} Å max={vals.max():.3f} Å "
                f"frac_nan={frac_nan:.4f}"
            )
    print(f"[run_local_fsc] wrote {args.out} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
