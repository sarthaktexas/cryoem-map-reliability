"""CLI: python -m cryoem_mrc <map.mrc> [--out features.npz] [--plot]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .analysis import build_contour_mask
from .io import load_mrc, save_rigidity_mrc
from .reliability import save_build_zone_mrc, save_reliability_mrc
from .pipeline import run_pipeline, save_feature_maps, save_feature_maps_npy


def main() -> int:
    p = argparse.ArgumentParser(description="Cryo-EM MRC feature pipeline")
    p.add_argument("mrc", type=Path, help="Path to .mrc / .map")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path (default: <mrc_stem>_features.npz)",
    )
    p.add_argument(
        "--out-npy-dir",
        type=Path,
        default=None,
        help="If set, also save each feature as a separate .npy file here",
    )
    p.add_argument("--local-window", type=int, default=5, help="Uniform window size (odd)")
    p.add_argument(
        "--norm",
        choices=("zscore", "minmax", "percentile"),
        default="zscore",
    )
    p.add_argument(
        "--start-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Raw map intensity: voxels below T are set to 0 before normalization "
        "(omit low-density / solvent you do not want to drive features)",
    )
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=None,
        help="3–5 Gaussian sigmas in voxels (default: 0.5 1 2 4 8)",
    )
    p.add_argument("--plot", action="store_true", help="Show matplotlib slice gallery")
    p.add_argument(
        "--plot-save",
        type=Path,
        default=None,
        help="Save slice figure to this path (non-interactive)",
    )
    p.add_argument(
        "--no-rigidity",
        action="store_true",
        help="Compute legacy equal-weight rigidity heuristic (off by default)",
    )
    p.add_argument(
        "--rigidity",
        action="store_true",
        help="Include legacy rigidity map in the feature NPZ",
    )
    p.add_argument(
        "--rigidity-w",
        type=float,
        nargs=3,
        metavar=("WG", "WV", "WC"),
        default=None,
        help="Weights (gradient, local variance, cross-scale consistency); default equal",
    )
    p.add_argument(
        "--float32",
        action="store_true",
        help="Load and compute in float32 (recommended for ~400³+ maps: less RAM, faster)",
    )
    p.add_argument(
        "--npz-uncompressed",
        action="store_true",
        help="Write .npz without zlib (much faster save on huge feature stacks; larger file)",
    )
    p.add_argument(
        "--half1",
        type=Path,
        default=None,
        help="Half-map 1 (.map) — with --half2 enables reliability_score export",
    )
    p.add_argument(
        "--half2",
        type=Path,
        default=None,
        help="Half-map 2 (.map) — with --half1 enables reliability_score export",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Deposited map for contour mask when computing build_zone (Decision 002)",
    )
    p.add_argument(
        "--contour",
        type=float,
        default=0.116,
        help="Contour on reference for build_zone labels (default 0.116)",
    )
    p.add_argument(
        "--reliability-mrc-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write reliability_score MRC (ChimeraX); requires --half1 and --half2",
    )
    p.add_argument(
        "--rigidity-mrc-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write legacy rigidity MRC; requires --rigidity",
    )
    p.add_argument(
        "--build-zone-mrc-out",
        dest="build_zone_mrc_out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write build_zone MRC (omit/caution/build); requires --half1, --half2, and --reference",
    )
    args = p.parse_args()

    if args.sigmas is not None and not (3 <= len(args.sigmas) <= 5):
        print("Error: --sigmas must have 3 to 5 values.", file=sys.stderr)
        return 2
    if (args.half1 is None) ^ (args.half2 is None):
        print("Error: provide both --half1 and --half2, or neither.", file=sys.stderr)
        return 2

    out = args.out
    if out is None:
        out = args.mrc.with_name(f"{args.mrc.stem}_features.npz")

    rel_mask = None
    if args.half1 and args.half2 and (args.reference or args.build_zone_mrc_out):
        ref_path = args.reference or args.mrc
        rel_mask = build_contour_mask(load_mrc(ref_path, dtype=np.float32), args.contour)

    feats = run_pipeline(
        args.mrc,
        normalization=args.norm,
        start_threshold=args.start_threshold,
        local_window=args.local_window,
        gaussian_sigmas=args.sigmas,
        use_float32=args.float32,
        compute_rigidity=args.rigidity and not args.no_rigidity,
        rigidity_weights=args.rigidity_w,
        half1_path=args.half1,
        half2_path=args.half2,
        reliability_mask=rel_mask,
        compute_reliability=bool(args.half1 and args.half2),
        plot=args.plot or (args.plot_save is not None),
        plot_save=args.plot_save,
    )
    save_feature_maps(feats, out, compressed=not args.npz_uncompressed)
    if args.out_npy_dir is not None:
        save_feature_maps_npy(feats, args.out_npy_dir)
    if args.rigidity_mrc_out is not None:
        if "rigidity" not in feats:
            print("Error: --rigidity-mrc-out requires --rigidity.", file=sys.stderr)
            return 2
        mrc_out = save_rigidity_mrc(args.mrc, feats["rigidity"], args.rigidity_mrc_out)
        print(f"Wrote rigidity MRC for ChimeraX: {mrc_out}")
    ref_for_mrc = args.reference or args.mrc
    if args.reliability_mrc_out is not None:
        if "reliability_score" not in feats:
            print("Error: --reliability-mrc-out requires --half1 and --half2.", file=sys.stderr)
            return 2
        mrc_out = save_reliability_mrc(ref_for_mrc, feats["reliability_score"], args.reliability_mrc_out)
        print(f"Wrote reliability MRC for ChimeraX: {mrc_out}")
    if args.build_zone_mrc_out is not None:
        if "build_zone" not in feats:
            print(
                "Error: --build-zone-mrc-out requires halves and --reference for contour mask.",
                file=sys.stderr,
            )
            return 2
        mrc_out = save_build_zone_mrc(ref_for_mrc, feats["build_zone"], args.build_zone_mrc_out)
        print(f"Wrote build-zone MRC for ChimeraX: {mrc_out}")
    print(f"Wrote {len(feats)} feature maps to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
