"""Driver: features + half-maps -> correlation CSV, summary, figures.

Usage
-----

End-to-end on EMD-49450 (after the user-side rerun documented in DECISIONS.md):

    python scripts/run_analysis.py \
        --features data/emd_49450-mgtA_tetramer/emd_49450_avg_features_t0116.npz \
        --half1 data/emd_49450-mgtA_tetramer/emd_49450_half_map_1.map \
        --half2 data/emd_49450-mgtA_tetramer/emd_49450_half_map_2.map \
        --reference data/emd_49450-mgtA_tetramer/emd_49450.map \
        --contour 0.116 \
        --window 5 \
        --out-dir outputs/emd_49450/analysis

What it produces under ``--out-dir``:

- ``halfmap_metrics/`` — four MRC volumes (CC, MSE, var-diff, repro-SNR) on the
  reference grid, viewable in ChimeraX.
- ``halfmap_metrics.npz`` — same arrays, single-file load for downstream notebooks.
- ``correlations.csv`` — tidy per-feature Pearson + Spearman against the chosen
  target signal (default ``local_cross_correlation``).
- ``summary.txt`` — top-N features, mask coverage, scientific caveats.
- ``figures/halfmap_metric_histograms.png`` — distributions inside vs outside
  the contour mask.
- ``figures/{feature}_vs_{target}.png`` — hexbin + binned-mean curve for the
  top-K features by |Spearman|.

``--local-res`` supplies the home-rolled FSC map. Use ``--reliability-target`` to
choose whether figures/CSV use half-map CC (default), Å local resolution, or both.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import (
    binned_feature_by_target,
    build_contour_mask,
    compute_feature_target_correlations,
    half_map_local_metrics_chunked,
    plot_feature_vs_target_scatter,
    plot_halfmap_metric_histogram,
    write_correlation_csv,
    write_summary_text,
)
from cryoem_mrc.half_map_repro import save_half_map_metrics_mrc
from cryoem_mrc.local_resolution_io import (
    load_local_resolution_map,
    resample_local_resolution_onto_reference,
)
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.pipeline import load_feature_maps


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path, help="features.npz from run_pipeline")
    p.add_argument("--half1", required=True, type=Path)
    p.add_argument("--half2", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path,
                   help="Reference MRC (used for grid + saving derived MRCs)")
    p.add_argument("--contour", type=float, default=0.116,
                   help="Density contour for the analysis mask (default: 0.116, "
                        "EMDB recommended for EMD-49450)")
    p.add_argument("--window", type=int, default=5, help="Sliding window for half-map metrics")
    p.add_argument("--chunk-z", type=int, default=64, help="Z-chunk size for memory-bounded compute")
    p.add_argument("--target", default="local_cross_correlation",
                   choices=("local_cross_correlation", "local_reproducibility_snr",
                            "local_mean_squared_difference", "local_variance_difference"),
                   help="Which half-map metric to use as the analysis target")
    p.add_argument("--max-samples", type=int, default=2_000_000,
                   help="Subsample size for correlations (None disables; default 2e6)")
    p.add_argument("--top-k-figures", type=int, default=4,
                   help="How many feature-vs-target scatter plots to generate")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--skip-halfmap-metrics", action="store_true",
                   help="If outputs/halfmap_metrics.npz already exists, skip recompute")
    p.add_argument("--local-res", type=Path, default=None,
                   help="Å-valued local_fsc MRC from scripts/run_local_fsc.py")
    p.add_argument(
        "--reliability-target",
        default="halfmap_cc",
        choices=("halfmap_cc", "local_resolution", "both"),
        help="Primary target for correlations.csv and top-K scatter figures. "
             "'local_resolution' and 'both' require --local-res.",
    )
    return p.parse_args(argv)


def _run_correlation_pass(
    features: dict,
    target_vol: np.ndarray,
    mask: np.ndarray,
    *,
    target_name: str,
    out_dir: Path,
    fig_dir: Path,
    contour: float,
    max_samples: int | None,
    top_k_figures: int,
    csv_name: str,
    summary_name: str,
    extra_metadata: dict,
) -> "MaskedAnalysisResult":
    from cryoem_mrc.analysis import MaskedAnalysisResult  # noqa: F401 — for type hint

    print(f"[run_analysis] computing correlations vs {target_name}")
    result = compute_feature_target_correlations(
        features, target_vol, mask,
        target_name=target_name,
        methods=("pearson", "spearman"),
        max_samples=max_samples,
    )
    result.contour = float(contour)
    result.extra_metadata = extra_metadata
    write_correlation_csv(result, out_dir / csv_name)
    write_summary_text(result, out_dir / summary_name)
    print(f"[run_analysis] wrote {out_dir / csv_name} and {out_dir / summary_name}")

    spearman_rows = [c for c in result.correlations if c.method == "spearman"]
    spearman_rows.sort(
        key=lambda c: abs(c.correlation) if np.isfinite(c.correlation) else -1.0,
        reverse=True,
    )
    for c in spearman_rows[:top_k_figures]:
        feat = features.get(c.feature_name)
        if feat is None or feat.shape != target_vol.shape:
            continue
        binned = binned_feature_by_target(
            feat, target_vol, mask,
            feature_name=c.feature_name, target_name=target_name, n_bins=10,
        )
        plot_feature_vs_target_scatter(
            feat, target_vol, mask,
            feature_name=c.feature_name, target_name=target_name,
            save_path=fig_dir / f"{c.feature_name}_vs_{target_name}.png",
            binned=binned,
        )
    return result


def _write_localres_vs_cc(
    metrics: dict[str, np.ndarray],
    local_res: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    from scipy import stats as scipy_stats

    target_lr_name = "local_resolution_A"
    rows_cc: list[tuple[str, str, str, int, float, float]] = []
    mb = mask.astype(bool)
    for metric_name in ("local_cross_correlation", "local_reproducibility_snr"):
        x = metrics[metric_name][mb].ravel()
        y = local_res[mb].ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 3 or x.std() == 0 or y.std() == 0:
            rho, pval = float("nan"), float("nan")
        else:
            r = scipy_stats.spearmanr(x, y)
            rho, pval = float(r.statistic), float(r.pvalue)
        rows_cc.append((metric_name, target_lr_name, "spearman", int(x.size), rho, pval))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["metric", "target", "method", "n_samples", "correlation", "p_value"])
        for row in rows_cc:
            w.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.6f}", f"{row[5]:.6e}"])
    print(f"[run_analysis] wrote {out_path}")
    for metric_name, _, _, n, rho, _ in rows_cc:
        print(f"[run_analysis]   {metric_name} vs local_resolution_A: "
              f"Spearman={rho:.4f} (n={n})")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.reliability_target in ("local_resolution", "both") and args.local_res is None:
        print("[run_analysis] ERROR: --reliability-target requires --local-res", file=sys.stderr)
        return 2
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_npz = out_dir / "halfmap_metrics.npz"
    metrics_mrc_dir = out_dir / "halfmap_metrics"

    # -- load aligned halves on the reference grid -----------------------------
    print(f"[run_analysis] loading halves vs reference {args.reference}")
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
            print(f"[run_analysis] WARNING: {name} required resampling: {rep.messages}",
                  file=sys.stderr)

    # -- half-map metrics ------------------------------------------------------
    if args.skip_halfmap_metrics and metrics_npz.exists():
        print(f"[run_analysis] reusing existing {metrics_npz}")
        data = np.load(metrics_npz, allow_pickle=False)
        metrics = {k: data[k] for k in data.files}
    else:
        print(f"[run_analysis] computing half-map metrics window={args.window} chunk_z={args.chunk_z}")
        metrics = half_map_local_metrics_chunked(
            bundle.half1.data, bundle.half2.data,
            window=args.window, chunk_z=args.chunk_z,
        )
        np.savez_compressed(metrics_npz, **metrics)
        save_half_map_metrics_mrc(metrics, args.reference, metrics_mrc_dir)
        print(f"[run_analysis] wrote {metrics_npz} and MRCs under {metrics_mrc_dir}")

    # -- mask ------------------------------------------------------------------
    print(f"[run_analysis] contour mask >= {args.contour}")
    mask = build_contour_mask(bundle.full.data, args.contour)
    n_total, n_in = int(mask.size), int(mask.sum())
    pct = 100.0 * n_in / max(1, n_total)
    print(f"[run_analysis] mask: {n_in:,}/{n_total:,} voxels ({pct:.2f}%)")
    if n_in < 1000:
        print(f"[run_analysis] ERROR: mask has only {n_in} voxels at contour={args.contour}; "
              "is this the right intensity scale for the reference map?", file=sys.stderr)
        return 2

    # -- features --------------------------------------------------------------
    print(f"[run_analysis] loading features from {args.features}")
    features = load_feature_maps(args.features)

    meta = {
        "features_npz": str(args.features),
        "half1_path": str(args.half1),
        "half2_path": str(args.half2),
        "reference_mrc": str(args.reference),
        "window": str(args.window),
        "reliability_target": args.reliability_target,
        "local_res_path": str(args.local_res) if args.local_res else None,
    }

    local_res: np.ndarray | None = None
    if args.local_res is not None:
        print(f"[run_analysis] loading local resolution {args.local_res}")
        ref_mg = load_map_grid(args.reference, normalize=None)
        local_mg = load_local_resolution_map(args.local_res)
        aligned = resample_local_resolution_onto_reference(local_mg, ref_mg)
        local_res = np.asarray(aligned.data, dtype=np.float32)
        if local_res.shape != bundle.full.data.shape:
            print("[run_analysis] ERROR: local resolution shape mismatch after resample",
                  file=sys.stderr)
            return 2

    print("[run_analysis] writing figures")
    plot_halfmap_metric_histogram(
        metrics, mask, save_path=fig_dir / "halfmap_metric_histograms.png",
        title=f"half-map metrics inside vs outside contour {args.contour}",
    )

    if args.reliability_target in ("halfmap_cc", "both"):
        target = metrics[args.target]
        _run_correlation_pass(
            features, target, mask,
            target_name=args.target,
            out_dir=out_dir,
            fig_dir=fig_dir,
            contour=args.contour,
            max_samples=args.max_samples,
            top_k_figures=args.top_k_figures,
            csv_name="correlations.csv" if args.reliability_target == "halfmap_cc" else "correlations_halfmap_cc.csv",
            summary_name="summary.txt" if args.reliability_target == "halfmap_cc" else "summary_halfmap_cc.txt",
            extra_metadata=meta,
        )

    if args.reliability_target in ("local_resolution", "both"):
        assert local_res is not None
        _run_correlation_pass(
            features, local_res, mask,
            target_name="local_resolution_A",
            out_dir=out_dir,
            fig_dir=fig_dir,
            contour=args.contour,
            max_samples=args.max_samples,
            top_k_figures=args.top_k_figures,
            csv_name="correlations.csv" if args.reliability_target == "local_resolution" else "correlations_localres.csv",
            summary_name="summary.txt" if args.reliability_target == "local_resolution" else "summary_localres.txt",
            extra_metadata=meta,
        )

    if local_res is not None:
        _write_localres_vs_cc(metrics, local_res, mask, out_dir / "localres_vs_cc.csv")

    print(f"[run_analysis] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
