"""Generate thesis overview / schematic figure panels (EMD-49450 defaults).

The analysis contour (default 0.116) is applied using the **deposited reference
map** (``emd_* .map``), matching ``run_analysis.py`` and Decision 002 — not the
averaged-half map, whose intensity scale differs.

After fixing a bad run, delete old PNGs under ``outputs/emd_49450/thesis_overview/`` and
re-run without ``--skip-existing``.

Example::

    source .venv/bin/activate
    rm -f outputs/emd_49450/thesis_overview/*.png
    python scripts/run_thesis_overview_figures.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import apply, label_panel, savefig as save_nature

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.io import load_mrc
from cryoem_mrc.local_resolution_io import load_local_resolution_map
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.repo_paths import (
    DATA_ROOT,
    analysis_dir,
    find_features_npz,
    locres_blocres_mrc,
    thesis_overview_dir,
)
from cryoem_mrc.rigidity import compute_rigidity_map_from_npz
from cryoem_mrc.thesis_figures import (
    CC_CBAR_LABEL,
    LOCRES_CBAR_LABEL,
    RELIABILITY_CMAP_CC,
    RELIABILITY_CMAP_LOCRES,
    SliceCrop,
    _locres_robust_limits,
    apply_contour_mask,
    crop_slice_2d,
    extract_slice,
    mask_slice_values,
    pick_slice_index,
    plot_feature_family_panel,
    plot_local_resolution_slice,
    plot_masked_slice,
    plot_parallel_reliability_readouts,
    plot_reliability_pair_only,
    plot_spearman_top_bar,
    slice_crop_from_mask,
    _add_scale_bar,
)

# All deliverables in generation order.
FIGURE_JOBS: tuple[str, ...] = (
    "local_resolution_slice",
    "parallel_readouts_density_cc_localres",
    "parallel_readouts_cc_localres",
    "feature_family_local_stats",
    "feature_family_multiscale",
    "feature_family_rigidity",
    "density_reference_slice",
    "mask_slice",
    "spearman_top_features",
    "overview_composite_row",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT / "emd_49450-mgtA_e2p+e1")
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Deposited primary map for contour mask (default: emd_<ID>.map in data-dir)",
    )
    p.add_argument("--avg-map", type=Path, default=None, help="Averaged map for density panels")
    p.add_argument(
        "--local-res",
        "--local-fsc",
        type=Path,
        default=None,
        dest="local_res",
        help="Å local-resolution MRC (BlocRes locres_blocres.mrc preferred when present)",
    )
    p.add_argument("--features", type=Path, default=None)
    p.add_argument(
        "--halfmap-npz",
        type=Path,
        default=analysis_dir("49450") / "halfmap_metrics.npz",
    )
    p.add_argument(
        "--correlations",
        type=Path,
        default=analysis_dir("49450") / "correlations.csv",
    )
    p.add_argument("--contour", type=float, default=0.116)
    p.add_argument("--slice-z", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=thesis_overview_dir("49450"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--zoom-padding",
        type=int,
        default=24,
        help="Crop panels to mask bbox + this many voxels (0 = full 430² slice)",
    )
    p.add_argument(
        "--scale-bar-a",
        type=float,
        default=None,
        help="Scale bar length in Å (default 20 when zoomed, 50 when not)",
    )
    p.add_argument("--emd-id", type=str, default="49450")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PNGs that already exist in --out-dir (resume after crash)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=FIGURE_JOBS,
        metavar="JOB",
        help="Generate only these panels (e.g. --only feature_family_rigidity overview_composite_row)",
    )
    p.add_argument(
        "--rigidity-cache",
        type=Path,
        default=None,
        help="Optional .npy cache for full rigidity volume (avoids recomputation)",
    )
    return p.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    d = args.data_dir
    emd = f"emd_{args.emd_id}"
    out = args.out_dir
    return {
        "reference": args.reference or d / f"{emd}.map",
        "avg": args.avg_map or d / f"{emd}_avg.map",
        "local_res": (
            args.local_res
            or (locres_blocres_mrc(args.emd_id) if locres_blocres_mrc(args.emd_id).is_file() else None)
            or d / f"{emd}_local_fsc_t0143_P17_s4.mrc"
        ),
        "features": (
            args.features
            or find_features_npz(d, args.emd_id, args.contour)
            or d / f"{emd}_avg_features_t0116.npz"
        ),
        "halfmap_npz": args.halfmap_npz,
        "correlations": args.correlations,
        "rigidity_cache": args.rigidity_cache or out / f"{emd}_rigidity_cache.npy",
        "state": out / ".thesis_overview_state.npz",
    }


def _fig_export(out_dir: Path, stem: str) -> Path:
    """Canonical export path (save_nature writes ``.pdf`` + 600 dpi ``.png``)."""
    return out_dir / f"{stem}.pdf"


def _fig_export_done(path: Path) -> bool:
    return path.with_suffix(".pdf").exists() and path.with_suffix(".png").exists()


def _should_run(job: str, path: Path, *, skip_existing: bool, only: set[str] | None) -> bool:
    if only is not None and job not in only:
        return False
    if skip_existing and _fig_export_done(path):
        print(f"[thesis_figures] skip existing {path.name}")
        return False
    return True


def _load_npz_keys(npz_path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Load only the requested arrays from a feature NPZ (one decompress at a time)."""
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in keys:
            if key not in data.files:
                raise KeyError(f"{npz_path.name} missing {key!r}")
            out[key] = np.asarray(data[key], dtype=np.float32)
    return out


def _contour_feature_dict(
    feats: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply analysis contour to each feature volume before plotting."""
    return {k: apply_contour_mask(v, mask) for k, v in feats.items()}


def _load_or_compute_rigidity(
    features_npz: Path,
    mask: np.ndarray,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        print(f"[thesis_figures] loading rigidity cache {cache_path.name}")
        rig = np.load(cache_path, mmap_mode="r")
        return np.asarray(rig, dtype=np.float32)
    print("[thesis_figures] computing rigidity (one NPZ array at a time; may take ~1–2 min)")
    rig = compute_rigidity_map_from_npz(features_npz, mask=mask)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, rig)
    print(f"[thesis_figures] cached rigidity -> {cache_path}")
    return rig


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_nature(fig, path, dpi=dpi)
    plt.close(fig)
    gc.collect()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _resolve_paths(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None

    for label, p in paths.items():
        if label in ("correlations", "rigidity_cache", "state"):
            continue
        if label == "correlations" and not p.exists():
            print(f"[thesis_figures] WARNING: missing optional {label}: {p}", file=sys.stderr)
            continue
        if not p.exists():
            print(f"[thesis_figures] ERROR: missing {label}: {p}", file=sys.stderr)
            return 2

    z = args.slice_z

    print(f"[thesis_figures] loading reference for contour {paths['reference'].name}")
    reference = load_mrc(paths["reference"], dtype=np.float32)
    ref_mg = load_map_grid(paths["reference"], normalize=None)
    voxel_size_a = float(ref_mg.voxel_size_zyx[0])

    # Decision 002: mask from deposited reference at depositor contour (same as run_analysis).
    mask = build_contour_mask(reference, args.contour)
    n_in = int(mask.sum())
    n_total = int(mask.size)
    print(
        f"[thesis_figures] contour mask: {n_in:,}/{n_total:,} voxels "
        f"({100.0 * n_in / n_total:.2f}%) at ρ ≥ {args.contour} on reference"
    )
    if args.emd_id == "49450" and not (150_000 <= n_in <= 350_000):
        print(
            "[thesis_figures] WARNING: EMD-49450 usually has ~235k masked voxels; "
            "check --reference and --contour",
            file=sys.stderr,
        )

    if z is None:
        z = pick_slice_index(mask, axis=0)
    print(f"[thesis_figures] slice Z={z} (mask voxels on slice: {int(mask[z].sum()):,})")

    np.savez(
        paths["state"],
        slice_z=np.int32(z),
        contour=np.float32(args.contour),
        emd_id=np.array(args.emd_id),
        mask_source=np.array(str(paths["reference"])),
    )

    print(f"[thesis_figures] loading averaged map {paths['avg'].name}")
    avg = load_mrc(paths["avg"], dtype=np.float32)
    if avg.shape != mask.shape:
        print(
            f"[thesis_figures] ERROR: avg shape {avg.shape} != reference {mask.shape}",
            file=sys.stderr,
        )
        return 2

    msl = mask[z]
    crop_bbox: SliceCrop | None = None
    if args.zoom_padding > 0:
        crop_bbox = slice_crop_from_mask(msl, pad_voxels=args.zoom_padding)
        print(
            f"[thesis_figures] zoom crop on slice: "
            f"Y=[{crop_bbox[0]}:{crop_bbox[1]}] X=[{crop_bbox[2]}:{crop_bbox[3]}] "
            f"({crop_bbox[1]-crop_bbox[0]}×{crop_bbox[3]-crop_bbox[2]} px)"
        )
    scale_bar_a = args.scale_bar_a
    if scale_bar_a is None:
        scale_bar_a = 20.0 if crop_bbox is not None else 50.0

    # Contour before display: only in-mask voxels contribute to panels.
    avg_contoured = apply_contour_mask(avg, mask)
    d_sl = extract_slice(avg_contoured, axis=0, index=z)
    dpi = args.dpi
    saved: list[Path] = []

    def run_job(job: str, fn: Callable[[], None]) -> None:
        dest = _fig_export(out_dir, job)
        if not _should_run(job, dest, skip_existing=args.skip_existing, only=only):
            if _fig_export_done(dest):
                saved.append(dest)
            return
        print(f"[thesis_figures] generating {job}")
        fn()
        saved.append(dest)
        print(f"[thesis_figures] wrote {dest.name}")

    # --- local FSC + half-map CC (load once, used by several jobs) ---
    need_cc_res = any(
        _should_run(j, _fig_export(out_dir, j), skip_existing=args.skip_existing, only=only)
        for j in (
            "local_resolution_slice",
            "parallel_readouts_density_cc_localres",
            "parallel_readouts_cc_localres",
            "overview_composite_row",
        )
    )
    local_res: np.ndarray | None = None
    local_cc: np.ndarray | None = None

    if need_cc_res:
        print(f"[thesis_figures] loading local resolution {paths['local_res'].name}")
        local_res = np.asarray(load_local_resolution_map(paths["local_res"]).data, dtype=np.float32)
        print("[thesis_figures] loading half-map CC from NPZ")
        with np.load(paths["halfmap_npz"], allow_pickle=False) as hm:
            local_cc = np.asarray(hm["local_cross_correlation"], dtype=np.float32)
        if local_cc.shape != mask.shape:
            print(
                f"[thesis_figures] ERROR: halfmap shape {local_cc.shape} != mask {mask.shape}",
                file=sys.stderr,
            )
            return 2
        print("[thesis_figures] applying contour mask to local resolution and half-map CC volumes")
        local_res = apply_contour_mask(local_res, mask)
        local_cc = apply_contour_mask(local_cc, mask)

    assert local_res is not None and local_cc is not None or not need_cc_res

    if need_cc_res:

        def _local_res_slice() -> None:
            assert local_res is not None
            fig = plot_local_resolution_slice(
                local_res,
                mask,
                slice_index=z,
                voxel_size_a=voxel_size_a,
                dpi=dpi,
                crop_bbox=crop_bbox,
                scale_bar_a=scale_bar_a,
            )
            _save_fig(fig, _fig_export(out_dir, "local_resolution_slice"), dpi)

        run_job("local_resolution_slice", _local_res_slice)

        def _parallel_3() -> None:
            assert local_res is not None and local_cc is not None
            fig = plot_parallel_reliability_readouts(
                avg_contoured,
                local_cc,
                local_res,
                mask,
                slice_index=z,
                voxel_size_a=voxel_size_a,
                contour=args.contour,
                dpi=dpi,
                crop_bbox=crop_bbox,
                scale_bar_a=scale_bar_a,
            )
            _save_fig(fig, _fig_export(out_dir, "parallel_readouts_density_cc_localres"), dpi)

        run_job("parallel_readouts_density_cc_localres", _parallel_3)

        def _parallel_2() -> None:
            assert local_res is not None and local_cc is not None
            fig = plot_reliability_pair_only(
                local_cc,
                local_res,
                mask,
                slice_index=z,
                dpi=dpi,
                crop_bbox=crop_bbox,
            )
            _save_fig(fig, _fig_export(out_dir, "parallel_readouts_cc_localres"), dpi)

        run_job("parallel_readouts_cc_localres", _parallel_2)

    # Free full 3D CC / res if feature jobs do not need them (composite will reload slices)
    if only is not None and "overview_composite_row" not in only:
        local_cc = None
        local_res = None
        gc.collect()

    # --- feature family panels (load only keys needed per panel) ---
    def _family_local_stats() -> None:
        feats = _contour_feature_dict(
            _load_npz_keys(
                paths["features"],
                ["density_normalized", "local_mean", "local_variance", "gradient_magnitude"],
            ),
            mask,
        )
        fig = plot_feature_family_panel(
            feats,
            list(feats.keys()),
            mask,
            family_title="Local density statistics (from ρ = ½(h₁+h₂))",
            slice_index=z,
            cmap="magma",
            cmap_overrides={"density_normalized": "gray"},
            subtitles={
                "density_normalized": "Normalized ρ",
                "local_mean": "Local mean",
                "local_variance": "Local variance",
                "gradient_magnitude": "|∇ρ|",
            },
            dpi=dpi,
            ncol=2,
            crop_bbox=crop_bbox,
        )
        _save_fig(fig, _fig_export(out_dir, "feature_family_local_stats"), dpi)
        del feats

    run_job("feature_family_local_stats", _family_local_stats)

    def _family_multiscale() -> None:
        keys = ["gauss_s0", "gauss_s2", "gauss_s4", "gauss_s1_local_variance"]
        feats = _contour_feature_dict(_load_npz_keys(paths["features"], keys), mask)
        fig = plot_feature_family_panel(
            feats,
            keys,
            mask,
            family_title="Multi-scale Gaussians (σ ladder)",
            slice_index=z,
            cmap="inferno",
            subtitles={
                "gauss_s0": "Gσ * ρ (fine)",
                "gauss_s2": "Gσ * ρ (mid)",
                "gauss_s4": "Gσ * ρ (coarse)",
                "gauss_s1_local_variance": "Local var (σ≈1)",
            },
            dpi=dpi,
            ncol=2,
            crop_bbox=crop_bbox,
        )
        _save_fig(fig, _fig_export(out_dir, "feature_family_multiscale"), dpi)
        del feats

    run_job("feature_family_multiscale", _family_multiscale)

    rigidity_vol: np.ndarray | None = None

    def _family_rigidity() -> None:
        nonlocal rigidity_vol
        rigidity_vol = apply_contour_mask(
            _load_or_compute_rigidity(paths["features"], mask, paths["rigidity_cache"]),
            mask,
        )
        feats = _contour_feature_dict(
            _load_npz_keys(
                paths["features"],
                ["gradient_magnitude", "local_variance"],
            ),
            mask,
        )
        feats["rigidity"] = rigidity_vol
        fig = plot_feature_family_panel(
            feats,
            ["gradient_magnitude", "local_variance", "rigidity"],
            mask,
            family_title="Rigidity heuristic (combined score)",
            slice_index=z,
            cmap="viridis",
            cmap_overrides={"gradient_magnitude": "magma", "local_variance": "magma"},
            subtitles={
                "gradient_magnitude": "Low |∇ρ| → rigid-like",
                "local_variance": "Low variance → rigid-like",
                "rigidity": "Combined rigidity",
            },
            dpi=dpi,
            ncol=3,
            crop_bbox=crop_bbox,
        )
        _save_fig(fig, _fig_export(out_dir, "feature_family_rigidity"), dpi)
        del feats

    run_job("feature_family_rigidity", _family_rigidity)

    def _density_slice() -> None:
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        apply(ax)
        plot_masked_slice(
            ax,
            d_sl,
            msl,
            cmap="gray",
            cbar_label="density",
            title=f"Averaged map Z={z}",
            already_contoured=True,
            crop_bbox=crop_bbox,
        )
        _add_scale_bar(ax, voxel_size_a=voxel_size_a, length_a=scale_bar_a)
        _save_fig(fig, _fig_export(out_dir, "density_reference_slice"), dpi)

    run_job("density_reference_slice", _density_slice)

    def _mask_slice() -> None:
        msl_show = crop_slice_2d(msl.astype(float), crop_bbox) if crop_bbox else msl.astype(float)
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        apply(ax)
        im = ax.imshow(msl_show, cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"Contour mask (ρ ≥ {args.contour}) Z={z}")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 1])
        cb.ax.set_yticklabels(["outside", "inside"])
        cb.set_label("mask", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        _save_fig(fig, _fig_export(out_dir, "mask_slice"), dpi)

    run_job("mask_slice", _mask_slice)

    def _spearman() -> None:
        if not paths["correlations"].exists():
            print("[thesis_figures] skip spearman (no correlations.csv)", file=sys.stderr)
            return
        fig = plot_spearman_top_bar(paths["correlations"], dpi=dpi)
        _save_fig(fig, _fig_export(out_dir, "spearman_top_features"), dpi)

    run_job("spearman_top_features", _spearman)

    def _composite() -> None:
        nonlocal rigidity_vol, local_cc, local_res
        if local_cc is None:
            with np.load(paths["halfmap_npz"], allow_pickle=False) as hm:
                local_cc = apply_contour_mask(
                    np.asarray(hm["local_cross_correlation"], dtype=np.float32),
                    mask,
                )
        if local_res is None:
            local_res = apply_contour_mask(
                np.asarray(load_local_resolution_map(paths["local_res"]).data, dtype=np.float32),
                mask,
            )
        if rigidity_vol is None and paths["rigidity_cache"].exists():
            rigidity_vol = apply_contour_mask(
                np.load(paths["rigidity_cache"], mmap_mode="r"), mask
            )
        elif rigidity_vol is None:
            rigidity_vol = apply_contour_mask(
                _load_or_compute_rigidity(paths["features"], mask, paths["rigidity_cache"]),
                mask,
            )

        lv = _contour_feature_dict(
            _load_npz_keys(paths["features"], ["local_variance", "gauss_s2"]),
            mask,
        )

        res_sl = extract_slice(local_res, axis=0, index=z)
        res_lo, res_hi = _locres_robust_limits(res_sl, msl)
        fig, axes = plt.subplots(1, 6, figsize=(22.0, 4.5))
        panels: list[tuple[np.ndarray, str, str, str, dict]] = [
            (d_sl, "gray", "ρ avg", "density", {}),
            (
                extract_slice(local_cc, axis=0, index=z),
                RELIABILITY_CMAP_CC,
                "half-map CC",
                CC_CBAR_LABEL,
                {"vmin": 0.0, "vmax": 1.0, "robust": False},
            ),
            (
                res_sl,
                RELIABILITY_CMAP_LOCRES,
                "local res Å",
                LOCRES_CBAR_LABEL,
                {"vmin": res_lo, "vmax": res_hi, "robust": False},
            ),
            (
                extract_slice(lv["local_variance"], axis=0, index=z),
                "magma",
                "local var",
                "variance",
                {},
            ),
            (
                extract_slice(lv["gauss_s2"], axis=0, index=z),
                "inferno",
                "Gσ mid",
                "Gσ * ρ",
                {},
            ),
            (
                extract_slice(rigidity_vol, axis=0, index=z),
                "viridis",
                "rigidity",
                "rigidity",
                {},
            ),
        ]
        for letter, (ax, (sl, cm, title, cbar_label, extra_kw)) in zip("abcdef", zip(axes, panels)):
            kw: dict = {
                "cmap": cm,
                "crop_bbox": crop_bbox,
                "already_contoured": True,
                "cbar_label": cbar_label,
                **extra_kw,
            }
            plot_masked_slice(ax, sl, msl, title=title, **kw)
            label_panel(ax, letter)
        fig.suptitle(
            f"EMD-{args.emd_id} overview row (Z={z}, mask ρ≥{args.contour})",
            fontsize=12,
        )
        fig.tight_layout()
        _save_fig(fig, _fig_export(out_dir, "overview_composite_row"), dpi)

    run_job("overview_composite_row", _composite)

    # Manifest lists expected PDFs (all jobs, or only those requested via --only).
    jobs_expected = tuple(args.only) if args.only else FIGURE_JOBS
    expected = [_fig_export(out_dir, job) for job in jobs_expected]
    manifest = out_dir / "MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                f"EMD-{args.emd_id} thesis overview figures",
                f"slice_Z: {z}",
                f"contour: {args.contour}",
                f"voxel_size_A: {voxel_size_a}",
                f"skip_existing: {args.skip_existing}",
                "",
                "Files:",
                *[
                    f"  {p.name}  ({'ok' if _fig_export_done(p) else 'MISSING'})"
                    for p in expected
                ],
            ]
        )
        + "\n"
    )
    n_ok = sum(1 for p in expected if _fig_export_done(p))
    print(f"[thesis_figures] done: {n_ok}/{len(expected)} PDF+PNG pairs present under {out_dir}")
    missing = [p.name for p in expected if not _fig_export_done(p)]
    if missing:
        print(f"[thesis_figures] still missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
