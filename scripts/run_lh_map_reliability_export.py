"""Thesis bundle: reliability maps, build zones, figures, and write-up (EMD-49450 defaults).

Generates under ``outputs/emd_<ID>/lh_map_reliability/``:

- ``reliability.npz`` — reliability_score, H_repro, LH maps (T, V, L), build_zone
- ``*.mrc`` — volume overlays on deposited reference grid
- ``figures/model_building_row.png`` — CC, reliability score, build zones (one slice)
- ``../analysis/figures/analysis_validation_panel.png`` — anchor map only (2×2 validation)
- ``LH_MAP_RELIABILITY_RESULTS.md`` — per-map results (methods in docs/LH_MAP_RELIABILITY.md)

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_lh_map_reliability_export.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import apply, label_panel, savefig as save_nature
from scipy import stats

from cryoem_mrc.analysis import (
    build_contour_mask,
    compute_feature_target_correlations,
    plot_analysis_validation_panel,
)
from cryoem_mrc.figure_cleanup import prune_lh_retired_figures
from cryoem_mrc.pipeline import load_feature_maps
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.reliability import (
    BUILD_ZONE_LABELS,
    attach_reliability_to_features,
    build_zone_colormap,
    save_build_zone_mrc,
    save_reliability_mrc,
)
from cryoem_mrc.repo_paths import (
    ANCHOR_EMDB_ID,
    DATA_ROOT,
    analysis_dir,
    halfmap_metrics_npz,
    lh_map_reliability_dir,
)
from cryoem_mrc.mask_bbox import (
    bbox_from_mask,
    crop_array,
    embed_array,
    format_bbox_log,
    pad_voxels_for_filters,
)
from cryoem_mrc.thesis_figures import (
    extract_slice,
    mask_slice_values,
    pick_slice_index,
    plot_masked_slice,
    slice_crop_from_mask,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT / "emd_49450-mgtA_e2p+e1")
    p.add_argument("--emd-id", type=str, default="49450")
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--avg-map", type=Path, default=None)
    p.add_argument("--half1", type=Path, default=None)
    p.add_argument("--half2", type=Path, default=None)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--halfmap-npz", type=Path, default=halfmap_metrics_npz("49450"))
    p.add_argument("--contour", type=float, default=0.116)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=lh_map_reliability_dir("49450"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--zoom-padding", type=int, default=24)
    p.add_argument(
        "--no-crop-to-contour",
        action="store_true",
        help="Compute H_repro on the full grid (default: tight bbox around contour mask)",
    )
    p.add_argument(
        "--write-analysis-panel",
        action="store_true",
        help=f"Write 2×2 validation panel under analysis/figures/ (default for EMD-{ANCHOR_EMDB_ID})",
    )
    p.add_argument(
        "--no-write-analysis-panel",
        action="store_true",
        help="Skip anchor validation panel even for the canonical anchor map",
    )
    p.add_argument(
        "--prune-retired-figures",
        action="store_true",
        help="Delete orphaned spearman/binned/bfactor LH figure exports in figures/",
    )
    return p.parse_args(argv)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    d = args.data_dir
    emd = f"emd_{args.emd_id}"
    return {
        "reference": args.reference or d / f"{emd}.map",
        "avg": args.avg_map or d / f"{emd}_avg.map",
        "half1": args.half1 or d / f"{emd}_half_map_1.map",
        "half2": args.half2 or d / f"{emd}_half_map_2.map",
        "features": args.features or d / f"{emd}_avg_features_t0116.npz",
    }


def _load_local_var(features_path: Path) -> np.ndarray:
    with np.load(features_path, allow_pickle=False) as d:
        return np.asarray(d["local_variance"], dtype=np.float32)


def _zscore_halfmap_average(half1: np.ndarray, half2: np.ndarray) -> np.ndarray:
    """Global z-score of ρ = ½(h₁+h₂) for LH gradient / constraint V (Decision 001)."""
    rho = 0.5 * (np.asarray(half1, dtype=np.float32) + np.asarray(half2, dtype=np.float32))
    mu = float(rho.mean())
    sig = float(rho.std())
    return ((rho - mu) / (sig + 1e-6)).astype(np.float32)


def _plot_build_zones(ax, zones_sl: np.ndarray, mask_sl: np.ndarray, *, title: str) -> None:
    apply(ax)
    show = np.ma.masked_where(~mask_sl, zones_sl.astype(float))
    im = ax.imshow(show, cmap=build_zone_colormap(), vmin=0, vmax=2, origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([BUILD_ZONE_LABELS[z] for z in (0, 1, 2)])
    return im


def _write_thesis_md(
    path: Path,
    *,
    emd_id: str,
    contour: float,
    n_mask: int,
    spearman: dict[str, float],
    partial: dict[str, float],
    zone_counts: dict[int, int],
    paths: dict[str, Path],
) -> None:
    rel_s = spearman.get("reliability_score", float("nan"))
    var_s = spearman.get("local_variance", float("nan"))
    h_s = spearman.get("reliability_H_repro", float("nan"))
    text = f"""# Map reliability — EMD-{emd_id}

Per-map results bundle. **Methods:** [docs/LH_MAP_RELIABILITY.md](docs/LH_MAP_RELIABILITY.md).

Mask: deposited reference at ρ ≥ {contour}. In-mask voxels: **{n_mask:,}**.

---

## Results (ρ_ref ≥ {contour})

| Feature | Spearman ρ vs half-map CC |
|---------|---------------------------|
| local_variance | {var_s:+.4f} |
| reliability_H_repro | {h_s:+.4f} |
| **reliability_score** | **{rel_s:+.4f}** |

Partial Spearman vs CC controlling for local_variance:

| Feature | Partial ρ |
|---------|-----------|
| reliability_score | {partial.get('reliability_score', float('nan')):+.4f} |
| reliability_H_repro | {partial.get('reliability_H_repro', float('nan')):+.4f} |

**Zone counts (in-mask voxels):**

| Zone | Count |
|------|------:|
| 0 omit | {zone_counts.get(0, 0):,} |
| 1 caution | {zone_counts.get(1, 0):,} |
| 2 build | {zone_counts.get(2, 0):,} |

**Interpretation:** `local_variance` remains the strongest single statistic (ρ ≈ {var_s:.2f}). The reproducibility Hamiltonian (ρ ≈ {h_s:.2f}) packages half-map disagreement and gradient smoothness; the rank-normalized **reliability_score** is the default export for model-building guidance.

---

## Draft paragraphs (methods)

> We computed voxel-wise **reliability scores** on the averaged half-map density ρ = ½(h₁+h₂). Half-map difference δρ = h₁−h₂ enters a **reproducibility Hamiltonian** H_repro combining windowed disagreement ½⟨δρ²⟩ and gradient smoothness ½⟨‖∇ρ‖²⟩ with equal coefficients. The exported **reliability_score** is the in-mask percentile rank of H_repro (higher = more reliable). Macromolecular voxels were selected with the EMDB-recommended contour ρ_ref ≥ {contour} on the deposited primary map (Decision 002). **Build zones** (omit / caution / build) were assigned by terciles of reliability_score inside this mask.

## Draft paragraphs (results)

> On EMD-{emd_id} ({n_mask:,} in-mask voxels), reliability_score correlated with windowed half-map cross-correlation at Spearman ρ = {rel_s:.2f}, comparable to the reproducibility energy H_repro (ρ = {h_s:.2f}) and below local variance (ρ = {var_s:.2f}). Zones labeled **build** ({zone_counts.get(2, 0):,} voxels) mark regions where independent half-maps agree and local statistics support confident model placement; **omit** zones ({zone_counts.get(0, 0):,} voxels) flag areas where the map should not be over-interpreted. We treat these labels as **map-quality guidance**, not biophysical flexibility measurements.

## Draft paragraphs (discussion / limitations)

> Reliability scoring identifies trustworthy regions **inside** the density contour. Flexible segments below the contour or absent from the map are invisible to this analysis. Future work should test build-zone transfer across a multi-map cohort and compare against deposited models residue-by-residue when PDB coordinates are available.

---

## Files generated

| File | Description |
|------|-------------|
| `reliability.npz` | reliability_score, H_repro, build_zone |
| `emd_{emd_id}_reliability.mrc` | Reliability overlay (0–1 score) |
| `emd_{emd_id}_build_zones.mrc` | 0/1/2 zone labels |
| `figures/model_building_row.png` | CC, reliability score, build zones (one slice) |
| `../analysis/figures/analysis_validation_panel.png` | Anchor map: 2×2 variance / reliability validation |
| `run_metadata.json` | Spearman / partial ρ (cohort heatmap reads this) |

**Inputs:** `{paths['reference'].name}`, `{paths['features'].name}`, half-maps, `{paths.get('halfmap_npz', 'halfmap_metrics.npz')}`.

See also: `docs/LH_MAP_RELIABILITY.md`, `docs/STATISTICS_METHODS.md`.
"""
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _paths(args)
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for k, p in paths.items():
        if not p.exists():
            print(f"[lh_map_reliability] ERROR: missing {k}: {p}", file=sys.stderr)
            return 2
    if not args.halfmap_npz.exists():
        print(f"[lh_map_reliability] ERROR: missing {args.halfmap_npz}", file=sys.stderr)
        return 2

    print("[lh_map_reliability] loading reference + mask", flush=True)
    reference = load_mrc(paths["reference"], dtype=np.float32)
    mask = build_contour_mask(reference, args.contour)
    n_mask = int(mask.sum())
    print(f"[lh_map_reliability] mask {n_mask:,} voxels at contour {args.contour}", flush=True)

    local_var = _load_local_var(paths["features"])
    bundle = load_full_and_half_maps(
        paths["reference"], paths["half1"], paths["half2"],
        reference="full", dtype=np.float32, resample_if_needed=True,
    )
    rho = _zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    full_shape = reference.shape
    pad = pad_voxels_for_filters(window=args.window)
    if args.no_crop_to_contour:
        work: dict[str, np.ndarray] = {"density_normalized": rho, "local_variance": local_var}
        attach_reliability_to_features(
            work, bundle.half1.data, bundle.half2.data, window=args.window, mask=mask
        )
        feats = work
    else:
        bbox = bbox_from_mask(mask, pad=pad)
        print(
            f"[lh_map_reliability] contour crop: {format_bbox_log(bbox, full_shape, pad=pad)}",
            flush=True,
        )
        work = {
            "density_normalized": crop_array(rho, bbox),
            "local_variance": crop_array(local_var, bbox),
        }
        attach_reliability_to_features(
            work,
            crop_array(bundle.half1.data, bbox),
            crop_array(bundle.half2.data, bbox),
            window=args.window,
            mask=crop_array(mask, bbox),
        )
        rel_keys = (
            "reliability_score",
            "reliability_H_repro",
            "reliability_fluctuation",
            "reliability_smoothness",
            "build_zone",
        )
        feats = {
            k: embed_array(full_shape, bbox, work[k], dtype=work[k].dtype)
            for k in rel_keys
        }
    del bundle
    gc.collect()

    with np.load(args.halfmap_npz, allow_pickle=False) as hm:
        cc = np.asarray(hm["local_cross_correlation"], dtype=np.float32)

    compare = {
        "reliability_score": feats["reliability_score"],
        "reliability_H_repro": feats["reliability_H_repro"],
        "local_variance": local_var,
    }
    result = compute_feature_target_correlations(
        compare, cc, mask, target_name="local_cross_correlation", methods=("spearman",),
        max_samples=2_000_000,
    )
    spearman = {c.feature_name: c.correlation for c in result.correlations}

    # Partial vs variance
    idx = np.flatnonzero(mask)
    y = cc.ravel()[idx]
    ctrl = local_var.ravel()[idx]
    partial: dict[str, float] = {}

    def _partial(x, y, z):
        xr, yr, zr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        r_xy = np.corrcoef(xr, yr)[0, 1]
        r_xz = np.corrcoef(xr, zr)[0, 1]
        r_yz = np.corrcoef(yr, zr)[0, 1]
        d = (1 - r_xz * r_xz) * (1 - r_yz * r_yz)
        return (r_xy - r_xz * r_yz) / np.sqrt(d) if d > 0 else float("nan")

    for name in ("reliability_score", "reliability_H_repro"):
        partial[name] = float(_partial(compare[name].ravel()[idx], y, ctrl))

    zones = feats["build_zone"]
    zone_counts = {int(z): int((zones[mask] == z).sum()) for z in (0, 1, 2)}

    # Save NPZ + MRC
    np.savez_compressed(
        out_dir / "reliability.npz",
        reliability_score=feats["reliability_score"],
        reliability_H_repro=feats["reliability_H_repro"],
        reliability_fluctuation=feats["reliability_fluctuation"],
        reliability_smoothness=feats["reliability_smoothness"],
        build_zone=zones,
        contour=np.float32(args.contour),
        emd_id=np.array(args.emd_id),
    )
    save_reliability_mrc(paths["reference"], feats["reliability_score"], out_dir / f"emd_{args.emd_id}_reliability.mrc")
    save_build_zone_mrc(paths["reference"], zones, out_dir / f"emd_{args.emd_id}_build_zones.mrc")
    (out_dir / "run_metadata.json").write_text(
        json.dumps({"spearman": spearman, "partial": partial, "zone_counts": zone_counts, "n_mask": n_mask}, indent=2) + "\n"
    )

    # Figures
    z = pick_slice_index(mask, axis=0)
    msl = mask[z]
    crop = slice_crop_from_mask(msl, pad_voxels=args.zoom_padding) if args.zoom_padding else None
    kw = {"crop_bbox": crop, "already_contoured": True}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        (extract_slice(cc, axis=0, index=z), "RdYlGn", "half-map CC", {**kw, "vmin": 0, "vmax": 1, "robust": False}),
        (extract_slice(feats["reliability_score"], axis=0, index=z), "viridis", "reliability score", kw),
        (extract_slice(zones.astype(float), axis=0, index=z), None, "build zones", {}),
    ]
    cbar_labels = {"half-map CC": "half-map CC", "reliability score": "reliability score"}
    for letter, (ax, (sl, cm, title, pkw)) in zip("abc", zip(axes, panels)):
        if title == "build zones":
            sl_c = sl if crop is None else sl[crop[0]:crop[1], crop[2]:crop[3]]
            m_c = msl if crop is None else msl[crop[0]:crop[1], crop[2]:crop[3]]
            _plot_build_zones(ax, sl_c, m_c, title=f"{title}\nZ={z}")
        else:
            plot_masked_slice(
                ax,
                sl,
                msl,
                cmap=cm,
                title=f"{title}\nZ={z}",
                cbar_label=cbar_labels.get(title),
                **pkw,
            )
        label_panel(ax, letter)
    fig.suptitle(f"EMD-{args.emd_id} model-building guidance (mask ρ≥{args.contour})", fontsize=12)
    fig.tight_layout()
    save_nature(fig, fig_dir / "model_building_row.png", dpi=args.dpi)
    plt.close(fig)

    write_panel = (
        args.write_analysis_panel
        or (str(args.emd_id).strip() == ANCHOR_EMDB_ID and not args.no_write_analysis_panel)
    )
    if write_panel:
        feature_maps = load_feature_maps(paths["features"])
        panel_path = analysis_dir(args.emd_id) / "figures" / "analysis_validation_panel.png"
        plot_analysis_validation_panel(
            feature_maps,
            {"local_cross_correlation": cc},
            mask,
            reliability_score=feats["reliability_score"],
            spearman=spearman,
            emd_id=str(args.emd_id),
            contour=args.contour,
            save_path=panel_path,
            dpi=args.dpi,
        )
        print(f"[lh_map_reliability] wrote {panel_path}", flush=True)

    if args.prune_retired_figures:
        removed = prune_lh_retired_figures(fig_dir)
        if removed:
            print(f"[lh_map_reliability] pruned {len(removed)} retired figure(s)", flush=True)

    paths_meta = {**paths, "halfmap_npz": args.halfmap_npz}
    _write_thesis_md(
        out_dir / "LH_MAP_RELIABILITY_RESULTS.md",
        emd_id=args.emd_id,
        contour=args.contour,
        n_mask=n_mask,
        spearman=spearman,
        partial=partial,
        zone_counts=zone_counts,
        paths=paths_meta,
    )

    print(f"[lh_map_reliability] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
