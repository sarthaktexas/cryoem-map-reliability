"""Thesis bundle: reliability maps, build zones, figures, and write-up (EMD-49450 defaults).

Generates under ``outputs/emd_<ID>/lh_map_reliability/``:

- ``reliability.npz`` — reliability_score, H_repro, LH maps (T, V, L), build_zone
- ``*.mrc`` — ChimeraX overlays on deposited reference grid
- ``figures/`` — slice panels, correlation bars, build-zone map, CC vs reliability
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
from scipy import stats

from cryoem_mrc.analysis import (
    build_contour_mask,
    binned_feature_by_target,
    compute_feature_target_correlations,
)
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.reliability import (
    attach_reliability_to_features,
    save_build_zone_mrc,
    save_reliability_mrc,
)
from cryoem_mrc.repo_paths import DATA_ROOT, halfmap_metrics_npz, lh_map_reliability_dir
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


def _load_rho_var(features_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(features_path, allow_pickle=False) as d:
        return (
            np.asarray(d["density_normalized"], dtype=np.float32),
            np.asarray(d["local_variance"], dtype=np.float32),
        )


def _plot_build_zones(ax, zones_sl: np.ndarray, mask_sl: np.ndarray, *, title: str) -> None:
    from matplotlib.colors import ListedColormap

    show = np.ma.masked_where(~mask_sl, zones_sl.astype(float))
    cmap = ListedColormap(["#d62728", "#ffbb78", "#2ca02c"])  # omit, caution, build
    im = ax.imshow(show, cmap=cmap, vmin=0, vmax=2, origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["omit", "caution", "build"])
    return im


def _plot_spearman_bar(spearman: dict[str, float], out: Path) -> None:
    order = [
        "local_variance",
        "reliability_score",
        "reliability_H_repro",
        "local_cross_correlation",
    ]
    extra = [k for k in spearman if k not in order]
    labels = [k for k in order if k in spearman] + sorted(extra)
    vals = [abs(spearman[k]) for k in labels]
    colors = ["#9467bd" if k == "reliability_score" else "#7f7f7f" for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(np.arange(len(labels)), vals, color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("|Spearman r| vs half-map CC")
    ax.set_title("Reliability predictors (EMD-49450, mask ρ≥0.116)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_partial_bar(rows: list[tuple[str, float]], out: Path) -> None:
    labels = [r[0] for r in rows]
    vals = [abs(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(np.arange(len(labels)), vals, color="#1f77b4", alpha=0.85)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("|partial Spearman| vs CC | local_variance")
    ax.set_title("Incremental value beyond local variance")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
| `emd_{emd_id}_reliability.mrc` | ChimeraX overlay (0–1 score) |
| `emd_{emd_id}_build_zones.mrc` | 0/1/2 zone labels |
| `figures/model_building_row.png` | CC, reliability, zones, variance |
| `figures/spearman_predictors.png` | Correlation bar chart |
| `figures/partial_incremental.png` | Partial correlation bar |
| `figures/reliability_vs_cc_binned.png` | Binned mean curve |

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

    rho, local_var = _load_rho_var(paths["features"])
    bundle = load_full_and_half_maps(
        paths["reference"], paths["half1"], paths["half2"],
        reference="full", dtype=np.float32, resample_if_needed=True,
    )
    feats: dict[str, np.ndarray] = {"density_normalized": rho, "local_variance": local_var}
    attach_reliability_to_features(
        feats, bundle.half1.data, bundle.half2.data, window=args.window, mask=mask
    )
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

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [
        (extract_slice(cc, axis=0, index=z), "RdYlGn", "half-map CC", {**kw, "vmin": 0, "vmax": 1, "robust": False}),
        (extract_slice(feats["reliability_score"], axis=0, index=z), "viridis", "reliability score", kw),
        (extract_slice(zones.astype(float), axis=0, index=z), None, "build zones", {}),
        (extract_slice(local_var, axis=0, index=z), "magma", "local variance", kw),
    ]
    for ax, (sl, cm, title, pkw) in zip(axes, panels):
        if title == "build zones":
            sl_c = sl if crop is None else sl[crop[0]:crop[1], crop[2]:crop[3]]
            m_c = msl if crop is None else msl[crop[0]:crop[1], crop[2]:crop[3]]
            _plot_build_zones(ax, sl_c, m_c, title=f"{title}\nZ={z}")
        else:
            plot_masked_slice(ax, sl, msl, cmap=cm, title=f"{title}\nZ={z}", **pkw)
    fig.suptitle(f"EMD-{args.emd_id} model-building guidance (mask ρ≥{args.contour})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_building_row.png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    _plot_spearman_bar(spearman, fig_dir / "spearman_predictors.png")
    _plot_partial_bar(list(partial.items()), fig_dir / "partial_incremental.png")

    binned = binned_feature_by_target(
        feats["reliability_score"], cc, mask,
        feature_name="reliability_score", target_name="local_cross_correlation", n_bins=10,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = np.isfinite(binned.mean_feature)
    ax.errorbar(
        binned.bin_centers[ok], binned.mean_feature[ok],
        yerr=binned.std_feature[ok], fmt="o-", capsize=3,
    )
    ax.set_xlabel("half-map CC (bin center)")
    ax.set_ylabel("mean reliability_score")
    ax.set_title("Reliability vs half-map agreement (binned)")
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability_vs_cc_binned.png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
