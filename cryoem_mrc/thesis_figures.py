"""Publication-style slice panels for thesis overview / schematic figures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .conformation_pair import interior_residue_indices
from .repo_paths import OUTPUTS_ROOT


def pick_slice_index(
    mask: np.ndarray,
    *,
    axis: int = 0,
    min_voxels: int = 500,
) -> int:
    """
    Choose a slice index along ``axis`` with substantial mask coverage.

    Prefers the slice with the most in-mask voxels; falls back to the volume center.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3D, got {m.shape}")
    counts = m.sum(axis=tuple(i for i in range(3) if i != axis))
    if counts.size == 0:
        return 0
    if counts.max() < min_voxels:
        return int(m.shape[axis] // 2)
    # Among slices with strong mask coverage, prefer near the volume center
    # (avoids picking an accidental edge slab when coverage is similar).
    threshold = 0.9 * float(counts.max())
    candidates = np.flatnonzero(counts >= threshold)
    center = (m.shape[axis] - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - center))])


def extract_slice(
    volume: np.ndarray,
    *,
    axis: int = 0,
    index: int,
) -> np.ndarray:
    """2D slice from a (Z, Y, X) volume."""
    vol = np.asarray(volume)
    if axis == 0:
        return vol[index, :, :]
    if axis == 1:
        return vol[:, index, :]
    return vol[:, :, index]


# (y0, y1, x0, x1) half-open row/col bounds on an XY slice (Z fixed).
SliceCrop = tuple[int, int, int, int]


def slice_crop_from_mask(
    mask_sl: np.ndarray,
    *,
    pad_voxels: int = 24,
) -> SliceCrop:
    """
    Tight bounding box around in-mask pixels on a 2D slice, plus ``pad_voxels``.

    Used to zoom thesis panels on the macromolecule instead of the full 430² box.
    """
    m = np.asarray(mask_sl, dtype=bool)
    ny, nx = m.shape
    if not m.any():
        return (0, ny, 0, nx)
    ys, xs = np.nonzero(m)
    y0 = max(0, int(ys.min()) - pad_voxels)
    y1 = min(ny, int(ys.max()) + 1 + pad_voxels)
    x0 = max(0, int(xs.min()) - pad_voxels)
    x1 = min(nx, int(xs.max()) + 1 + pad_voxels)
    return (y0, y1, x0, x1)


def crop_slice_2d(sl: np.ndarray, crop: SliceCrop) -> np.ndarray:
    """Crop a 2D array with ``(y0, y1, x0, x1)`` bounds."""
    y0, y1, x0, x1 = crop
    return np.asarray(sl)[y0:y1, x0:x1]


def mask_slice_values(
    sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    outside: float = np.nan,
) -> np.ndarray:
    """Return slice with out-of-mask voxels replaced by ``outside`` (default NaN)."""
    out = np.asarray(sl, dtype=np.float64).copy()
    m = np.asarray(mask_sl, dtype=bool)
    out[~m] = outside
    return out


def apply_contour_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    outside: float = np.nan,
) -> np.ndarray:
    """
    Restrict a 3D volume to the analysis contour (Decision 002).

    Voxels outside ``mask`` are set to ``outside`` (default NaN) so colormaps and
    percentiles reflect only the macromolecular region, not solvent.
    """
    out = np.asarray(volume, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    if out.shape != m.shape:
        raise ValueError(f"volume shape {out.shape} != mask shape {m.shape}")
    out[~m] = outside
    return out.astype(np.asarray(volume).dtype, copy=False)


def _robust_limits(
    sl: np.ndarray,
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    mask_sl: np.ndarray | None = None,
) -> tuple[float, float]:
    v = np.asarray(sl, dtype=np.float64).ravel()
    if mask_sl is not None:
        v = v[np.asarray(mask_sl, dtype=bool).ravel()]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _add_scale_bar(
    ax: plt.Axes,
    *,
    voxel_size_a: float,
    length_a: float = 50.0,
    xy: tuple[float, float] = (0.05, 0.05),
    color: str = "white",
) -> None:
    """Draw a horizontal scale bar in axes fraction coords (lower-left anchor)."""
    ny, nx = ax.images[0].get_array().shape
    bar_px = length_a / voxel_size_a
    x0 = xy[0] * nx
    y0 = xy[1] * ny
    rect = Rectangle(
        (x0, y0),
        bar_px,
        max(2.0, 0.01 * ny),
        linewidth=0,
        facecolor=color,
        edgecolor="black",
        clip_on=False,
        transform=ax.transData,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + bar_px / 2,
        y0 + max(4.0, 0.03 * ny),
        f"{length_a:g} Å",
        ha="center",
        va="bottom",
        color=color,
        fontsize=8,
        transform=ax.transData,
    )


def plot_masked_slice(
    ax: plt.Axes,
    sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = True,
    cbar_label: str | None = None,
    title: str | None = None,
    show_mask_outline: bool = False,
    already_contoured: bool = False,
    crop_bbox: SliceCrop | None = None,
) -> plt.cm.ScalarMappable:
    """
    Single 2D panel with contour-restricted display and optional colorbar.

    When ``already_contoured`` is False (default), out-of-mask voxels are cleared
    before scaling. Colormap limits use in-mask pixels only.

    ``crop_bbox`` zooms to ``(y0, y1, x0, x1)`` on the slice (see
    :func:`slice_crop_from_mask`).
    """
    if crop_bbox is not None:
        sl = crop_slice_2d(sl, crop_bbox)
        mask_sl = crop_slice_2d(mask_sl, crop_bbox)
    masked = sl if already_contoured else mask_slice_values(sl, mask_sl)
    if robust and (vmin is None or vmax is None):
        rlo, rhi = _robust_limits(masked, mask_sl=mask_sl)
        vmin = rlo if vmin is None else vmin
        vmax = rhi if vmax is None else vmax
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=(0.12, 0.12, 0.14, 1.0))
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="lower")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_mask_outline:
        ax.contour(mask_sl.astype(float), levels=(0.5,), colors="w", linewidths=0.4, alpha=0.6)
    if cbar_label:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return im


def plot_local_resolution_slice(
    local_res_a: np.ndarray,
    mask: np.ndarray,
    *,
    slice_index: int | None = None,
    axis: int = 0,
    voxel_size_a: float = 1.0,
    vmin_a: float | None = None,
    vmax_a: float | None = None,
    title: str = "Local resolution (Å)",
    save_path: str | Path | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (6.0, 5.5),
    add_scale_bar: bool = True,
    crop_bbox: SliceCrop | None = None,
    scale_bar_a: float = 50.0,
) -> plt.Figure:
    """Mid-slice (or chosen index) of the local-FSC resolution map inside the mask."""
    mask = np.asarray(mask, dtype=bool)
    z = slice_index if slice_index is not None else pick_slice_index(mask, axis=axis)
    sl = extract_slice(local_res_a, axis=axis, index=z)
    msl = extract_slice(mask, axis=axis, index=z)
    masked = mask_slice_values(sl, msl)
    finite = masked[np.isfinite(masked)]
    if vmin_a is None:
        vmin_a = float(np.nanpercentile(finite, 5)) if finite.size else 2.0
    if vmax_a is None:
        vmax_a = float(np.nanpercentile(finite, 95)) if finite.size else 8.0

    fig, ax = plt.subplots(figsize=figsize)
    plot_masked_slice(
        ax,
        sl,
        msl,
        cmap="viridis_r",
        vmin=vmin_a,
        vmax=vmax_a,
        robust=False,
        cbar_label="Å (lower = sharper)",
        title=f"{title}\nZ = {z}",
        crop_bbox=crop_bbox,
    )
    if add_scale_bar:
        _add_scale_bar(ax, voxel_size_a=voxel_size_a, length_a=scale_bar_a)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_parallel_reliability_readouts(
    density: np.ndarray,
    local_cc: np.ndarray,
    local_res_a: np.ndarray,
    mask: np.ndarray,
    *,
    slice_index: int | None = None,
    axis: int = 0,
    voxel_size_a: float = 1.0,
    contour: float | None = None,
    cc_vmin: float = 0.0,
    cc_vmax: float = 1.0,
    res_vmin_a: float | None = None,
    res_vmax_a: float | None = None,
    save_path: str | Path | None = None,
    dpi: int = 200,
    crop_bbox: SliceCrop | None = None,
    scale_bar_a: float = 50.0,
) -> plt.Figure:
    """
    Three-panel row: averaged density, windowed half-map CC, local FSC resolution (Å).

    Intended for the thesis overview schematic (parallel reliability readouts).
    """
    mask = np.asarray(mask, dtype=bool)
    z = slice_index if slice_index is not None else pick_slice_index(mask, axis=axis)
    d_sl = extract_slice(density, axis=axis, index=z)
    cc_sl = extract_slice(local_cc, axis=axis, index=z)
    res_sl = extract_slice(local_res_a, axis=axis, index=z)
    msl = extract_slice(mask, axis=axis, index=z)

    res_masked = mask_slice_values(res_sl, msl)
    finite = res_masked[np.isfinite(res_masked)]
    if res_vmin_a is None:
        res_vmin_a = float(np.nanpercentile(finite, 5)) if finite.size else 2.0
    if res_vmax_a is None:
        res_vmax_a = float(np.nanpercentile(finite, 95)) if finite.size else 8.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
    kw = {"crop_bbox": crop_bbox}
    plot_masked_slice(
        axes[0],
        d_sl,
        msl,
        cmap="gray",
        cbar_label="density",
        title=f"Averaged map ρ\nZ = {z}",
        already_contoured=True,
        **kw,
    )
    plot_masked_slice(
        axes[1],
        cc_sl,
        msl,
        cmap="RdYlGn",
        vmin=cc_vmin,
        vmax=cc_vmax,
        robust=False,
        cbar_label="CC (↑ reliable)",
        title="Half-map local CC",
        already_contoured=True,
        **kw,
    )
    plot_masked_slice(
        axes[2],
        res_sl,
        msl,
        cmap="viridis_r",
        vmin=res_vmin_a,
        vmax=res_vmax_a,
        robust=False,
        cbar_label="Å (↓ sharper)",
        title="Local FSC resolution",
        already_contoured=True,
        **kw,
    )
    if contour is not None:
        fig.suptitle(
            f"Parallel reliability readouts (mask ρ ≥ {contour:g}, windowed metrics)",
            fontsize=12,
            y=1.02,
        )
    _add_scale_bar(axes[0], voxel_size_a=voxel_size_a, length_a=scale_bar_a)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_feature_family_panel(
    feature_maps: Mapping[str, np.ndarray],
    keys: Sequence[str],
    mask: np.ndarray,
    *,
    family_title: str,
    slice_index: int,
    axis: int = 0,
    cmap: str = "magma",
    cmap_overrides: Mapping[str, str] | None = None,
    subtitles: Mapping[str, str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 200,
    ncol: int | None = None,
    crop_bbox: SliceCrop | None = None,
) -> plt.Figure:
    """
    One overview box: several feature channels on the same slice (for schematic figures).
    """
    keys = list(keys)
    n = len(keys)
    ncol = ncol or min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    axes_flat = np.atleast_1d(axes).ravel()
    overrides = dict(cmap_overrides or {})
    subs = dict(subtitles or {})
    msl = extract_slice(mask, axis=axis, index=slice_index)

    for ax, key in zip(axes_flat, keys):
        vol = np.asarray(feature_maps[key])
        sl = extract_slice(vol, axis=axis, index=slice_index)
        cm = overrides.get(key, cmap)
        plot_masked_slice(
            ax,
            sl,
            msl,
            cmap=cm,
            cbar_label=None,
            title=subs.get(key, key.replace("_", " ")),
            already_contoured=True,
            crop_bbox=crop_bbox,
        )
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(family_title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_reliability_pair_only(
    local_cc: np.ndarray,
    local_res_a: np.ndarray,
    mask: np.ndarray,
    *,
    slice_index: int,
    axis: int = 0,
    cc_vmin: float = 0.0,
    cc_vmax: float = 1.0,
    res_vmin_a: float | None = None,
    res_vmax_a: float | None = None,
    save_path: str | Path | None = None,
    dpi: int = 200,
    crop_bbox: SliceCrop | None = None,
) -> plt.Figure:
    """Two-panel row: CC and local resolution only (no density)."""
    msl = extract_slice(mask, axis=axis, index=slice_index)
    cc_sl = extract_slice(local_cc, axis=axis, index=slice_index)
    res_sl = extract_slice(local_res_a, axis=axis, index=slice_index)
    res_masked = mask_slice_values(res_sl, msl)
    finite = res_masked[np.isfinite(res_masked)]
    if res_vmin_a is None:
        res_vmin_a = float(np.nanpercentile(finite, 5)) if finite.size else 2.0
    if res_vmax_a is None:
        res_vmax_a = float(np.nanpercentile(finite, 95)) if finite.size else 8.0

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    kw = {"crop_bbox": crop_bbox, "already_contoured": True}
    plot_masked_slice(
        axes[0],
        cc_sl,
        msl,
        cmap="RdYlGn",
        vmin=cc_vmin,
        vmax=cc_vmax,
        robust=False,
        cbar_label="CC",
        title="Windowed half-map CC",
        **kw,
    )
    plot_masked_slice(
        axes[1],
        res_sl,
        msl,
        cmap="viridis_r",
        vmin=res_vmin_a,
        vmax=res_vmax_a,
        robust=False,
        cbar_label="Å",
        title="Local FSC resolution",
        **kw,
    )
    fig.suptitle(f"Same slice Z = {slice_index}", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_spearman_top_bar(
    correlations_csv: str | Path,
    *,
    top_n: int = 8,
    method: str = "spearman",
    save_path: str | Path | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """Horizontal bar chart of top |ρ| features from ``correlations.csv``."""
    import csv

    rows: list[dict[str, str]] = []
    with Path(correlations_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method", "").lower() == method.lower():
                rows.append(row)
    if not rows:
        raise ValueError(f"No {method} rows in {correlations_csv}")

    target = rows[0].get("target", "local_cross_correlation")
    rho_key = "correlation" if "correlation" in rows[0] else "spearman_r"
    ranked = sorted(
        rows,
        key=lambda r: abs(float(r[rho_key])),
        reverse=True,
    )[:top_n]
    labels = [r["feature"] for r in ranked]
    vals = [float(r[rho_key]) for r in ranked]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * top_n + 1.5))
    colors = ["#2c7bb6" if v >= 0 else "#d7191c" for v in vals]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel(f"{method.capitalize()} ρ vs {target}")
    ax.set_title(f"Top {top_n} map features (masked voxels)")
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


@dataclass(frozen=True)
class CohortMetricRow:
    """Per-map summary metrics for cohort comparison heatmaps."""

    emdb_id: str
    var_vs_cc: float
    rel_vs_cc: float
    partial_rel_given_var: float
    b_vs_rel: float = float("nan")


def collect_cohort_metrics(outputs_root: Path | None = None) -> list[CohortMetricRow]:
    """
    Gather Spearman summaries from ``outputs/emd_<ID>/lh_map_reliability/``.

    Reads ``run_metadata.json`` (voxel-level CC correlations) and, when present,
    ``bfactor_validation_stats.json`` (residue-level B vs reliability).
    """
    root = OUTPUTS_ROOT if outputs_root is None else Path(outputs_root)
    rows: list[CohortMetricRow] = []
    for meta_path in sorted(root.glob("emd_*/lh_map_reliability/run_metadata.json")):
        emdb_id = meta_path.parent.parent.name.removeprefix("emd_")
        with meta_path.open() as f:
            meta = json.load(f)
        spearman = meta.get("spearman", {})
        partial = meta.get("partial", {})
        b_vs_rel = float("nan")
        bfac_path = meta_path.parent / "bfactor_validation_stats.json"
        if bfac_path.is_file():
            with bfac_path.open() as f:
                bfac = json.load(f)
            b_vs_rel = float(bfac.get("spearman_b_vs_reliability", float("nan")))
        rows.append(
            CohortMetricRow(
                emdb_id=emdb_id,
                var_vs_cc=float(spearman.get("local_variance", float("nan"))),
                rel_vs_cc=float(spearman.get("reliability_score", float("nan"))),
                partial_rel_given_var=float(partial.get("reliability_score", float("nan"))),
                b_vs_rel=b_vs_rel,
            )
        )
    return rows


def write_cohort_metrics_csv(rows: Sequence[CohortMetricRow], path: str | Path) -> None:
    """Write :func:`collect_cohort_metrics` output as a tidy CSV."""
    path = Path(path)
    fieldnames = [
        "emdb_id",
        "var_vs_cc",
        "rel_vs_cc",
        "partial_rel_given_var",
        "b_vs_rel",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "emdb_id": row.emdb_id,
                    "var_vs_cc": f"{row.var_vs_cc:.6f}",
                    "rel_vs_cc": f"{row.rel_vs_cc:.6f}",
                    "partial_rel_given_var": f"{row.partial_rel_given_var:.6f}",
                    "b_vs_rel": f"{row.b_vs_rel:.6f}" if np.isfinite(row.b_vs_rel) else "",
                }
            )


def plot_cohort_metrics_heatmap(
    rows: Sequence[CohortMetricRow],
    *,
    save_path: str | Path | None = None,
    dpi: int = 200,
    include_b_factor: bool = True,
) -> plt.Figure:
    """
    Maps × metrics heatmap for cross-cohort comparison (thesis Results §3.2).

    Columns: local_variance vs CC, reliability vs CC, partial reliability | variance,
    and optionally B_iso vs reliability when ``bfactor_validation_stats.json`` exists.
    """
    if not rows:
        raise ValueError("No cohort metric rows to plot")

    metric_cols: list[tuple[str, str]] = [
        ("var_vs_cc", "local variance ↔ CC"),
        ("rel_vs_cc", "reliability ↔ CC"),
        ("partial_rel_given_var", "partial rel | variance"),
    ]
    if include_b_factor:
        metric_cols.append(("b_vs_rel", "B_iso ↔ reliability"))

    labels = [f"EMD-{r.emdb_id}" for r in rows]
    data = np.array(
        [[getattr(r, key) for key, _ in metric_cols] for r in rows],
        dtype=np.float64,
    )
    col_titles = [title for _, title in metric_cols]

    fig_h = max(3.0, 0.45 * len(rows) + 1.5)
    fig_w = max(6.0, 1.4 * len(col_titles) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e0e0e0")
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_titles)), col_titles, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    ax.set_title("Cohort summary: masked Spearman correlations", fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > 0.55 else "0.15"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def _sorted_conformation_deltas(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool = True,
) -> tuple[list[tuple[object, object]], np.ndarray, np.ndarray, list[str]] | None:
    """Matched in-mask pairs sorted by chain order with ΔB_iso and Δreliability (B − A)."""
    use = list(pairs)
    if in_mask_both:
        use = [(a, b) for a, b in pairs if a.in_contour_mask and b.in_contour_mask]
    if len(use) < 10:
        return None
    use.sort(key=lambda ab: (ab[0].chain, ab[0].seq_num, ab[0].seq_icode))
    db = np.array([b.b_iso - a.b_iso for a, b in use], dtype=np.float64)
    drel = np.array([b.reliability_score - a.reliability_score for a, b in use], dtype=np.float64)
    chains = [a.chain for a, _ in use]
    return use, db, drel, chains


def _local_profile_cross_corr_matrix(
    a: np.ndarray,
    b: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    """
    Residue × residue matrix of Pearson r between local sequence windows.

    Entry (i, j) compares a window centered at residue i in ``a`` with a window centered
    at j in ``b``. Diagonal entries summarize local ΔB vs Δreliability co-variation
    (AlphaFold-PAE-style layout on sequence axes).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = int(a.size)
    w = 2 * half_window + 1
    if n < w:
        return np.full((n, n), np.nan, dtype=np.float64)

    win_a = np.full((n, w), np.nan, dtype=np.float64)
    win_b = np.full((n, w), np.nan, dtype=np.float64)
    for i in range(n):
        i0, i1 = i - half_window, i + half_window + 1
        if i0 >= 0 and i1 <= n:
            win_a[i] = a[i0:i1]
            win_b[i] = b[i0:i1]

    mean_a = np.nanmean(win_a, axis=1, keepdims=True)
    mean_b = np.nanmean(win_b, axis=1, keepdims=True)
    std_a = np.nanstd(win_a, axis=1, ddof=0, keepdims=True)
    std_b = np.nanstd(win_b, axis=1, ddof=0, keepdims=True)
    valid = (
        np.all(np.isfinite(win_a), axis=1)
        & np.all(np.isfinite(win_b), axis=1)
        & np.isfinite(std_a[:, 0])
        & np.isfinite(std_b[:, 0])
        & (std_a[:, 0] > 0)
        & (std_b[:, 0] > 0)
    )
    za = np.where(valid[:, None], (win_a - mean_a) / std_a, 0.0)
    zb = np.where(valid[:, None], (win_b - mean_b) / std_b, 0.0)
    corr = (za @ zb.T) / w
    corr[~valid, :] = np.nan
    corr[:, ~valid] = np.nan
    return corr


def _chain_break_positions(chains: Sequence[str]) -> list[int]:
    return [i for i in range(1, len(chains)) if chains[i] != chains[i - 1]]


def _normalize_strip(values: np.ndarray, *, percentile: float = 98) -> np.ndarray:
    """Scale to ±1 by in-metric magnitude so strips share one colorbar across units."""
    values = np.asarray(values, dtype=np.float64)
    lim = float(np.nanpercentile(np.abs(values), percentile)) if values.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        return np.zeros_like(values)
    return values / lim


def plot_conformation_sequence_strip(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """
    Two-row sequence strip: ΔB_iso and Δreliability along matched Cα (state B − A).

    Residues are ordered by (chain, seq_num, seq_icode). Vertical lines mark chain breaks.
    Returns ``None`` when fewer than 10 in-mask matched residues.
    """
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, db, drel, chains = packed
    strip_rows: list[tuple[str, np.ndarray]] = [
        (f"ΔB_iso ({emdb_b} − {emdb_a})", _normalize_strip(db)),
        (f"Δreliability ({emdb_b} − {emdb_a})", _normalize_strip(drel)),
    ]

    n = len(use)
    fig_w = min(16.0, max(7.0, n * 0.006))
    fig, axes = plt.subplots(
        len(strip_rows),
        1,
        figsize=(fig_w, 1.35 * len(strip_rows) + 1.2),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    cmap = plt.get_cmap("RdBu_r")
    tick_idx = np.linspace(0, n - 1, num=min(12, n), dtype=int)
    tick_labels = [
        f"{use[i][0].chain}:{use[i][0].seq_num}{use[i][0].seq_icode.strip() or ''}"
        for i in tick_idx
    ]

    im = None
    for ax, (ylabel, values) in zip(axes[:, 0], strip_rows):
        row = values.reshape(1, -1)
        im = ax.imshow(
            row,
            aspect="auto",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_yticks([0], [ylabel], fontsize=9)
        ax.set_xticks([])
        for i in range(1, n):
            if chains[i] != chains[i - 1]:
                ax.axvline(i - 0.5, color="0.2", lw=0.6, alpha=0.7)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:, 0], fraction=0.02, pad=0.02)
        cbar.set_label("normalized Δ (98th pct → ±1)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    axes[-1, 0].set_xticks(tick_idx, tick_labels, rotation=45, ha="right", fontsize=7)
    axes[-1, 0].set_xlabel("Residue index (chain order)", fontsize=9)
    fig.suptitle(
        f"Conformation pair EMD-{emdb_a} vs EMD-{emdb_b} (n={n} in-mask Cα)",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_conformation_pair_coupling_heatmap(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    half_window: int | None = None,
    spearman_rho: float | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """
    AF3-style coupling figure: ΔB strip (top), Δreliability strip (left), local cross-correlation
    heatmap (center) on matched Cα sequence indices.

    Center panel entry (i, j) is Pearson r between a sequence window around residue i in
    ΔB_iso and a window around j in Δreliability. Strong diagonal band → local co-variation
    of deposited B-factor change with map-reliability change between conformations.
    """
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, db, drel, chains = packed
    n = len(use)
    hw = half_window if half_window is not None else max(5, min(21, n // 25))
    corr_full = _local_profile_cross_corr_matrix(db, drel, half_window=hw)
    corr, db_i, drel_i, use_i, _idx = _coupling_interior_slice(
        corr_full, db, drel, use, half_window=hw
    )
    n_i = len(use_i)

    strip_cmap = plt.get_cmap("RdBu_r")
    db_strip = _normalize_strip(db_i)
    drel_strip = _normalize_strip(drel_i)
    chains_i = [a.chain for a, _ in use_i]

    size = min(11.0, max(7.0, n_i * 0.007))
    fig = plt.figure(figsize=(size, size), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.06, 1.0, 0.05],
        height_ratios=[0.06, 1.0],
        wspace=0.03,
        hspace=0.03,
    )
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    ax_corner.axis("off")

    x0, x1 = -0.5, n_i - 0.5
    y0, y1 = -0.5, n_i - 0.5

    breaks = _chain_break_positions(chains_i)

    im_top = ax_top.imshow(
        db_strip.reshape(1, -1),
        extent=(x0, x1, -0.5, 0.5),
        aspect="auto",
        cmap=strip_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    ax_top.set_xlim(x0, x1)
    ax_top.set_ylim(-0.5, 0.5)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_frame_on(False)
    ax_top.set_title(f"ΔB_iso ({emdb_b} − {emdb_a})", fontsize=8, pad=2)
    for i in breaks:
        ax_top.axvline(i - 0.5, color="0.15", lw=0.5, alpha=0.8)

    im_left = ax_left.imshow(
        drel_strip.reshape(-1, 1),
        extent=(-0.5, 0.5, y0, y1),
        aspect="auto",
        cmap=strip_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    ax_left.set_xlim(-0.5, 0.5)
    ax_left.set_ylim(y0, y1)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_frame_on(False)
    ax_left.set_ylabel(
        f"Δreliability ({emdb_b} − {emdb_a})",
        fontsize=8,
        labelpad=2,
    )
    for i in breaks:
        ax_left.axhline(i - 0.5, color="0.15", lw=0.5, alpha=0.8)

    corr_cmap = plt.get_cmap("RdBu_r")
    im_main = ax_main.imshow(
        corr,
        extent=(x0, x1, y0, y1),
        aspect="equal",
        origin="lower",
        cmap=corr_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_main.set_xlim(x0, x1)
    ax_main.set_ylim(y0, y1)
    for i in breaks:
        ax_main.axvline(i - 0.5, color="0.2", lw=0.4, alpha=0.55)
        ax_main.axhline(i - 0.5, color="0.2", lw=0.4, alpha=0.55)

    tick_idx = np.linspace(0, n_i - 1, num=min(10, n_i), dtype=int)
    tick_labels = [
        f"{use_i[i][0].chain}:{use_i[i][0].seq_num}{use_i[i][0].seq_icode.strip() or ''}"
        for i in tick_idx
    ]
    ax_main.set_xticks(tick_idx, tick_labels, rotation=45, ha="right", fontsize=7)
    ax_main.set_yticks(tick_idx, tick_labels, fontsize=7)
    ax_main.set_xlabel("Residue index (chain order)", fontsize=9)
    ax_main.set_ylabel("Residue index (chain order)", fontsize=9)

    rho_txt = ""
    if spearman_rho is not None and np.isfinite(spearman_rho):
        rho_txt = f" · Spearman ρ(ΔB, Δrel) = {spearman_rho:+.2f}"
    ax_main.set_title(
        f"Local ΔB vs Δreliability coupling (window ±{hw} res.){rho_txt}",
        fontsize=9,
    )

    strip_cbar = fig.colorbar(im_top, ax=ax_top, location="top", fraction=0.85, pad=0.08)
    strip_cbar.set_label("normalized Δ (98th pct → ±1)", fontsize=8)
    strip_cbar.ax.tick_params(labelsize=7)

    cbar = fig.colorbar(im_main, cax=ax_cbar)
    cbar.set_label("Pearson r (local windows)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Conformation pair EMD-{emdb_a} vs EMD-{emdb_b} (n={n_i} interior / {n} in-mask Cα)",
        fontsize=10,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_conformation_delta_joint_heatmap(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    spearman_rho: float | None = None,
    bins: int = 40,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """
    Joint density of per-residue (ΔB_iso, Δreliability) with 1D marginal histograms.

    Complements :func:`plot_conformation_pair_coupling_heatmap`: shows global co-variation
    of the two delta tracks (2D histogram + scatter overlay).
    """
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, db, drel, _chains = packed
    n = len(use)

    db_lim = float(np.nanpercentile(np.abs(db), 98)) if db.size else 1.0
    drel_lim = float(np.nanpercentile(np.abs(drel), 98)) if drel.size else 1.0
    if not np.isfinite(db_lim) or db_lim <= 0:
        db_lim = 1.0
    if not np.isfinite(drel_lim) or drel_lim <= 0:
        drel_lim = 1.0

    fig = plt.figure(figsize=(6.5, 6.0), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.2],
        height_ratios=[0.2, 1.0],
        wspace=0.04,
        hspace=0.04,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_top.hist(db, bins=bins, range=(-db_lim, db_lim), color="#9467bd", alpha=0.85, edgecolor="none")
    ax_top.set_ylabel("count", fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=8)

    hist, xedges, yedges = np.histogram2d(
        db,
        drel,
        bins=bins,
        range=[[-db_lim, db_lim], [-drel_lim, drel_lim]],
    )
    hist = hist.T
    ax_main.imshow(
        hist,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
        interpolation="nearest",
    )
    ax_main.scatter(db, drel, s=8, alpha=0.3, c="white", edgecolors="none", linewidths=0)
    ax_main.axhline(0, color="0.75", lw=0.7)
    ax_main.axvline(0, color="0.75", lw=0.7)
    ax_main.set_xlabel(f"ΔB_iso ({emdb_b} − {emdb_a})")
    ax_main.set_ylabel(f"Δreliability ({emdb_b} − {emdb_a})")
    rho_txt = ""
    if spearman_rho is not None and np.isfinite(spearman_rho):
        rho_txt = f" (Spearman ρ = {spearman_rho:+.2f})"
    ax_main.set_title(f"Per-residue joint density{rho_txt}", fontsize=9)

    ax_right.hist(
        drel,
        bins=bins,
        range=(-drel_lim, drel_lim),
        color="#1f77b4",
        alpha=0.85,
        edgecolor="none",
        orientation="horizontal",
    )
    ax_right.tick_params(labelleft=False, labelsize=8)

    fig.suptitle(
        f"Conformation pair EMD-{emdb_a} vs EMD-{emdb_b} (n={n} in-mask Cα)",
        fontsize=10,
        y=0.98,
    )
    fig.subplots_adjust(top=0.93)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def compute_conformation_coupling(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool = True,
    half_window: int | None = None,
) -> dict[str, object] | None:
    """
    Coupling matrices and per-residue Δ tracks for export / ChimeraX / triptych.

    Returns full in-mask arrays plus an interior crop with complete local windows.
    """
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, db, drel, chains = packed
    n = len(use)
    hw = half_window if half_window is not None else max(5, min(21, n // 25))
    corr = _local_profile_cross_corr_matrix(db, drel, half_window=hw)
    row_mean_abs = np.nanmean(np.abs(corr), axis=1)
    corr_i, db_i, drel_i, use_i, idx = _coupling_interior_slice(
        corr, db, drel, use, half_window=hw
    )
    return {
        "use": use,
        "db": db,
        "drel": drel,
        "chains": chains,
        "corr": corr,
        "row_mean_abs": row_mean_abs,
        "half_window": hw,
        "interior_use": use_i,
        "interior_corr": corr_i,
        "interior_db": db_i,
        "interior_drel": drel_i,
        "interior_indices": idx,
    }


def _hierarchical_cluster(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average-linkage order and linkage matrix for dendrogram + reordering."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    n = int(corr.shape[0])
    if n < 3:
        return np.arange(n, dtype=int), np.zeros((0, 4), dtype=np.float64)
    profiles = np.nan_to_num(corr, nan=0.0)
    d = pdist(profiles, metric="euclidean")
    if not np.all(np.isfinite(d)) or d.size == 0:
        return np.arange(n, dtype=int), np.zeros((0, 4), dtype=np.float64)
    z = linkage(d, method="average")
    return np.asarray(leaves_list(z), dtype=int), z


def _hierarchical_residue_order(corr: np.ndarray) -> np.ndarray:
    order, _ = _hierarchical_cluster(corr)
    return order


def _coupling_interior_slice(
    corr: np.ndarray,
    db: np.ndarray,
    drel: np.ndarray,
    use: list[tuple[object, object]],
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[object, object]], np.ndarray]:
    """Restrict to residues with full local windows (removes edge NaN bands)."""
    idx = interior_residue_indices(len(use), half_window)
    if idx.size < 10:
        idx = np.arange(len(use), dtype=int)
    sub_use = [use[int(i)] for i in idx]
    return (
        corr[np.ix_(idx, idx)],
        db[idx],
        drel[idx],
        sub_use,
        idx,
    )


def _set_equal_3d_limits(ax, coords: np.ndarray) -> None:
    """Cube bounding box so the structure is not sheared."""
    ctr = coords.mean(axis=0)
    span = float(np.max(np.linalg.norm(coords - ctr, axis=1)))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)


def plot_conformation_pair_summary_triptych(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    half_window: int | None = None,
    spearman_rho: float | None = None,
    coverage_note: str | None = None,
    coords_b_aligned: np.ndarray | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """
    Three-panel conformation-pair summary for publication.

    A) Coupling matrix + dendrogram, clustered (interior residues only).
    B) Cα trace (state A) colored by row-mean |coupling|; optional aligned B overlay.
    C) Per-residue ΔB vs Δreliability scatter with Spearman ρ.
    """
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, db, drel, _chains = packed
    n_full = len(use)
    db_full, drel_full = db, drel
    hw = half_window if half_window is not None else max(5, min(21, n_full // 25))
    corr_full = _local_profile_cross_corr_matrix(db, drel, half_window=hw)
    corr, db, drel, use, _idx = _coupling_interior_slice(
        corr_full, db, drel, use, half_window=hw
    )
    n = len(use)
    row_mean_abs = np.nanmean(np.abs(corr), axis=1)
    order, z_link = _hierarchical_cluster(corr)
    corr_ord = corr[np.ix_(order, order)]

    coords = np.array([[a.x, a.y, a.z] for a, _ in use], dtype=np.float64)
    color_vals = row_mean_abs

    fig = plt.figure(figsize=(17.5, 5.8), facecolor="white", layout="constrained")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 0.95], wspace=0.22)
    inner = outer[0].subgridspec(
        2, 2, width_ratios=[0.08, 1.0], height_ratios=[0.08, 1.0], wspace=0.02, hspace=0.02
    )
    ax_corner = fig.add_subplot(inner[0, 0])
    ax_dendro_top = fig.add_subplot(inner[0, 1])
    ax_dendro_left = fig.add_subplot(inner[1, 0])
    ax_a = fig.add_subplot(inner[1, 1])
    ax_b = fig.add_subplot(outer[1], projection="3d")
    ax_c = fig.add_subplot(outer[2])

    ax_corner.axis("off")

    if z_link.size:
        from scipy.cluster.hierarchy import dendrogram

        dendrogram(
            z_link,
            ax=ax_dendro_top,
            orientation="top",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="0.25",
        )
        dendrogram(
            z_link,
            ax=ax_dendro_left,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="0.25",
        )
    ax_dendro_top.axis("off")
    ax_dendro_left.axis("off")

    corr_cmap = plt.get_cmap("RdBu_r")
    im_a = ax_a.imshow(
        corr_ord,
        origin="lower",
        aspect="equal",
        cmap=corr_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_a.set_xlabel("Clustered residue index", fontsize=9)
    ax_a.set_ylabel("Clustered residue index", fontsize=9)
    ax_a.set_title(f"A · Clustered coupling (±{hw} res.; n={n} interior)", fontsize=10)
    tick_n = min(8, n)
    tick_idx = np.linspace(0, n - 1, num=tick_n, dtype=int)
    ax_a.set_xticks(tick_idx)
    ax_a.set_yticks(tick_idx)
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.02)
    cbar_a.set_label("Pearson r", fontsize=8)

    if coords_b_aligned is not None and coords_b_aligned.shape == coords.shape:
        ax_b.scatter(
            coords_b_aligned[:, 0],
            coords_b_aligned[:, 1],
            coords_b_aligned[:, 2],
            c="0.75",
            s=max(1.0, 800 / n),
            alpha=0.35,
            linewidths=0,
            label=f"EMD-{emdb_b} (aligned)",
        )
    sc = ax_b.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color_vals,
        cmap="YlOrRd",
        s=max(1.5, 1200 / n),
        alpha=0.9,
        linewidths=0,
        label=f"EMD-{emdb_a}",
    )
    _set_equal_3d_limits(ax_b, coords)
    ax_b.view_init(elev=22, azim=-68)
    ax_b.set_xlabel("x (Å)", fontsize=8, labelpad=-2)
    ax_b.set_ylabel("y (Å)", fontsize=8, labelpad=-2)
    ax_b.set_zlabel("z (Å)", fontsize=8, labelpad=-2)
    ax_b.set_title("B · mean |coupling| on Cα (A frame)", fontsize=10)
    ax_b.tick_params(labelsize=7, pad=0)
    if coords_b_aligned is not None:
        ax_b.legend(loc="upper left", fontsize=7, frameon=False)
    cbar_b = fig.colorbar(sc, ax=ax_b, fraction=0.05, pad=0.08, shrink=0.75)
    cbar_b.set_label("row-mean |r|", fontsize=8)

    ax_c.scatter(db_full, drel_full, s=max(4, 6000 / n_full), alpha=0.35, c="#9467bd", edgecolors="none")
    ax_c.axhline(0, color="0.5", lw=0.8)
    ax_c.axvline(0, color="0.5", lw=0.8)
    ax_c.set_xlabel(f"ΔB_iso ({emdb_b} − {emdb_a})", fontsize=9)
    ax_c.set_ylabel(f"Δreliability ({emdb_b} − {emdb_a})", fontsize=9)
    rho_txt = "n/a"
    if spearman_rho is not None and np.isfinite(spearman_rho):
        rho_txt = f"{spearman_rho:+.2f}"
    ax_c.set_title(f"C · ΔB vs Δreliability (ρ = {rho_txt}; n={n_full})", fontsize=10)

    subtitle = f"Conformation pair EMD-{emdb_a} vs EMD-{emdb_b} (n={n_full} in-mask Cα)"
    if coverage_note:
        subtitle += f" · {coverage_note}"
    fig.suptitle(subtitle, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


__all__ = [
    "pick_slice_index",
    "extract_slice",
    "SliceCrop",
    "slice_crop_from_mask",
    "crop_slice_2d",
    "mask_slice_values",
    "apply_contour_mask",
    "plot_local_resolution_slice",
    "plot_parallel_reliability_readouts",
    "plot_feature_family_panel",
    "plot_reliability_pair_only",
    "plot_spearman_top_bar",
    "CohortMetricRow",
    "collect_cohort_metrics",
    "write_cohort_metrics_csv",
    "plot_cohort_metrics_heatmap",
    "plot_conformation_sequence_strip",
    "plot_conformation_pair_coupling_heatmap",
    "plot_conformation_delta_joint_heatmap",
    "plot_conformation_pair_summary_triptych",
    "compute_conformation_coupling",
]
