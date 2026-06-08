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

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from .conformation_pair import (
    DOMAIN_COLORS,
    UNASSIGNED_DOMAIN_COLOR,
    compute_domain_mean_coupling,
    get_domain_assignments,
    get_domain_regions_for_pair,
    interior_residue_indices,
    reload_domain_colors,
)
from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from .structure_validation import load_cohort_manifest_row


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
    apply(ax)
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
        save_nature(fig, save_path, dpi=dpi)
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
    for letter, ax in zip("abc", axes):
        label_panel(ax, letter)
    if contour is not None:
        fig.suptitle(
            f"Parallel reliability readouts (mask ρ ≥ {contour:g}, windowed metrics)",
            fontsize=12,
            y=1.02,
        )
    _add_scale_bar(axes[0], voxel_size_a=voxel_size_a, length_a=scale_bar_a)
    fig.tight_layout()
    if save_path:
        save_nature(fig, save_path, dpi=dpi)
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
    for i, ax in enumerate(axes_flat[:n]):
        label_panel(ax, chr(ord("a") + i))
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(family_title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        save_nature(fig, save_path, dpi=dpi)
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
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.suptitle(f"Same slice Z = {slice_index}", fontsize=11)
    fig.tight_layout()
    if save_path:
        save_nature(fig, save_path, dpi=dpi)
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
    apply(ax)
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
        save_nature(fig, save_path, dpi=dpi)
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
    apply(ax)
    cmap = PALETTES["diverging"].copy()
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
        save_nature(fig, save_path, dpi=dpi)
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


def compute_conformation_coupling(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool = True,
    half_window: int | None = None,
) -> dict[str, object] | None:
    """
    Coupling matrices and per-residue Δ tracks for conformation-pair summary figures.

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


def compute_coupling_cluster_separation_score(
    corr: np.ndarray,
    *,
    k_max: int = 6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Block-structure diagnostic for adaptive conformation-pair figures.

    Average-linkage clustering is cut at k = 2 … k_max; the score is the
    maximum normalized within-minus-across |r| contrast over those cuts.
    Higher → clearer block structure in the cluster-reordered matrix.

    Typical interior-matrix values:
        TRPV1 23129/23130 ≈ 0.05 (domain layout)
        MgtA 49450/48534 ≈ 0.15 (block layout)

    Returns (score, leaf_order, linkage_matrix).
    """
    from scipy.cluster.hierarchy import fcluster

    order, z = _hierarchical_cluster(corr)
    n = int(corr.shape[0])
    if n < 10 or z.shape[0] < 2:
        return 0.0, order, z

    best = 0.0
    for k in range(2, min(k_max + 1, max(3, n // 50 + 2))):
        labels = fcluster(z, t=k, criterion="maxclust")
        same_abs: list[float] = []
        diff_abs: list[float] = []
        for i in range(n):
            for j in range(n):
                val = float(corr[i, j])
                if not np.isfinite(val):
                    continue
                mag = abs(val)
                if int(labels[i]) == int(labels[j]):
                    same_abs.append(mag)
                else:
                    diff_abs.append(mag)
        if not same_abs or not diff_abs:
            continue
        ms = float(np.mean(same_abs))
        md = float(np.mean(diff_abs))
        best = max(best, float((ms - md) / (ms + md + 1e-9)))

    return best, order, z


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


def _apply_panel_style(ax) -> None:
    """Nature styling for 2D axes."""
    apply(ax)


def _add_anchor_colorbar(
    fig,
    mappable,
    anchor_ax,
    *,
    label: str,
    pad: float = 0.012,
    width: float = 0.007,
):
    """Place a thin colorbar just right of ``anchor_ax`` (after layout is finalized)."""
    pos = anchor_ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    return cbar


def _conformation_pair_summary_data(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool,
    half_window: int | None,
) -> dict[str, object] | None:
    """Shared arrays for conformation-pair summary figures."""
    coupling = compute_conformation_coupling(
        pairs, in_mask_both=in_mask_both, half_window=half_window
    )
    if coupling is None:
        return None
    packed = _sorted_conformation_deltas(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use_full_list, db_full, drel_full, _chains = packed
    corr_i = coupling["interior_corr"]
    use_int = coupling["interior_use"]
    n_int = len(use_int)
    hw = int(coupling["half_window"])
    sep_score, cluster_order, _z_link = compute_coupling_cluster_separation_score(corr_i)
    corr_ord = corr_i[np.ix_(cluster_order, cluster_order)]
    use_all = coupling["use"]
    row_mean = np.asarray(coupling["row_mean_abs"], dtype=np.float64)
    coords_all = np.array([[a.x, a.y, a.z] for a, _ in use_all], dtype=np.float64)
    return {
        "coupling": coupling,
        "db_full": db_full,
        "drel_full": drel_full,
        "use_full": use_full_list,
        "n_full": len(use_full_list),
        "n_int": n_int,
        "hw": hw,
        "corr_ord": corr_ord,
        "corr_int": corr_i,
        "cluster_order": cluster_order,
        "cluster_separation_score": sep_score,
        "interior_use": use_int,
        "coords_all": coords_all,
        "row_mean": row_mean,
        "c_lo": float(np.nanmin(row_mean)),
        "c_hi": float(np.nanmax(row_mean)),
    }


DEFAULT_CLUSTER_SEPARATION_THRESHOLD = 0.10


def _cohort_display_name(emdb_id: str, manifest: Path | None = None) -> str:
    """Human-readable structure name from ``cohort/manifest.csv`` (``display_name`` column)."""
    eid = str(emdb_id).strip()
    path = manifest if manifest is not None else COHORT_MANIFEST
    try:
        row = load_cohort_manifest_row(path, eid)
        name = str(row.get("display_name", "")).strip()
        if name:
            return name
    except (KeyError, OSError, csv.Error):
        pass
    return f"EMD-{eid}"


def select_conformation_pair_figure_layout(
    separation_score: float,
    *,
    threshold: float = DEFAULT_CLUSTER_SEPARATION_THRESHOLD,
    layout: str = "auto",
) -> str:
    """
    Diagnostic label from coupling block-structure score (legacy / stats only).

    Main figures always use the cluster-reordered matrix in panel a regardless of
    this recommendation. ``block`` = score ≥ threshold; ``domain`` = diffuse coupling.
    """
    if layout == "block":
        return "block"
    if layout == "domain":
        return "domain"
    if separation_score >= threshold:
        return "block"
    return "domain"


def _draw_conformation_domain_coupling_heatmap(
    ax,
    *,
    corr: np.ndarray,
    assignments: dict[str, list[int]],
    domain_order: Sequence[str],
    metric: str = "mean_abs",
    abs_threshold: float = 0.5,
    panel_letter: str | None = None,
):
    """Domain×domain mean |coupling| heatmap (supplementary figure). Returns mappable or None."""
    import pandas as pd
    import seaborn as sns

    domain_mat, names = compute_domain_mean_coupling(
        corr,
        assignments,
        domain_order=domain_order,
        metric=metric,
        abs_threshold=abs_threshold,
    )
    if not names:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Domain summary\nnot available\nfor this pair",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="0.45",
        )
        if panel_letter:
            label_panel(ax, panel_letter)
        return None

    df = pd.DataFrame(domain_mat, index=names, columns=names)
    if metric == "frac_strong":
        fmt = ".0%"
        title = f"Domain |coupling| > {abs_threshold:.1f}"
        cbar_label = f"frac |r| > {abs_threshold:.1f}"
        cmap = PALETTES["sequential"]
    else:
        fmt = ".2f"
        title = "Domain mean |coupling|"
        cbar_label = "mean |r|"
        cmap = "YlOrRd"

    hm = sns.heatmap(
        df,
        ax=ax,
        annot=True,
        fmt=fmt,
        annot_kws={"size": 6},
        square=True,
        cbar=False,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_panel_style(ax)
    if panel_letter:
        label_panel(ax, panel_letter)
    mappable = hm.collections[0] if hm.collections else None
    return (mappable, cbar_label) if mappable is not None else None


def _domain_scatter_colors(
    use: Sequence[tuple[object, object]],
    regions: Sequence[object],
) -> list[str]:
    """Per-residue hex colors for panel f from domain assignments."""
    colors: list[str] = []
    for row, _ in use:
        color = UNASSIGNED_DOMAIN_COLOR
        seq_num = int(row.seq_num)
        for reg in regions:
            if reg.seq_start <= seq_num <= reg.seq_end:
                color = DOMAIN_COLORS.get(reg.name, reg.color)
                break
        colors.append(color)
    return colors


def _domain_name_per_residue(
    use: Sequence[tuple[object, object]],
    regions: Sequence[object],
) -> list[str | None]:
    """Domain name for each residue in chain order (None if unassigned)."""
    names: list[str | None] = []
    for row, _ in use:
        domain: str | None = None
        seq_num = int(row.seq_num)
        for reg in regions:
            if reg.seq_start <= seq_num <= reg.seq_end:
                domain = reg.name
                break
        names.append(domain)
    return names


def _contiguous_domain_stretches(
    domain_names: Sequence[str | None],
) -> list[tuple[int, int, str | None]]:
    """Inclusive start, exclusive end, domain label for each contiguous run."""
    if not domain_names:
        return []
    stretches: list[tuple[int, int, str | None]] = []
    start = 0
    current = domain_names[0]
    for i in range(1, len(domain_names)):
        if domain_names[i] != current:
            stretches.append((start, i, current))
            start = i
            current = domain_names[i]
    stretches.append((start, len(domain_names), current))
    return stretches


def _per_domain_spearman_stats(
    db: np.ndarray,
    drel: np.ndarray,
    assignments: dict[str, list[int]],
    domain_order: Sequence[str],
) -> list[tuple[str, float, int, str]]:
    """Per-domain Spearman ρ(ΔB, Δrel): (name, rho, n, color)."""
    from scipy.stats import spearmanr

    stats: list[tuple[str, float, int, str]] = []
    for name in domain_order:
        idx = assignments.get(name, [])
        color = DOMAIN_COLORS.get(name, UNASSIGNED_DOMAIN_COLOR)
        if len(idx) < 3:
            stats.append((name, float("nan"), len(idx), color))
            continue
        sub_db = np.asarray(db, dtype=np.float64)[idx]
        sub_drel = np.asarray(drel, dtype=np.float64)[idx]
        ok = np.isfinite(sub_db) & np.isfinite(sub_drel)
        n_ok = int(ok.sum())
        if n_ok < 3:
            stats.append((name, float("nan"), n_ok, color))
            continue
        rho, _ = spearmanr(sub_db[ok], sub_drel[ok])
        stats.append((name, float(rho) if np.isfinite(rho) else float("nan"), n_ok, color))
    return stats


def _draw_conformation_delta_reliability_profile(
    ax,
    *,
    drel_full: np.ndarray,
    row_mean: np.ndarray,
    use_full: Sequence[tuple[object, object]],
    regions: Sequence[object] | None,
    domain_order: Sequence[str] | None,
    panel_letter: str | None = "c",
) -> None:
    """Per-residue Δreliability along chain order, colored by domain, with |coupling| overlay."""
    drel = np.asarray(drel_full, dtype=np.float64)
    coupling_row = np.asarray(row_mean, dtype=np.float64)
    n = int(drel.size)
    assert coupling_row.size == n == len(use_full), (
        "Δreliability and row-mean |coupling| must align to use_full residue order"
    )

    _apply_panel_style(ax)
    x = np.arange(n, dtype=np.float64)

    if regions:
        domain_names = _domain_name_per_residue(use_full, regions)
    else:
        domain_names = [None] * n

    for start, end, domain in _contiguous_domain_stretches(domain_names):
        xs = x[start:end]
        ys = drel[start:end]
        color = (
            DOMAIN_COLORS.get(domain, UNASSIGNED_DOMAIN_COLOR)
            if domain
            else UNASSIGNED_DOMAIN_COLOR
        )
        ax.fill_between(xs, 0.0, ys, where=ys < 0, color=color, alpha=0.4, linewidth=0)
        ax.fill_between(xs, 0.0, ys, where=ys >= 0, color=color, alpha=0.4, linewidth=0)

    for i in range(1, n):
        if domain_names[i] != domain_names[i - 1]:
            ax.axvline(i, color="#cccccc", ls="--", lw=0.75, zorder=0)

    finite_drel = np.isfinite(drel)
    max_abs = float(np.percentile(np.abs(drel[finite_drel]), 98)) if finite_drel.any() else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    y_lo, y_hi = -max_abs * 1.1, max_abs * 1.1
    ax.set_ylim(y_lo, y_hi)

    if regions and domain_order:
        for name in domain_order:
            idx = [i for i, d in enumerate(domain_names) if d == name]
            if not idx:
                continue
            mid_x = 0.5 * (min(idx) + max(idx))
            ax.text(
                mid_x,
                y_lo * 0.92,
                name,
                ha="center",
                va="bottom",
                fontsize=6,
                color=DOMAIN_COLORS.get(name, UNASSIGNED_DOMAIN_COLOR),
            )

    ax.axhline(0.0, color="#999999", lw=0.5, zorder=1)

    ax2 = ax.twinx()
    ax2.plot(x, coupling_row, color="#333333", lw=0.75, alpha=0.5)
    ax2.set_ylabel("mean |coupling|", fontsize=6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.5)
    ax2.tick_params(axis="y", which="major", labelsize=6, width=0.5, length=2)

    x0, x1 = 0.0, float(max(n - 1, 0))
    x_pad = 0.01 * max(1.0, x1 - x0)
    ax.set_xlim(x0 - x_pad, x1 + x_pad)
    ax.set_xlabel("Residue index (chain order)", fontsize=7)
    ax.set_ylabel("Δreliability", fontsize=7)
    ax.set_title("Per-residue Δreliability · colored by domain", fontsize=7)
    if panel_letter is not None:
        label_panel(ax, panel_letter)


def _draw_conformation_clustered_coupling_panel(
    ax,
    *,
    corr: np.ndarray,
    order: np.ndarray,
    hw: int,
    separation_score: float | None = None,
    panel_letter: str = "a",
):
    """Compact cluster-reordered coupling matrix for adaptive block-layout figures."""
    corr_ord = corr[np.ix_(order, order)]
    n = int(corr_ord.shape[0])
    im = ax.imshow(
        corr_ord,
        origin="lower",
        aspect="equal",
        cmap=PALETTES["diverging"],
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    title = f"Cluster-reordered coupling (±{hw})"
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("Residue index (cluster order)", fontsize=7)
    ax.set_ylabel("Residue index (cluster order)", fontsize=7)
    tick_n = min(5, n)
    tick_idx = np.linspace(0, n - 1, num=tick_n, dtype=int)
    ax.set_xticks(tick_idx, [str(int(i)) for i in tick_idx], fontsize=5)
    ax.set_yticks(tick_idx, [str(int(i)) for i in tick_idx], fontsize=5)
    _apply_panel_style(ax)
    label_panel(ax, panel_letter)
    return im


def _draw_conformation_summary_scatter_panel(
    ax,
    *,
    db_full: np.ndarray,
    drel_full: np.ndarray,
    n_full: int,
    emdb_a: str,
    emdb_b: str,
    spearman_rho: float | None,
    use_full: Sequence[tuple[object, object]] | None = None,
    regions: Sequence[object] | None = None,
    domain_order: Sequence[str] | None = None,
    panel_letter: str = "b",
) -> None:
    _apply_panel_style(ax)
    point_colors: str | list[str] = UNASSIGNED_DOMAIN_COLOR
    if use_full and regions:
        point_colors = _domain_scatter_colors(use_full, regions)
    ax.scatter(db_full, drel_full, s=8, alpha=0.6, c=point_colors, edgecolors="none")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    x = np.asarray(db_full, dtype=np.float64)
    y = np.asarray(drel_full, dtype=np.float64)
    domain_rho_map: dict[str, float] = {}
    if use_full and regions and domain_order:
        assignments = get_domain_assignments(use_full, regions)
        domain_rho_map = {
            name: rho
            for name, rho, _n, _color in _per_domain_spearman_stats(
                db_full, drel_full, assignments, domain_order
            )
        }
        for name in domain_order:
            idx = assignments.get(name, [])
            if len(idx) < 2:
                continue
            rho = domain_rho_map.get(name, float("nan"))
            if not np.isfinite(rho) or rho > -0.2:
                continue
            sub_x = x[idx]
            sub_y = y[idx]
            ok = np.isfinite(sub_x) & np.isfinite(sub_y)
            if ok.sum() < 2:
                continue
            coeffs = np.polyfit(sub_x[ok], sub_y[ok], 1)
            xline = np.linspace(float(np.nanmin(sub_x[ok])), float(np.nanmax(sub_x[ok])), 100)
            ls = "-" if rho <= -0.4 else "--"
            ax.plot(
                xline,
                np.polyval(coeffs, xline),
                ls,
                color=DOMAIN_COLORS.get(name, UNASSIGNED_DOMAIN_COLOR),
                lw=1.0,
                alpha=0.85,
                zorder=2,
            )
    ax.set_xlabel(f"ΔB_iso ({emdb_b} − {emdb_a})", fontsize=7)
    ax.set_ylabel(f"Δreliability ({emdb_b} − {emdb_a})", fontsize=7)
    ax.set_title("ΔB vs Δreliability", fontsize=7)
    rho_txt = "n/a"
    if spearman_rho is not None and np.isfinite(spearman_rho):
        rho_txt = f"{spearman_rho:.2f}"
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho_txt}\nn = {n_full}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
    )
    if use_full and regions and domain_order:
        from matplotlib.patches import Patch

        assignments = get_domain_assignments(use_full, regions)
        legend_patches = [
            Patch(facecolor=DOMAIN_COLORS.get(name, UNASSIGNED_DOMAIN_COLOR), label=name)
            for name in domain_order
            if assignments.get(name)
        ]
        if legend_patches:
            ax.legend(
                handles=legend_patches,
                loc="upper right",
                fontsize=6,
                frameon=False,
                handlelength=1.0,
                borderaxespad=0.3,
            )
    label_panel(ax, panel_letter)


def plot_conformation_pair_domain_coupling_supplement(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    half_window: int | None = None,
    coverage_note: str | None = None,
    manifest: Path | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Supplementary single-panel domain mean |coupling| heatmap."""
    data = _conformation_pair_summary_data(
        pairs, in_mask_both=in_mask_both, half_window=half_window
    )
    if data is None:
        return None

    reload_domain_colors()
    regions = get_domain_regions_for_pair(emdb_a, emdb_b)
    if not regions:
        return None

    domain_order = [reg.name for reg in regions]
    assignments = get_domain_assignments(data["use_full"], regions)
    corr_full = np.asarray(data["coupling"]["corr"], dtype=np.float64)
    n_full = int(data["n_full"])

    fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor="white")
    result = _draw_conformation_domain_coupling_heatmap(
        ax,
        corr=corr_full,
        assignments=assignments,
        domain_order=domain_order,
        metric="mean_abs",
    )

    coverage_str = coverage_note if coverage_note else ""
    name_a = _cohort_display_name(emdb_a, manifest)
    name_b = _cohort_display_name(emdb_b, manifest)
    fig.suptitle(
        f"{name_a} vs {name_b} · domain mean |coupling|",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        f"EMD-{emdb_a} vs EMD-{emdb_b} · n = {n_full} in-mask Cα · coverage {coverage_str}",
        fontsize=7,
        ha="center",
        color="#444444",
    )

    fig.canvas.draw()
    if result is not None:
        mappable, cbar_label = result
        _add_anchor_colorbar(fig, mappable, ax, label=cbar_label, pad=0.04, width=0.025)

    if save_path:
        save_nature(fig, save_path, dpi=dpi)
    return fig


def plot_conformation_pair_summary_triptych(
    pairs: Sequence[tuple[object, object]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
    half_window: int | None = None,
    spearman_rho: float | None = None,
    spearman_rho_h: float | None = None,
    coverage_note: str | None = None,
    coords_b_aligned: np.ndarray | None = None,
    cluster_separation_threshold: float = DEFAULT_CLUSTER_SEPARATION_THRESHOLD,
    layout: str = "auto",
    manifest: Path | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure | None, str]:
    """
    Conformation-pair summary triptych (14×10 in, three panels).

    Panel a is always the cluster-reordered coupling matrix (Pearson r colorbar).
    Panels b–c: Δ scatter and full-width per-residue Δreliability profile.

    Returns ``(figure, recommended_layout)`` where ``recommended_layout`` is the
    legacy block/domain diagnostic from the cluster separation score (stats only).
    """
    del spearman_rho_h, coords_b_aligned
    data = _conformation_pair_summary_data(
        pairs, in_mask_both=in_mask_both, half_window=half_window
    )
    if data is None:
        return None, "domain"

    reload_domain_colors()

    n_full = int(data["n_full"])
    hw = int(data["hw"])
    sep_score = float(data["cluster_separation_score"])
    cluster_order = data["cluster_order"]
    use_full = data["use_full"]
    db_full = data["db_full"]
    drel_full = data["drel_full"]
    row_mean = data["row_mean"]
    recommended_layout = select_conformation_pair_figure_layout(
        sep_score, threshold=cluster_separation_threshold, layout=layout
    )

    regions = get_domain_regions_for_pair(emdb_a, emdb_b)
    domain_order = [reg.name for reg in regions]

    coverage_str = coverage_note if coverage_note else ""
    name_a = _cohort_display_name(emdb_a, manifest)
    name_b = _cohort_display_name(emdb_b, manifest)

    fig = plt.figure(figsize=(14.0, 10.0), facecolor="white")
    # Dedicated header band — keep suptitle/subtitle well above panel labels (y≈1.05).
    fig.text(
        0.5,
        0.975,
        f"{name_a} vs {name_b} · conformation pair summary",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.948,
        f"EMD-{emdb_a} vs EMD-{emdb_b} · n = {n_full} in-mask Cα · coverage {coverage_str}",
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.45, 0.55],
        width_ratios=[0.45, 0.55],
        hspace=0.38,
        wspace=0.28,
        left=0.08,
        right=0.92,
        top=0.82,
        bottom=0.10,
    )

    im_cluster = None
    ax_cluster = fig.add_subplot(gs[0, 0])
    im_cluster = _draw_conformation_clustered_coupling_panel(
        ax_cluster,
        corr=data["corr_int"],
        order=cluster_order,
        hw=hw,
        separation_score=sep_score,
        panel_letter="a",
    )

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_conformation_summary_scatter_panel(
        ax_b,
        db_full=db_full,
        drel_full=drel_full,
        n_full=n_full,
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        spearman_rho=spearman_rho,
        use_full=use_full,
        regions=regions or None,
        domain_order=domain_order or None,
        panel_letter="b",
    )

    ax_c = fig.add_subplot(gs[1, :])
    _draw_conformation_delta_reliability_profile(
        ax_c,
        drel_full=drel_full,
        row_mean=row_mean,
        use_full=use_full,
        regions=regions or None,
        domain_order=domain_order or None,
        panel_letter=None,
    )

    fig.canvas.draw()
    if im_cluster is not None and ax_cluster is not None:
        cbar = fig.colorbar(im_cluster, ax=ax_cluster, fraction=0.046, pad=0.06)
        cbar.set_label("Pearson r", fontsize=7)
        cbar.ax.tick_params(labelsize=6, length=2)

    fig.canvas.draw()
    pos_a = ax_cluster.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    full_width = pos_b.x1 - pos_a.x0
    ax_c.set_position([pos_a.x0, pos_c.y0, full_width, pos_c.height])
    label_panel(ax_c, "c", x=-0.1 * pos_a.width / full_width)

    if save_path:
        save_nature(fig, save_path, dpi=dpi)
    return fig, recommended_layout


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
    "plot_conformation_pair_summary_triptych",
    "plot_conformation_pair_domain_coupling_supplement",
    "compute_conformation_coupling",
    "compute_coupling_cluster_separation_score",
    "select_conformation_pair_figure_layout",
    "DEFAULT_CLUSTER_SEPARATION_THRESHOLD",
]
