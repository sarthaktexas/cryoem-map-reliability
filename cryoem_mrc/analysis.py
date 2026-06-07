"""Stats-vs-half-map-agreement analysis layer.

This module turns the per-voxel feature maps produced by :mod:`cryoem_mrc.pipeline`
into the comparison tables and figures called for in the project handoff §4 and §7:

- A masked, per-voxel correlation table (Pearson + Spearman) of every density-derived
  feature against a chosen reliability target — typically ``local_cross_correlation``
  from :func:`cryoem_mrc.half_map_repro.half_map_local_metrics`, optionally also
  against a local-resolution map in Å when one is available.
- A tidy CSV for the thesis appendix.
- A plain-text ``summary.txt`` listing the strongest correlated features with the
  scientific caveats that they are map-derived rigidity proxies, not biophysical
  flexibility measurements.
- Histograms of the half-map metrics inside vs. outside the analysis mask, plus
  feature-vs-target scatter and binned-mean curves for visual inspection.

Scientific framing
------------------

The thesis question is whether voxel-wise local statistics computed on a cryo-EM
density map identify regions of high vs. low map reliability. ``local_cross_correlation``
(half-map agreement in a sliding cubic window) is the primary reliability signal.
An Å-valued local resolution target (windowed FSC via :mod:`cryoem_mrc.local_fsc`,
loaded through :mod:`cryoem_mrc.local_resolution_io`) can be used as a second
target column — see :class:`MaskedAnalysisResult`.

Performance notes
-----------------

For large boxes (~10⁸ voxels) the chunked variant
:func:`half_map_local_metrics_chunked` keeps peak memory bounded by processing
Z-slabs with overlap; outputs are bit-equivalent to the unchunked routine within
floating-point noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import csv
import json

import numpy as np
from scipy import ndimage, stats

from .half_map_repro import half_map_local_metrics


# ---------------------------------------------------------------------------
# Half-map metrics: chunked wrapper for large volumes
# ---------------------------------------------------------------------------


def half_map_local_metrics_chunked(
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    window: int = 5,
    chunk_z: int = 64,
    eps: float = 1e-12,
    out_dtype: type[np.float32] | type[np.float64] = np.float32,
) -> dict[str, np.ndarray]:
    """
    Z-chunked wrapper around :func:`cryoem_mrc.half_map_repro.half_map_local_metrics`.

    Identical math; bounded peak memory. Each Z-slab is padded by ``window // 2``
    voxels on each side so the central output region sees the same uniform-filter
    neighborhoods it would in a full-volume pass — outputs match the unchunked
    routine within float-rounding noise.

    Parameters
    ----------
    half1, half2
        Two cryo-EM half-maps with identical shape ``(Z, Y, X)`` on the same grid.
    window
        Cubic uniform window side (odd recommended); same semantics as
        :func:`half_map_local_metrics`.
    chunk_z
        Number of Z-slices processed per slab. ~64 is a good balance for 400³–500³
        boxes on a 16 GB workstation.
    out_dtype
        Output dtype for storage; default float32 cuts MRC output size in half
        relative to float64 with negligible precision loss for these metrics.

    Returns
    -------
    dict
        Same keys as :func:`half_map_local_metrics`:
        ``local_cross_correlation``, ``local_mean_squared_difference``,
        ``local_variance_difference``, ``local_reproducibility_snr``.
    """
    if half1.shape != half2.shape:
        raise ValueError(f"Shape mismatch: {half1.shape} vs {half2.shape}")
    if half1.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got shape {half1.shape}")
    if window < 1:
        raise ValueError("window must be positive")

    nz = int(half1.shape[0])
    shape = half1.shape
    pad = int(window // 2)

    keys = (
        "local_cross_correlation",
        "local_mean_squared_difference",
        "local_variance_difference",
        "local_reproducibility_snr",
    )
    out: dict[str, np.ndarray] = {k: np.empty(shape, dtype=out_dtype) for k in keys}

    z0 = 0
    while z0 < nz:
        z1 = min(nz, z0 + int(chunk_z))
        za = max(0, z0 - pad)
        zb = min(nz, z1 + pad)
        z_off = z0 - za
        take = z1 - z0
        m = half_map_local_metrics(half1[za:zb], half2[za:zb], window=window, eps=eps)
        for k in keys:
            out[k][z0:z1] = m[k][z_off : z_off + take].astype(out_dtype, copy=False)
        z0 = z1

    return out


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def build_contour_mask(
    density: np.ndarray,
    contour: float,
    *,
    erode_voxels: int = 0,
) -> np.ndarray:
    """
    Boolean (Z, Y, X) mask of voxels with density at or above the contour level.

    ``contour`` should be in the same intensity units as ``density`` (typically the
    raw or threshold-aware MRC intensity, **not** a z-scored normalization). For
    EMD-49450 the deposited recommended contour is 0.116 — see DECISIONS Decision
    002 for the discussion.

    Optional ``erode_voxels`` shrinks the mask by morphological erosion so that
    boundary voxels (where uniform-filter neighborhoods straddle the protein /
    solvent edge) are excluded from the analysis. Useful when the analysis target
    itself was computed with a sliding window of similar size.
    """
    d = np.asarray(density)
    if d.ndim != 3:
        raise ValueError(f"Expected 3D density, got shape {d.shape}")
    mask = d >= float(contour)
    if erode_voxels > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_erosion(mask, structure=struct, iterations=int(erode_voxels))
    return mask


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


CorrelationMethod = Literal["pearson", "spearman"]


@dataclass
class FeatureCorrelation:
    """Correlation summary for one feature against one target signal."""

    feature_name: str
    target_name: str
    method: CorrelationMethod
    n_samples: int
    correlation: float
    p_value: float


@dataclass
class MaskedAnalysisResult:
    """Container for a complete masked analysis run; serializable to CSV / text."""

    contour: float | None
    n_total_voxels: int
    n_masked_voxels: int
    target_name: str
    correlations: list[FeatureCorrelation]
    extra_metadata: dict[str, str] = field(default_factory=dict)


def _flatten_under_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if volume.shape != mask.shape:
        raise ValueError(f"Volume shape {volume.shape} != mask shape {mask.shape}")
    return np.asarray(volume)[mask]


def _safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: CorrelationMethod,
) -> tuple[float, float]:
    """Pearson or Spearman; drops non-finite pairs before computing."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan"), float("nan")
    xf = x[finite]
    yf = y[finite]
    # If either side is constant the correlation is undefined; return NaN rather
    # than raising so the row appears in the CSV with a clear flag.
    if xf.std() == 0 or yf.std() == 0:
        return float("nan"), float("nan")
    if method == "pearson":
        r = stats.pearsonr(xf, yf)
        return float(r.statistic), float(r.pvalue)
    if method == "spearman":
        r = stats.spearmanr(xf, yf)
        return float(r.statistic), float(r.pvalue)
    raise ValueError(f"Unknown method: {method!r}")


def compute_feature_target_correlations(
    features: Mapping[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    *,
    target_name: str = "local_cross_correlation",
    methods: Sequence[CorrelationMethod] = ("pearson", "spearman"),
    feature_keys: Sequence[str] | None = None,
    max_samples: int | None = 2_000_000,
    rng_seed: int = 0,
) -> MaskedAnalysisResult:
    """
    For every (3D, same-shape-as-target) feature in ``features``, compute correlation
    against ``target`` over voxels selected by ``mask``.

    Both Pearson (linear) and Spearman (rank) are reported by default; Spearman is
    more robust to monotonic-but-nonlinear feature/target relationships, which are
    common in cryo-EM density statistics. Subsamples down to ``max_samples`` voxels
    when the masked set is larger; pass ``max_samples=None`` to force a full-data
    pass (slower for very large masks, no statistical benefit beyond ~10⁶ samples
    for these effect sizes).

    Returns a :class:`MaskedAnalysisResult` ready for CSV / summary export.
    """
    target = np.asarray(target)
    mask_b = np.asarray(mask).astype(bool)
    if target.shape != mask_b.shape:
        raise ValueError(f"target shape {target.shape} != mask shape {mask_b.shape}")
    n_total = int(mask_b.size)
    n_masked = int(mask_b.sum())
    if n_masked < 3:
        raise ValueError(f"Mask has only {n_masked} voxels; need >= 3 for correlation.")

    target_flat = target[mask_b]
    if max_samples is not None and target_flat.size > max_samples:
        rng = np.random.default_rng(rng_seed)
        sample_idx = rng.choice(target_flat.size, size=int(max_samples), replace=False)
        target_flat = target_flat[sample_idx]
    else:
        sample_idx = None

    keys = list(feature_keys) if feature_keys is not None else list(features.keys())

    correlations: list[FeatureCorrelation] = []
    for key in keys:
        arr = features[key]
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape != mask_b.shape:
            # Non-3D or mis-shaped (e.g. multiscale_sigmas); skip silently.
            continue
        feat_flat = a[mask_b]
        if sample_idx is not None:
            feat_flat = feat_flat[sample_idx]
        for method in methods:
            r, p = _safe_correlation(feat_flat, target_flat, method)
            correlations.append(
                FeatureCorrelation(
                    feature_name=key,
                    target_name=target_name,
                    method=method,
                    n_samples=int(target_flat.size),
                    correlation=r,
                    p_value=p,
                )
            )

    return MaskedAnalysisResult(
        contour=None,
        n_total_voxels=n_total,
        n_masked_voxels=n_masked,
        target_name=target_name,
        correlations=correlations,
    )


# ---------------------------------------------------------------------------
# Binned analysis (handoff §4: "binned analysis")
# ---------------------------------------------------------------------------


@dataclass
class BinnedRelationship:
    """Mean / std / count of one feature in equal-count bins of a target signal."""

    feature_name: str
    target_name: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    mean_feature: np.ndarray
    std_feature: np.ndarray
    count: np.ndarray


def binned_feature_by_target(
    feature: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    feature_name: str,
    target_name: str = "local_cross_correlation",
    n_bins: int = 10,
    quantile_bins: bool = True,
) -> BinnedRelationship:
    """
    Bin masked voxels by ``target`` (quantile-bins by default for balanced counts),
    return mean / std of ``feature`` per bin. Handoff §4 binned-analysis primitive.
    """
    f = _flatten_under_mask(feature, mask)
    t = _flatten_under_mask(target, mask)
    finite = np.isfinite(f) & np.isfinite(t)
    f = f[finite]
    t = t[finite]
    if quantile_bins:
        edges = np.quantile(t, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(t.min(), t.max(), n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(t, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    means = np.zeros(n_bins, dtype=np.float64)
    stds = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = f[bin_idx == b]
        counts[b] = sel.size
        if sel.size > 0:
            means[b] = float(sel.mean())
            stds[b] = float(sel.std())
        else:
            means[b] = np.nan
            stds[b] = np.nan
    centers = 0.5 * (np.where(np.isfinite(edges[:-1]), edges[:-1], edges[1] - 1.0)
                     + np.where(np.isfinite(edges[1:]), edges[1:], edges[-2] + 1.0))
    return BinnedRelationship(
        feature_name=feature_name,
        target_name=target_name,
        bin_edges=edges,
        bin_centers=centers,
        mean_feature=means,
        std_feature=stds,
        count=counts,
    )


# ---------------------------------------------------------------------------
# Output writers (CSV, summary text, JSON metadata)
# ---------------------------------------------------------------------------


def write_correlation_csv(result: MaskedAnalysisResult, path: str | Path) -> Path:
    """
    Tidy CSV: one row per (feature, method). Columns:
    feature, target, method, n_samples, correlation, p_value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "target", "method", "n_samples", "correlation", "p_value"])
        for c in result.correlations:
            w.writerow([
                c.feature_name,
                c.target_name,
                c.method,
                c.n_samples,
                f"{c.correlation:.6f}",
                f"{c.p_value:.6e}",
            ])
    return path


def write_summary_text(
    result: MaskedAnalysisResult,
    path: str | Path,
    *,
    top_n: int = 10,
    method_for_ranking: CorrelationMethod = "spearman",
) -> Path:
    """
    Plain-text summary suitable for ``outputs/reports/summary.txt`` (handoff §7).

    Lists the top-``top_n`` features by ``|correlation|`` under the chosen method,
    plus mask coverage and the standard scientific caveats. Wording is intentionally
    conservative: "rigid-like / flexible-like / map-derived rigidity proxies".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c for c in result.correlations if c.method == method_for_ranking]
    rows.sort(key=lambda c: abs(c.correlation) if np.isfinite(c.correlation) else -1.0, reverse=True)

    lines: list[str] = []
    lines.append("Cryo-EM local-density analysis — summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Target signal:           {result.target_name}")
    if result.contour is not None:
        lines.append(f"Mask contour level:      {result.contour:.6f}")
    lines.append(f"Total voxels in box:     {result.n_total_voxels:,}")
    lines.append(f"Voxels in analysis mask: {result.n_masked_voxels:,} "
                 f"({100.0 * result.n_masked_voxels / max(1, result.n_total_voxels):.2f} %)")
    if rows:
        lines.append(f"Samples per correlation: {rows[0].n_samples:,}")
    lines.append("")
    lines.append(f"Top {min(top_n, len(rows))} features by |{method_for_ranking}| correlation:")
    lines.append("-" * 50)
    lines.append(f"{'feature':<40s} {'r':>8s} {'p':>10s}")
    for c in rows[:top_n]:
        lines.append(f"{c.feature_name:<40s} {c.correlation:>+8.4f} {c.p_value:>10.2e}")
    lines.append("")
    lines.append("Scientific caveats")
    lines.append("-" * 50)
    lines.append(
        "These correlations relate map-derived voxel statistics to half-map\n"
        "reproducibility (a reliability proxy), not to biophysical flexibility.\n"
        "High agreement with the target signal indicates that the feature tracks\n"
        "regions of the map that two independent halves agree on; conservative\n"
        "interpretation is 'rigid-like' for high agreement and 'flexible-like'\n"
        "for low agreement, with no claim about absolute molecular dynamics."
    )
    if result.extra_metadata:
        lines.append("")
        lines.append("Run metadata")
        lines.append("-" * 50)
        lines.append(json.dumps(result.extra_metadata, indent=2))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figures (matplotlib)
# ---------------------------------------------------------------------------


def plot_halfmap_metric_histogram(
    metrics: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    save_path: str | Path,
    bins: int = 80,
    metric_keys: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """
    Histogram each half-map metric inside vs. outside the analysis mask on the
    same axes per metric. Demonstrates that the contour mask separates a
    bimodal distribution (solvent vs. protein density) cleanly.

    Saves a PNG; does not show interactively.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(metric_keys) if metric_keys is not None else list(metrics.keys())
    n = len(keys)
    if n == 0:
        raise ValueError("No metric keys to plot.")
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows), squeeze=False)
    flat = axes.ravel()

    mask_b = np.asarray(mask).astype(bool)
    for i, k in enumerate(keys):
        ax = flat[i]
        v = np.asarray(metrics[k])
        inside = v[mask_b]
        outside = v[~mask_b]
        # Subsample for plotting speed; bin counts are insensitive at this size.
        rng = np.random.default_rng(0)
        if inside.size > 200_000:
            inside = inside[rng.choice(inside.size, 200_000, replace=False)]
        if outside.size > 200_000:
            outside = outside[rng.choice(outside.size, 200_000, replace=False)]
        inside = inside[np.isfinite(inside)]
        outside = outside[np.isfinite(outside)]
        ax.hist(outside, bins=bins, density=True, alpha=0.5, color="lightgray",
                label=f"outside mask (n≈{outside.size:,})")
        ax.hist(inside, bins=bins, density=True, alpha=0.7, color="steelblue",
                label=f"inside mask (n≈{inside.size:,})")
        ax.set_title(k)
        ax.set_xlabel(k)
        ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="best")
    for j in range(n, len(flat)):
        flat[j].set_visible(False)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_vs_target_scatter(
    feature: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    feature_name: str,
    target_name: str,
    save_path: str | Path,
    max_points: int = 50_000,
    binned: BinnedRelationship | None = None,
) -> Path:
    """
    Hexbin (or scatter) of feature vs target inside the mask, with the binned
    mean-of-feature curve overlaid. Subsamples to ``max_points`` for the scatter
    layer; correlation calculations should use the full mask elsewhere.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = _flatten_under_mask(feature, mask)
    t = _flatten_under_mask(target, mask)
    finite = np.isfinite(f) & np.isfinite(t)
    f = f[finite]
    t = t[finite]
    if f.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(f.size, size=max_points, replace=False)
        f_plot, t_plot = f[idx], t[idx]
    else:
        f_plot, t_plot = f, t

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    hb = ax.hexbin(t_plot, f_plot, gridsize=60, mincnt=1, cmap="viridis", bins="log")
    fig.colorbar(hb, ax=ax, label="log(count)")
    if binned is not None:
        # Overlay binned-mean curve with std error bars.
        ax.errorbar(
            binned.bin_centers,
            binned.mean_feature,
            yerr=binned.std_feature,
            fmt="o-",
            color="orange",
            ecolor="orange",
            elinewidth=1.0,
            capsize=2.5,
            label="binned mean ± std",
        )
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel(target_name)
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name} vs {target_name}  (n={f.size:,} masked voxels)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


__all__ = [
    "BinnedRelationship",
    "FeatureCorrelation",
    "MaskedAnalysisResult",
    "binned_feature_by_target",
    "build_contour_mask",
    "compute_feature_target_correlations",
    "half_map_local_metrics_chunked",
    "plot_feature_vs_target_scatter",
    "plot_halfmap_metric_histogram",
    "write_correlation_csv",
    "write_summary_text",
]
