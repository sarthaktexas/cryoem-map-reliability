"""Cryo-EM MRC density map feature extraction pipeline."""

from .analysis import (
    BinnedRelationship,
    FeatureCorrelation,
    MaskedAnalysisResult,
    binned_feature_by_target,
    build_contour_mask,
    compute_feature_target_correlations,
    half_map_local_metrics_chunked,
    plot_feature_vs_target_scatter,
    plot_halfmap_metric_histogram,
    write_correlation_csv,
    write_summary_text,
)
from .half_map_repro import (
    half_map_local_metrics,
    plot_half_map_metric_distributions,
    save_half_map_metrics_mrc,
)
from .io import load_mrc, save_rigidity_mrc, save_volume_like_reference
from .local_fsc import compute_local_fsc_resolution, save_local_fsc_resolution_mrc
from .local_stats import sliding_local_statistics_pipeline
from .local_resolution_io import (
    LocalResolutionGridReport,
    LocalResolutionSource,
    build_dataset_from_pipeline,
    export_masked_feature_dataset,
    load_local_resolution_map,
    resample_local_resolution_onto_reference,
    verify_local_resolution_matches_reference,
)
from .map_grid import (
    FullHalfMapBundle,
    GridAlignmentReport,
    MapGrid,
    ensure_same_grid,
    load_full_and_half_maps,
    load_map_grid,
    resample_volume_onto_grid,
    verify_grid_alignment,
    verify_same_grid_as_reference,
)
from .multiscale import group_multiscale_features
from .pipeline import (
    load_feature_maps,
    run_pipeline,
    save_feature_maps,
    save_feature_maps_npy,
)
from .mechanics import (
    compute_mechanics_headlines,
    compute_mechanics_maps,
    fluctuation_constraint_decomposition,
)
from .reliability import (
    attach_reliability_to_features,
    classify_build_zones,
    compute_reliability_maps,
    save_build_zone_mrc,
    save_reliability_mrc,
)
from .rigidity import compute_rigidity_map
from .visualize import (
    plot_central_orthogonal_slices,
    plot_feature_slices,
    plot_rigidity_inspection,
    plot_volume_histogram,
    rigidity_inspection_keys,
)

__all__ = [
    "load_mrc",
    "save_volume_like_reference",
    "save_rigidity_mrc",
    "run_pipeline",
    "save_feature_maps",
    "save_feature_maps_npy",
    "load_feature_maps",
    "group_multiscale_features",
    "compute_mechanics_headlines",
    "compute_mechanics_maps",
    "fluctuation_constraint_decomposition",
    "attach_reliability_to_features",
    "classify_build_zones",
    "compute_reliability_maps",
    "save_build_zone_mrc",
    "save_reliability_mrc",
    "compute_rigidity_map",
    "plot_feature_slices",
    "plot_rigidity_inspection",
    "rigidity_inspection_keys",
    "compute_local_fsc_resolution",
    "save_local_fsc_resolution_mrc",
    "LocalResolutionSource",
    "LocalResolutionGridReport",
    "load_local_resolution_map",
    "resample_local_resolution_onto_reference",
    "verify_local_resolution_matches_reference",
    "export_masked_feature_dataset",
    "build_dataset_from_pipeline",
    "MapGrid",
    "GridAlignmentReport",
    "FullHalfMapBundle",
    "load_map_grid",
    "verify_grid_alignment",
    "verify_same_grid_as_reference",
    "resample_volume_onto_grid",
    "ensure_same_grid",
    "load_full_and_half_maps",
    "sliding_local_statistics_pipeline",
    "half_map_local_metrics",
    "save_half_map_metrics_mrc",
    "plot_half_map_metric_distributions",
    "plot_central_orthogonal_slices",
    "plot_volume_histogram",
    # analysis layer
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
