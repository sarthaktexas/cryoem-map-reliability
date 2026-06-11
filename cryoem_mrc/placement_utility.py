"""Operational utility analyses for pre-model placement guidance vs Q-scores.

Tier-1 Structure-paper analyses: low-Q enrichment, head-to-head flag rules,
calibration of reliability vs Q, mis-ranking under BlocRes, and per-map rank
recovery ρ(proxy, Q) compared across pre-model readouts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .half_map_repro import WINDOWED_HALFMAP_CORRELATION_KEY
from .incremental_prediction import (
    TARGET_Q,
    iter_eligible_emdb_ids,
    load_metrics_dataframe,
    load_qscore_target,
)
from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, lh_map_reliability_dir
from .structure_validation import load_cohort_manifest_row

QSCORE_PANEL_EXCLUDE = frozenset({"33736"})

PredictorId = Literal[
    "omit_zone",
    "reliability_below_0_33",
    "cc_below_0_5",
    "locres_worse_than_median",
    "variance_above_median",
]

PREDICTOR_LABELS: dict[PredictorId, str] = {
    "omit_zone": "Omit build zone (tercile)",
    "reliability_below_0_33": "Reliability score < 0.33",
    "cc_below_0_5": "Windowed half-map CC < 0.5",
    "locres_worse_than_median": "BlocRes worse than in-map median (Å)",
    "variance_above_median": "Local variance above in-map median",
}


@dataclass(frozen=True)
class LowQEnrichmentRow:
    emdb_id: str
    display_name: str
    global_resolution_a: float
    n_in_mask: int
    n_low_q: int
    q_threshold: float
    frac_low_q: float
    frac_low_q_in_omit_zone: float
    frac_low_q_reliability_below: float
    frac_low_q_cc_below: float
    frac_low_q_locres_worse_than_median: float
    frac_low_q_variance_above_median: float
    omit_zone_baseline: float


@dataclass(frozen=True)
class PredictorUtilityRow:
    predictor: PredictorId
    n_maps: int
    n_residues_pooled: int
    n_low_q_pooled: int
    median_frac_low_q_flagged: float
    pooled_frac_low_q_flagged: float
    pooled_sensitivity: float
    pooled_specificity: float
    pooled_balanced_accuracy: float
    median_map_balanced_accuracy: float
    median_map_auc: float
    median_map_spearman_vs_q: float


@dataclass(frozen=True)
class RankRecoveryRow:
    emdb_id: str
    global_resolution_a: float
    n_in_mask: int
    spearman_q_vs_reliability: float
    spearman_q_vs_cc: float
    spearman_q_vs_locres: float
    spearman_q_vs_variance: float
    spearman_q_vs_v: float


@dataclass(frozen=True)
class MisrankingRow:
    emdb_id: str
    global_resolution_a: float
    n_in_mask: int
    frac_sharp_locres_low_q_tercile: float
    frac_omit_zone_low_q_tercile: float
    frac_cc_above_0_7_low_q_tercile: float


@dataclass(frozen=True)
class CalibrationBin:
    reliability_bin_lo: float
    reliability_bin_hi: float
    n_residues: int
    mean_q: float
    median_q: float


@dataclass(frozen=True)
class PlacementUtilitySummary:
    q_threshold: float
    enrichment_rows: tuple[LowQEnrichmentRow, ...]
    predictor_rows: tuple[PredictorUtilityRow, ...]
    rank_recovery_rows: tuple[RankRecoveryRow, ...]
    misranking_rows: tuple[MisrankingRow, ...]
    calibration_bins: tuple[CalibrationBin, ...]
    resolution_bins: dict[str, float] = field(default_factory=dict)


def _global_resolution(manifest_row: dict[str, str]) -> float:
    raw = manifest_row.get("global_resolution_a", "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def load_map_with_qscore(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
) -> pd.DataFrame | None:
    """In-mask per-residue metrics merged with Q-scores when available."""
    metrics = load_metrics_dataframe(
        emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
    )
    if metrics is None:
        return None
    merged = load_qscore_target(metrics, emdb_id)
    if merged is None:
        return None
    if "in_contour_mask" in merged.columns:
        merged = merged[merged["in_contour_mask"].astype(bool)].copy()
    merged["emdb_id"] = str(emdb_id)
    return merged


def iter_qscore_maps(
    *,
    manifest: Path = COHORT_MANIFEST,
    exclude: frozenset[str] | None = None,
) -> list[str]:
    exclude = exclude or QSCORE_PANEL_EXCLUDE
    return iter_eligible_emdb_ids(TARGET_Q, manifest=manifest, qscore_exclude=exclude)


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    xm = x[m]
    ym = y[m]
    if np.nanstd(xm) == 0 or np.nanstd(ym) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(xm, ym)
    return float(rho)


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC; higher ``scores`` ⇒ more likely ``y_true == 1``."""
    y = y_true.astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(bool)
    p = y_pred.astype(bool)
    m = np.isfinite(y_true)  # both are bool arrays; all finite
    y = y[m]
    p = p[m]
    tp = int((p & y).sum())
    tn = int((~p & ~y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    if not np.isfinite(sens) or not np.isfinite(spec):
        return float("nan")
    return float(0.5 * (sens + spec))


def _predictor_flags(df: pd.DataFrame) -> dict[PredictorId, np.ndarray]:
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy(dtype=np.int32)

    loc_med = np.nanmedian(loc) if np.isfinite(loc).any() else float("nan")
    var_med = np.nanmedian(var) if np.isfinite(var).any() else float("nan")

    return {
        "omit_zone": zone == 0,
        "reliability_below_0_33": rel < 0.33,
        "cc_below_0_5": cc < 0.5,
        "locres_worse_than_median": loc > loc_med if np.isfinite(loc_med) else np.zeros(len(df), bool),
        "variance_above_median": var > var_med if np.isfinite(var_med) else np.zeros(len(df), bool),
    }


def _predictor_scores(df: pd.DataFrame) -> dict[PredictorId, np.ndarray]:
    """Continuous scores where higher ⇒ more likely low Q (for AUC)."""
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()

    loc_score = loc.copy()
    cc_score = -cc.copy()
    var_score = var.copy()
    rel_score = 1.0 - rel
    zone_score = np.where(zone == 0, 1.0, np.where(zone == 1, 0.5, 0.0))

    return {
        "omit_zone": zone_score,
        "reliability_below_0_33": rel_score,
        "cc_below_0_5": cc_score,
        "locres_worse_than_median": loc_score,
        "variance_above_median": var_score,
    }


def compute_low_q_enrichment_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    display_name: str = "",
    global_resolution_a: float = float("nan"),
    q_threshold: float = 0.5,
) -> LowQEnrichmentRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    n = int(m.sum())
    if n == 0:
        return None

    low = q < q_threshold
    n_low = int(low.sum())
    flags = _predictor_flags(df)

    def frac_flag(name: PredictorId) -> float:
        if n_low == 0:
            return float("nan")
        return float(flags[name][m][low[m]].mean())

    omit_base = float((pd.to_numeric(df["build_zone"], errors="coerce") == 0)[m].mean())

    return LowQEnrichmentRow(
        emdb_id=str(emdb_id),
        display_name=display_name,
        global_resolution_a=global_resolution_a,
        n_in_mask=n,
        n_low_q=n_low,
        q_threshold=q_threshold,
        frac_low_q=float(low[m].mean()),
        frac_low_q_in_omit_zone=frac_flag("omit_zone"),
        frac_low_q_reliability_below=frac_flag("reliability_below_0_33"),
        frac_low_q_cc_below=frac_flag("cc_below_0_5"),
        frac_low_q_locres_worse_than_median=frac_flag("locres_worse_than_median"),
        frac_low_q_variance_above_median=frac_flag("variance_above_median"),
        omit_zone_baseline=omit_base,
    )


def compute_misranking_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    global_resolution_a: float = float("nan"),
    q_tercile: Literal["bottom"] = "bottom",
) -> MisrankingRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()

    m = np.isfinite(q)
    if m.sum() < 30:
        return None

    q_m = q[m]
    t1 = np.percentile(q_m, 100 / 3)
    low_q = np.zeros_like(q, dtype=bool)
    low_q[m] = q_m <= t1

    loc_m = loc[m]
    sharp_locres = np.zeros_like(loc, dtype=bool)
    if np.isfinite(loc_m).sum() >= 10:
        sharp_locres[m] = loc_m <= np.nanmedian(loc_m)

    cc_m = cc[m]
    high_cc = np.zeros_like(cc, dtype=bool)
    if np.isfinite(cc_m).sum() >= 10:
        high_cc[m] = cc_m >= 0.7

    n = int(m.sum())

    def frac(cond: np.ndarray) -> float:
        n_hit = int(low_q.sum())
        if n_hit == 0:
            return float("nan")
        return float(cond[low_q].mean())

    return MisrankingRow(
        emdb_id=str(emdb_id),
        global_resolution_a=global_resolution_a,
        n_in_mask=n,
        frac_sharp_locres_low_q_tercile=frac(sharp_locres),
        frac_omit_zone_low_q_tercile=frac(zone == 0),
        frac_cc_above_0_7_low_q_tercile=frac(high_cc),
    )


def compute_rank_recovery_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    global_resolution_a: float = float("nan"),
) -> RankRecoveryRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    if m.sum() < 30:
        return None

    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    v = pd.to_numeric(df.get("v_metric", np.nan), errors="coerce").to_numpy()

    return RankRecoveryRow(
        emdb_id=str(emdb_id),
        global_resolution_a=global_resolution_a,
        n_in_mask=int(m.sum()),
        spearman_q_vs_reliability=_finite_spearman(q, rel),
        spearman_q_vs_cc=_finite_spearman(q, cc),
        spearman_q_vs_locres=_finite_spearman(q, loc),
        spearman_q_vs_variance=_finite_spearman(q, var),
        spearman_q_vs_v=_finite_spearman(q, v),
    )


def compute_calibration_bins(
    frames: Sequence[pd.DataFrame],
    *,
    n_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    rel_all: list[float] = []
    q_all: list[float] = []
    for df in frames:
        rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(rel) & np.isfinite(q)
        rel_all.extend(rel[m].tolist())
        q_all.extend(q[m].tolist())

    if not rel_all:
        return ()

    rel_a = np.asarray(rel_all, dtype=np.float64)
    q_a = np.asarray(q_all, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            mask = (rel_a >= lo) & (rel_a < hi)
        else:
            mask = (rel_a >= lo) & (rel_a <= hi)
        if not mask.any():
            continue
        bins.append(
            CalibrationBin(
                reliability_bin_lo=lo,
                reliability_bin_hi=hi,
                n_residues=int(mask.sum()),
                mean_q=float(q_a[mask].mean()),
                median_q=float(np.median(q_a[mask])),
            )
        )
    return tuple(bins)


def _summarize_predictor(
    predictor: PredictorId,
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    q_threshold: float,
) -> PredictorUtilityRow:
    flags_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    low_q_list: list[np.ndarray] = []
    frac_flagged_per_map: list[float] = []
    ba_per_map: list[float] = []
    auc_per_map: list[float] = []
    rho_per_map: list[float] = []

    for _emd, df in per_map_frames:
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(q)
        if m.sum() < 10:
            continue
        low = q < q_threshold
        flags = _predictor_flags(df)[predictor]
        scores = _predictor_scores(df)[predictor]

        flags_list.append(flags[m])
        scores_list.append(scores[m])
        low_q_list.append(low[m])

        n_low = int(low[m].sum())
        if n_low > 0:
            frac_flagged_per_map.append(float(flags[m][low[m]].mean()))
        ba_per_map.append(balanced_accuracy(low[m], flags[m]))
        auc_per_map.append(rank_auc(low[m], scores[m]))
        rho_per_map.append(_finite_spearman(q[m], scores[m]))

    if not low_q_list:
        nan = float("nan")
        return PredictorUtilityRow(
            predictor=predictor,
            n_maps=0,
            n_residues_pooled=0,
            n_low_q_pooled=0,
            median_frac_low_q_flagged=nan,
            pooled_frac_low_q_flagged=nan,
            pooled_sensitivity=nan,
            pooled_specificity=nan,
            pooled_balanced_accuracy=nan,
            median_map_balanced_accuracy=nan,
            median_map_auc=nan,
            median_map_spearman_vs_q=nan,
        )

    flags_p = np.concatenate(flags_list)
    low_p = np.concatenate(low_q_list)
    n_low_p = int(low_p.sum())
    pooled_frac = float(flags_p[low_p].mean()) if n_low_p else float("nan")

    tp = int((flags_p & low_p).sum())
    fn = int((~flags_p & low_p).sum())
    fp = int((flags_p & ~low_p).sum())
    tn = int((~flags_p & ~low_p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    pooled_ba = float(0.5 * (sens + spec)) if np.isfinite(sens) and np.isfinite(spec) else float("nan")

    return PredictorUtilityRow(
        predictor=predictor,
        n_maps=len(frac_flagged_per_map),
        n_residues_pooled=int(len(low_p)),
        n_low_q_pooled=n_low_p,
        median_frac_low_q_flagged=float(np.nanmedian(frac_flagged_per_map)),
        pooled_frac_low_q_flagged=pooled_frac,
        pooled_sensitivity=float(sens),
        pooled_specificity=float(spec),
        pooled_balanced_accuracy=pooled_ba,
        median_map_balanced_accuracy=float(np.nanmedian(ba_per_map)),
        median_map_auc=float(np.nanmedian(auc_per_map)),
        median_map_spearman_vs_q=float(np.nanmedian(rho_per_map)),
    )


def run_placement_utility_analysis(
    *,
    manifest: Path = COHORT_MANIFEST,
    q_threshold: float = 0.5,
    sphere_radius_a: float = 2.0,
    exclude: frozenset[str] | None = None,
) -> PlacementUtilitySummary:
    """Run cohort placement-utility analyses for maps with Q-score validation."""
    emdb_ids = iter_qscore_maps(manifest=manifest, exclude=exclude)
    enrichment: list[LowQEnrichmentRow] = []
    rank_rows: list[RankRecoveryRow] = []
    misrank: list[MisrankingRow] = []
    per_map_frames: list[tuple[str, pd.DataFrame]] = []
    all_frames: list[pd.DataFrame] = []

    for emdb_id in emdb_ids:
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
        except KeyError:
            continue
        display_name = row.get("display_name", "").strip()
        gres = _global_resolution(row)

        df = load_map_with_qscore(
            emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
        )
        if df is None or df.empty:
            continue

        per_map_frames.append((emdb_id, df))
        all_frames.append(df)

        enr = compute_low_q_enrichment_row(
            df,
            emdb_id=emdb_id,
            display_name=display_name,
            global_resolution_a=gres,
            q_threshold=q_threshold,
        )
        if enr is not None:
            enrichment.append(enr)

        rr = compute_rank_recovery_row(df, emdb_id=emdb_id, global_resolution_a=gres)
        if rr is not None:
            rank_rows.append(rr)

        mr = compute_misranking_row(df, emdb_id=emdb_id, global_resolution_a=gres)
        if mr is not None:
            misrank.append(mr)

    predictors: list[PredictorUtilityRow] = []
    for pid in PREDICTOR_LABELS:
        predictors.append(
            _summarize_predictor(pid, per_map_frames, q_threshold=q_threshold)
        )

    cal = compute_calibration_bins(all_frames)

    # Resolution-bin medians for ρ(Q, reliability) — atomic-building regime check.
    res_bins: dict[str, float] = {}
    if rank_rows:
        arr = [
            (r.global_resolution_a, r.spearman_q_vs_reliability)
            for r in rank_rows
            if np.isfinite(r.global_resolution_a) and np.isfinite(r.spearman_q_vs_reliability)
        ]
        if arr:
            for lo, hi, label in (
                (2.5, 3.5, "2.5_3.5"),
                (3.5, 4.5, "3.5_4.5"),
                (4.5, 99.0, "ge_4.5"),
            ):
                vals = [rho for g, rho in arr if lo <= g < hi]
                if vals:
                    res_bins[f"median_spearman_q_reliability_{label}"] = float(np.median(vals))

    return PlacementUtilitySummary(
        q_threshold=q_threshold,
        enrichment_rows=tuple(enrichment),
        predictor_rows=tuple(predictors),
        rank_recovery_rows=tuple(rank_rows),
        misranking_rows=tuple(misrank),
        calibration_bins=cal,
        resolution_bins=res_bins,
    )


def write_placement_utility_csvs(
    summary: PlacementUtilitySummary,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_enr = out_dir / "placement_low_q_enrichment.csv"
    with p_enr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "display_name",
                "global_resolution_a",
                "n_in_mask",
                "n_low_q",
                "q_threshold",
                "frac_low_q",
                "frac_low_q_in_omit_zone",
                "frac_low_q_reliability_below",
                "frac_low_q_cc_below",
                "frac_low_q_locres_worse_than_median",
                "frac_low_q_variance_above_median",
                "omit_zone_baseline",
            ],
        )
        w.writeheader()
        for r in summary.enrichment_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "display_name": r.display_name,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "n_low_q": r.n_low_q,
                    "q_threshold": f"{r.q_threshold:.2f}",
                    "frac_low_q": f"{r.frac_low_q:.4f}",
                    "frac_low_q_in_omit_zone": f"{r.frac_low_q_in_omit_zone:.4f}",
                    "frac_low_q_reliability_below": f"{r.frac_low_q_reliability_below:.4f}",
                    "frac_low_q_cc_below": f"{r.frac_low_q_cc_below:.4f}",
                    "frac_low_q_locres_worse_than_median": f"{r.frac_low_q_locres_worse_than_median:.4f}",
                    "frac_low_q_variance_above_median": f"{r.frac_low_q_variance_above_median:.4f}",
                    "omit_zone_baseline": f"{r.omit_zone_baseline:.4f}",
                }
            )
    paths["enrichment"] = p_enr

    p_pred = out_dir / "placement_predictor_head_to_head.csv"
    with p_pred.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "n_maps",
                "n_residues_pooled",
                "n_low_q_pooled",
                "median_frac_low_q_flagged",
                "pooled_frac_low_q_flagged",
                "pooled_sensitivity",
                "pooled_specificity",
                "pooled_balanced_accuracy",
                "median_map_balanced_accuracy",
                "median_map_auc",
                "median_map_spearman_vs_q",
            ],
        )
        w.writeheader()
        for r in summary.predictor_rows:
            w.writerow(
                {
                    "predictor": r.predictor,
                    "label": PREDICTOR_LABELS[r.predictor],
                    "n_maps": r.n_maps,
                    "n_residues_pooled": r.n_residues_pooled,
                    "n_low_q_pooled": r.n_low_q_pooled,
                    "median_frac_low_q_flagged": f"{r.median_frac_low_q_flagged:.4f}",
                    "pooled_frac_low_q_flagged": f"{r.pooled_frac_low_q_flagged:.4f}",
                    "pooled_sensitivity": f"{r.pooled_sensitivity:.4f}",
                    "pooled_specificity": f"{r.pooled_specificity:.4f}",
                    "pooled_balanced_accuracy": f"{r.pooled_balanced_accuracy:.4f}",
                    "median_map_balanced_accuracy": f"{r.median_map_balanced_accuracy:.4f}",
                    "median_map_auc": f"{r.median_map_auc:.4f}",
                    "median_map_spearman_vs_q": f"{r.median_map_spearman_vs_q:.4f}",
                }
            )
    paths["predictors"] = p_pred

    p_rr = out_dir / "placement_rank_recovery.csv"
    with p_rr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "global_resolution_a",
                "n_in_mask",
                "spearman_q_vs_reliability",
                "spearman_q_vs_cc",
                "spearman_q_vs_locres",
                "spearman_q_vs_variance",
                "spearman_q_vs_v",
            ],
        )
        w.writeheader()
        for r in summary.rank_recovery_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "spearman_q_vs_reliability": f"{r.spearman_q_vs_reliability:.4f}",
                    "spearman_q_vs_cc": f"{r.spearman_q_vs_cc:.4f}",
                    "spearman_q_vs_locres": f"{r.spearman_q_vs_locres:.4f}",
                    "spearman_q_vs_variance": f"{r.spearman_q_vs_variance:.4f}",
                    "spearman_q_vs_v": f"{r.spearman_q_vs_v:.4f}",
                }
            )
    paths["rank_recovery"] = p_rr

    p_mr = out_dir / "placement_misranking.csv"
    with p_mr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "global_resolution_a",
                "n_in_mask",
                "frac_sharp_locres_low_q_tercile",
                "frac_omit_zone_low_q_tercile",
                "frac_cc_above_0_7_low_q_tercile",
            ],
        )
        w.writeheader()
        for r in summary.misranking_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "frac_sharp_locres_low_q_tercile": f"{r.frac_sharp_locres_low_q_tercile:.4f}",
                    "frac_omit_zone_low_q_tercile": f"{r.frac_omit_zone_low_q_tercile:.4f}",
                    "frac_cc_above_0_7_low_q_tercile": f"{r.frac_cc_above_0_7_low_q_tercile:.4f}",
                }
            )
    paths["misranking"] = p_mr

    p_cal = out_dir / "placement_q_calibration_bins.csv"
    with p_cal.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "reliability_bin_lo",
                "reliability_bin_hi",
                "n_residues",
                "mean_q",
                "median_q",
            ],
        )
        w.writeheader()
        for b in summary.calibration_bins:
            w.writerow(
                {
                    "reliability_bin_lo": f"{b.reliability_bin_lo:.2f}",
                    "reliability_bin_hi": f"{b.reliability_bin_hi:.2f}",
                    "n_residues": b.n_residues,
                    "mean_q": f"{b.mean_q:.4f}",
                    "median_q": f"{b.median_q:.4f}",
                }
            )
    paths["calibration"] = p_cal

    return paths


def write_placement_utility_markdown(
    summary: PlacementUtilitySummary,
    path: Path,
) -> Path:
    """Human-readable summary for thesis / Structure paper supplement."""
    path = Path(path)
    lines: list[str] = [
        "# Placement utility analysis (Q-score operational validation)",
        "",
        f"Low-Q definition: Q-score < **{summary.q_threshold:.2f}** at in-mask Cα.",
        f"Maps analyzed: **{len(summary.enrichment_rows)}** with Q-score validation.",
        "",
        "## Tier 1 — Low-Q enrichment",
        "",
        "Among residues with Q below threshold, fraction flagged by each pre-model readout.",
        "Omit-zone baseline ≈ 0.33 by construction (tercile). Enrichment above baseline ⇒ utility.",
        "",
    ]

    if summary.enrichment_rows:
        def med(attr: str) -> float:
            vals = [getattr(r, attr) for r in summary.enrichment_rows]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lines.extend(
            [
                "| Readout | Median frac. of low-Q residues flagged |",
                "|---------|----------------------------------------|",
                f"| Omit zone | {med('frac_low_q_in_omit_zone'):.3f} |",
                f"| Reliability < 0.33 | {med('frac_low_q_reliability_below'):.3f} |",
                f"| CC < 0.5 | {med('frac_low_q_cc_below'):.3f} |",
                f"| BlocRes worse than median | {med('frac_low_q_locres_worse_than_median'):.3f} |",
                f"| Variance above median | {med('frac_low_q_variance_above_median'):.3f} |",
                f"| Omit-zone baseline (all Cα) | {med('omit_zone_baseline'):.3f} |",
                "",
            ]
        )

    lines.extend(["## Tier 1 — Head-to-head predictors (pooled cohort)", ""])
    if summary.predictor_rows:
        lines.extend(
            [
                "| Predictor | Pooled frac. low-Q flagged | Pooled BA | Median map AUC |",
                "|-----------|----------------------------|-----------|----------------|",
            ]
        )
        for r in summary.predictor_rows:
            lines.append(
                f"| {PREDICTOR_LABELS[r.predictor]} | "
                f"{r.pooled_frac_low_q_flagged:.3f} | "
                f"{r.pooled_balanced_accuracy:.3f} | "
                f"{r.median_map_auc:.3f} |"
            )
        lines.append("")

    lines.extend(["## Tier 2 — Rank recovery ρ(Q, proxy) medians", ""])
    if summary.rank_recovery_rows:
        def med_rr(attr: str) -> float:
            vals = [getattr(r, attr) for r in summary.rank_recovery_rows]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lines.extend(
            [
                f"- ρ(Q, reliability): **{med_rr('spearman_q_vs_reliability'):.3f}**",
                f"- ρ(Q, windowed CC): {med_rr('spearman_q_vs_cc'):.3f}",
                f"- ρ(Q, BlocRes): {med_rr('spearman_q_vs_locres'):.3f}",
                f"- ρ(Q, local variance): {med_rr('spearman_q_vs_variance'):.3f}",
                f"- ρ(Q, constraint V): {med_rr('spearman_q_vs_v'):.3f}",
                "",
            ]
        )
        for k, v in sorted(summary.resolution_bins.items()):
            lines.append(f"- {k}: {v:.3f}")
        lines.append("")

    lines.extend(["## Tier 1 — Mis-ranking (bottom Q tercile)", ""])
    if summary.misranking_rows:
        def med_mr(attr: str) -> float:
            vals = [getattr(r, attr) for r in summary.misranking_rows]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lines.extend(
            [
                f"- Fraction with **sharp BlocRes** (≤ median Å) but bottom-Q tercile: "
                f"{med_mr('frac_sharp_locres_low_q_tercile'):.3f}",
                f"- Fraction in **omit zone** among bottom-Q tercile: "
                f"{med_mr('frac_omit_zone_low_q_tercile'):.3f}",
                f"- Fraction with **CC ≥ 0.7** among bottom-Q tercile: "
                f"{med_mr('frac_cc_above_0_7_low_q_tercile'):.3f}",
                "",
            ]
        )

    lines.append("## Calibration")
    lines.append("")
    lines.append("Reliability score deciles vs mean Q — see `placement_q_calibration_bins.csv`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- Semi-prospective (leave-one-map-out) and ROC utilities ---

TRAIN_DERIVED_PREDICTORS: frozenset[PredictorId] = frozenset(
    {"locres_worse_than_median", "variance_above_median"}
)

MAIN_ROC_PREDICTORS: tuple[PredictorId, ...] = (
    "reliability_below_0_33",
    "cc_below_0_5",
    "omit_zone",
    "locres_worse_than_median",
)


@dataclass(frozen=True)
class LomoPredictorFoldRow:
    held_out_emdb_id: str
    predictor: PredictorId
    global_resolution_a: float
    n_residues: int
    n_low_q: int
    balanced_accuracy: float
    auc: float
    spearman_q_vs_score: float
    frac_low_q_flagged: float
    train_locres_median: float = float("nan")
    train_variance_median: float = float("nan")


@dataclass(frozen=True)
class LomoPlacementSummary:
    q_threshold: float
    fold_rows: tuple[LomoPredictorFoldRow, ...]
    predictor_medians: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RocCurve:
    predictor: PredictorId
    fpr: tuple[float, ...]
    tpr: tuple[float, ...]
    auc: float


def _train_medians(train_dfs: Sequence[pd.DataFrame]) -> tuple[float, float]:
    loc_parts: list[np.ndarray] = []
    var_parts: list[np.ndarray] = []
    for df in train_dfs:
        loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
        var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
        loc_parts.append(loc[np.isfinite(loc)])
        var_parts.append(var[np.isfinite(var)])
    loc_all = np.concatenate(loc_parts) if loc_parts else np.array([], dtype=np.float64)
    var_all = np.concatenate(var_parts) if var_parts else np.array([], dtype=np.float64)
    loc_med = float(np.median(loc_all)) if loc_all.size else float("nan")
    var_med = float(np.median(var_all)) if var_all.size else float("nan")
    return loc_med, var_med


def _lomo_predictor_flags(
    df: pd.DataFrame,
    predictor: PredictorId,
    *,
    train_locres_median: float = float("nan"),
    train_variance_median: float = float("nan"),
) -> np.ndarray:
    if predictor not in TRAIN_DERIVED_PREDICTORS:
        return _predictor_flags(df)[predictor]
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    if predictor == "locres_worse_than_median":
        if not np.isfinite(train_locres_median):
            return np.zeros(len(df), dtype=bool)
        return loc > train_locres_median
    if not np.isfinite(train_variance_median):
        return np.zeros(len(df), dtype=bool)
    return var > train_variance_median


def _predictor_rank_proxy(df: pd.DataFrame, predictor: PredictorId) -> np.ndarray:
    """Continuous proxy where higher values should track higher Q (for Spearman ρ)."""
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()
    return {
        "omit_zone": zone,
        "reliability_below_0_33": rel,
        "cc_below_0_5": cc,
        "locres_worse_than_median": loc,
        "variance_above_median": var,
    }[predictor]


def evaluate_map_predictor(
    df: pd.DataFrame,
    predictor: PredictorId,
    *,
    q_threshold: float,
    train_locres_median: float = float("nan"),
    train_variance_median: float = float("nan"),
) -> tuple[float, float, float, float, int, int]:
    """Return BA, AUC, Spearman ρ, frac low-Q flagged, n_residues, n_low_q."""
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    if int(m.sum()) < 10:
        nan = float("nan")
        return nan, nan, nan, nan, int(m.sum()), 0
    low = q < q_threshold
    flags = _lomo_predictor_flags(
        df,
        predictor,
        train_locres_median=train_locres_median,
        train_variance_median=train_variance_median,
    )
    scores = _predictor_scores(df)[predictor]
    n_low = int(low[m].sum())
    frac_flag = float(flags[m][low[m]].mean()) if n_low else float("nan")
    return (
        balanced_accuracy(low[m], flags[m]),
        rank_auc(low[m], scores[m]),
        _finite_spearman(q[m], _predictor_rank_proxy(df, predictor)[m]),
        frac_flag,
        int(m.sum()),
        n_low,
    )


def run_lomo_placement_validation(
    per_map_frames: Sequence[tuple[str, pd.DataFrame, float]],
    *,
    q_threshold: float = 0.5,
) -> LomoPlacementSummary:
    """Leave-one-map-out evaluation; train-derived medians for BlocRes/variance flags."""
    if len(per_map_frames) < 3:
        raise ValueError("need at least three maps for leave-one-map-out validation")

    fold_rows: list[LomoPredictorFoldRow] = []
    for held_out_id, test_df, gres in per_map_frames:
        train = [(eid, df) for eid, df, _ in per_map_frames if eid != held_out_id]
        train_dfs = [df for _, df in train]
        loc_med, var_med = _train_medians(train_dfs)
        for pid in PREDICTOR_LABELS:
            ba, auc, rho, frac_flag, n_res, n_low = evaluate_map_predictor(
                test_df,
                pid,
                q_threshold=q_threshold,
                train_locres_median=loc_med,
                train_variance_median=var_med,
            )
            fold_rows.append(
                LomoPredictorFoldRow(
                    held_out_emdb_id=str(held_out_id),
                    predictor=pid,
                    global_resolution_a=gres,
                    n_residues=n_res,
                    n_low_q=n_low,
                    balanced_accuracy=ba,
                    auc=auc,
                    spearman_q_vs_score=rho,
                    frac_low_q_flagged=frac_flag,
                    train_locres_median=loc_med,
                    train_variance_median=var_med,
                )
            )

    predictor_medians: dict[str, dict[str, float]] = {}
    for pid in PREDICTOR_LABELS:
        sub = [r for r in fold_rows if r.predictor == pid]
        for attr in ("balanced_accuracy", "auc", "spearman_q_vs_score", "frac_low_q_flagged"):
            vals = [getattr(r, attr) for r in sub if np.isfinite(getattr(r, attr))]
            predictor_medians.setdefault(pid, {})[f"median_{attr}"] = (
                float(np.median(vals)) if vals else float("nan")
            )

    return LomoPlacementSummary(
        q_threshold=q_threshold,
        fold_rows=tuple(fold_rows),
        predictor_medians=predictor_medians,
    )


def pooled_roc_curve(
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    predictor: PredictorId,
    *,
    q_threshold: float,
) -> RocCurve:
    """Pooled cohort ROC for low-Q classification (y = Q < threshold)."""
    low_list: list[np.ndarray] = []
    score_list: list[np.ndarray] = []
    for _emd, df in per_map_frames:
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(q)
        if m.sum() < 5:
            continue
        low_list.append((q < q_threshold)[m])
        score_list.append(_predictor_scores(df)[predictor][m])
    if not low_list:
        return RocCurve(predictor=predictor, fpr=(), tpr=(), auc=float("nan"))

    y = np.concatenate(low_list).astype(bool)
    s = np.concatenate(score_list)
    auc = rank_auc(y, s)
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return RocCurve(predictor=predictor, fpr=(), tpr=(), auc=auc)

    tps = np.cumsum(y_sorted).astype(np.float64)
    fps = np.cumsum(~y_sorted).astype(np.float64)
    tpr_pts = np.concatenate([[0.0], tps / n_pos])
    fpr_pts = np.concatenate([[0.0], fps / n_neg])
    return RocCurve(
        predictor=predictor,
        fpr=tuple(float(x) for x in fpr_pts),
        tpr=tuple(float(x) for x in tpr_pts),
        auc=float(auc),
    )


def load_per_map_frames_for_lomo(
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    exclude: frozenset[str] | None = None,
) -> list[tuple[str, pd.DataFrame, float]]:
    """Load (emdb_id, dataframe, global_resolution) for maps with Q-scores."""
    frames: list[tuple[str, pd.DataFrame, float]] = []
    for emdb_id in iter_qscore_maps(manifest=manifest, exclude=exclude):
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
        except KeyError:
            continue
        df = load_map_with_qscore(
            emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
        )
        if df is None or df.empty:
            continue
        frames.append((emdb_id, df, _global_resolution(row)))
    return frames


def write_lomo_placement_csvs(
    summary: LomoPlacementSummary,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_folds = out_dir / "placement_lomo_folds.csv"
    with p_folds.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "held_out_emdb_id",
                "predictor",
                "label",
                "global_resolution_a",
                "n_residues",
                "n_low_q",
                "balanced_accuracy",
                "auc",
                "spearman_q_vs_score",
                "frac_low_q_flagged",
                "train_locres_median",
                "train_variance_median",
            ],
        )
        w.writeheader()
        for r in summary.fold_rows:
            w.writerow(
                {
                    "held_out_emdb_id": r.held_out_emdb_id,
                    "predictor": r.predictor,
                    "label": PREDICTOR_LABELS[r.predictor],
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_residues": r.n_residues,
                    "n_low_q": r.n_low_q,
                    "balanced_accuracy": f"{r.balanced_accuracy:.4f}",
                    "auc": f"{r.auc:.4f}",
                    "spearman_q_vs_score": f"{r.spearman_q_vs_score:.4f}",
                    "frac_low_q_flagged": f"{r.frac_low_q_flagged:.4f}",
                    "train_locres_median": f"{r.train_locres_median:.3f}"
                    if np.isfinite(r.train_locres_median)
                    else "",
                    "train_variance_median": f"{r.train_variance_median:.3f}"
                    if np.isfinite(r.train_variance_median)
                    else "",
                }
            )
    paths["folds"] = p_folds

    p_med = out_dir / "placement_lomo_medians.csv"
    with p_med.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "median_balanced_accuracy",
                "median_auc",
                "median_spearman_q_vs_score",
                "median_frac_low_q_flagged",
            ],
        )
        w.writeheader()
        for pid in PREDICTOR_LABELS:
            meds = summary.predictor_medians.get(pid, {})
            w.writerow(
                {
                    "predictor": pid,
                    "label": PREDICTOR_LABELS[pid],
                    "median_balanced_accuracy": f"{meds.get('median_balanced_accuracy', float('nan')):.4f}",
                    "median_auc": f"{meds.get('median_auc', float('nan')):.4f}",
                    "median_spearman_q_vs_score": f"{meds.get('median_spearman_q_vs_score', float('nan')):.4f}",
                    "median_frac_low_q_flagged": f"{meds.get('median_frac_low_q_flagged', float('nan')):.4f}",
                }
            )
    paths["medians"] = p_med
    return paths


def write_lomo_placement_markdown(summary: LomoPlacementSummary, path: Path) -> Path:
    path = Path(path)
    n_maps = len({r.held_out_emdb_id for r in summary.fold_rows})
    lines = [
        "# Semi-prospective placement validation (leave-one-map-out)",
        "",
        f"Low-Q definition: Q-score < **{summary.q_threshold:.2f}**.",
        f"Maps: **{n_maps}** held-out folds; BlocRes/variance thresholds fit on the other *N*−1 maps.",
        "",
        "## Median held-out metrics",
        "",
        "| Predictor | Median BA | Median AUC | Median ρ(Q, score) |",
        "|-----------|-----------|------------|---------------------|",
    ]
    for pid in PREDICTOR_LABELS:
        meds = summary.predictor_medians.get(pid, {})
        lines.append(
            f"| {PREDICTOR_LABELS[pid]} | "
            f"{meds.get('median_balanced_accuracy', float('nan')):.3f} | "
            f"{meds.get('median_auc', float('nan')):.3f} | "
            f"{meds.get('median_spearman_q_vs_score', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "Fixed-threshold readouts (reliability < 0.33, CC < 0.5, omit zone) do not use training data;",
            "LOMO confirms per-map generalization rather than cohort pooling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
