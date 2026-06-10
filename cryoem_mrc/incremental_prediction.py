"""Leave-one-map-out nested prediction: does V add held-out value beyond variance?

Design A (capability gate): baseline = {local variance, half-map CC, BlocRes locres};
full model adds constraint V. Resampling unit is the map (EMDB entry), not residue.
Predictors and targets are percentile-ranked within each map before pooling so scale
differences across maps do not dominate OLS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .half_map_repro import LEGACY_HALFMAP_CORRELATION_KEY, WINDOWED_HALFMAP_CORRELATION_KEY
from .metric_comparison import load_all_metrics
from .repo_paths import COHORT_MANIFEST, emd_output_dir, lh_map_reliability_dir
from .structure_validation import _b_iso_is_uniform, load_cohort_manifest_row

BASELINE_COLUMNS: tuple[str, ...] = (
    "local_variance",
    WINDOWED_HALFMAP_CORRELATION_KEY,
    "local_resolution",
)
V_COLUMN = "v_metric"
TARGET_Q = "q_score"
TARGET_B = "b_factor"

ModelKind = Literal["baseline", "full"]


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map finite values to percentile ranks in (0, 1]; NaN elsewhere."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    m = np.isfinite(x)
    n = int(m.sum())
    if n == 0:
        return out
    out[m] = stats.rankdata(x[m]) / n
    return out


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return OLS coefficients ``[intercept, beta...]``."""
    design = np.column_stack([np.ones(len(y), dtype=np.float64), X])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def ols_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    return design @ coef


def ols_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ols_gaussian_loglik(y: np.ndarray, y_hat: np.ndarray, *, n_params: int) -> float:
    """Gaussian log-likelihood with MLE sigma^2 = RSS/n."""
    n = len(y)
    if n == 0:
        return float("nan")
    rss = float(np.sum((y - y_hat) ** 2))
    sigma2 = rss / n
    if sigma2 <= 0.0:
        return float("inf")
    return float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0))


@dataclass(frozen=True)
class MapPredictionFrame:
    """Rank-transformed residues for one map."""

    emdb_id: str
    X_baseline: np.ndarray
    X_full: np.ndarray
    y: np.ndarray
    n_residues: int


@dataclass(frozen=True)
class LomOFoldResult:
    emdb_id: str
    target: str
    n_residues: int
    r2_baseline: float
    r2_full: float
    delta_r2: float
    delta_loglik: float
    delta_loglik_per_residue: float


@dataclass(frozen=True)
class IncrementalPredictionSummary:
    target: str
    n_maps: int
    fold_results: tuple[LomOFoldResult, ...]
    median_delta_r2: float
    mean_delta_r2: float
    n_positive_delta_r2: int
    sign_test_p_value: float
    median_delta_loglik: float


def _required_columns(target_col: str) -> tuple[str, ...]:
    return (*BASELINE_COLUMNS, V_COLUMN, target_col)


def _rank_frame(raw: pd.DataFrame, *, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cols = list(BASELINE_COLUMNS) + [V_COLUMN, target_col]
    sub = raw[cols].apply(pd.to_numeric, errors="coerce")
    ranked = sub.apply(percentile_rank, axis=0)
    ok = ranked.notna().all(axis=1)
    if int(ok.sum()) < 2:
        return None
    baseline = ranked.loc[ok, list(BASELINE_COLUMNS)].to_numpy(dtype=np.float64)
    full = ranked.loc[ok, [*BASELINE_COLUMNS, V_COLUMN]].to_numpy(dtype=np.float64)
    y = ranked.loc[ok, target_col].to_numpy(dtype=np.float64)
    return baseline, full, y


def build_map_frame_from_metrics(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    target_col: str,
    min_residues: int = 30,
) -> MapPredictionFrame | None:
    """Build one map frame from ``load_all_metrics`` (+ merged target column)."""
    if "in_contour_mask" in df.columns:
        use = df[df["in_contour_mask"].astype(bool)].copy()
    else:
        use = df.copy()

    missing = [c for c in _required_columns(target_col) if c not in use.columns]
    if missing:
        return None

    if target_col == TARGET_B:
        b = pd.to_numeric(use[TARGET_B], errors="coerce").to_numpy(dtype=np.float64)
        if _b_iso_is_uniform(b):
            return None

    ok = use[list(_required_columns(target_col))].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    n_ok = int(ok.sum())
    if n_ok < min_residues:
        return None

    ranked = _rank_frame(use.loc[ok], target_col=target_col)
    if ranked is None:
        return None
    baseline, full, y = ranked
    return MapPredictionFrame(
        emdb_id=str(emdb_id),
        X_baseline=baseline,
        X_full=full,
        y=y,
        n_residues=n_ok,
    )


def load_qscore_target(metrics_df: pd.DataFrame, emdb_id: str) -> pd.DataFrame | None:
    """Merge per-residue Q-scores from ``lh_map_reliability/qscore_validation.csv``."""
    q_path = lh_map_reliability_dir(emdb_id) / "qscore_validation.csv"
    if not q_path.is_file():
        return None
    q_df = pd.read_csv(q_path, usecols=["chain", "seq_num", "q_score"])
    q_df["q_score"] = pd.to_numeric(q_df["q_score"], errors="coerce")
    merged = metrics_df.merge(q_df, on=["chain", "seq_num"], how="left")
    return merged


def normalize_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Align cached CSV column names with ``load_all_metrics`` output."""
    out = df.copy()
    if (
        WINDOWED_HALFMAP_CORRELATION_KEY not in out.columns
        and LEGACY_HALFMAP_CORRELATION_KEY in out.columns
    ):
        out[WINDOWED_HALFMAP_CORRELATION_KEY] = out[LEGACY_HALFMAP_CORRELATION_KEY]
    return out


def load_metrics_dataframe(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
) -> pd.DataFrame | None:
    """Load per-residue metrics, preferring cached ``metric_comparison/residue_metrics.csv``."""
    cached = emd_output_dir(emdb_id) / "metric_comparison" / "residue_metrics.csv"
    if cached.is_file():
        return normalize_metrics_columns(pd.read_csv(cached))
    try:
        return load_all_metrics(emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a)
    except (FileNotFoundError, ValueError, KeyError):
        return None


def load_map_frame(
    emdb_id: str,
    *,
    target: Literal["q_score", "b_factor"],
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    min_residues: int = 30,
) -> MapPredictionFrame | None:
    """Load metrics and build a rank-transformed frame for one map."""
    metrics = load_metrics_dataframe(
        emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
    )
    if metrics is None:
        return None

    if target == TARGET_Q:
        metrics = load_qscore_target(metrics, emdb_id)
        if metrics is None:
            return None
        target_col = TARGET_Q
    elif target == TARGET_B:
        row = load_cohort_manifest_row(manifest, emdb_id)
        if row.get("flexibility_source", "").strip() != "b_factor":
            return None
        target_col = TARGET_B
    else:
        raise ValueError(f"unsupported target: {target!r}")

    return build_map_frame_from_metrics(
        metrics,
        emdb_id=emdb_id,
        target_col=target_col,
        min_residues=min_residues,
    )


def iter_eligible_emdb_ids(
    target: Literal["q_score", "b_factor"],
    *,
    manifest: Path = COHORT_MANIFEST,
    outputs_root: Path | None = None,
    qscore_exclude: frozenset[str] | None = None,
) -> list[str]:
    """List candidate EMDB IDs for a target without loading full metrics."""
    from .repo_paths import OUTPUTS_ROOT

    exclude = qscore_exclude or frozenset()
    root = outputs_root or OUTPUTS_ROOT
    if target == TARGET_Q:
        ids: list[str] = []
        for path in sorted(root.glob("emd_*/lh_map_reliability/qscore_validation.csv")):
            emdb_id = path.parts[-3].replace("emd_", "")
            if emdb_id not in exclude:
                ids.append(emdb_id)
        return ids

    ids = []
    with manifest.open(newline="") as f:
        import csv

        for row in csv.DictReader(f):
            if row.get("flexibility_source", "").strip() == "b_factor":
                ids.append(str(row["emdb_id"]).strip())
    return ids


def lomo_fold(
    frames: Sequence[MapPredictionFrame],
    held_out_id: str,
    *,
    target: str,
    model: ModelKind,
) -> LomOFoldResult:
    """Train on all maps except ``held_out_id``; evaluate on the held-out map."""
    train = [f for f in frames if f.emdb_id != held_out_id]
    test = next(f for f in frames if f.emdb_id == held_out_id)

    X_train = np.vstack([f.X_baseline if model == "baseline" else f.X_full for f in train])
    y_train = np.concatenate([f.y for f in train])
    coef = ols_fit(X_train, y_train)

    X_test = test.X_baseline if model == "baseline" else test.X_full
    y_hat = ols_predict(coef, X_test)
    n_params = X_test.shape[1] + 1

    r2 = ols_r2(test.y, y_hat)
    ll = ols_gaussian_loglik(test.y, y_hat, n_params=n_params)

    return LomOFoldResult(
        emdb_id=held_out_id,
        target=target,
        n_residues=test.n_residues,
        r2_baseline=r2 if model == "baseline" else float("nan"),
        r2_full=r2 if model == "full" else float("nan"),
        delta_r2=float("nan"),
        delta_loglik=float("nan"),
        delta_loglik_per_residue=float("nan"),
    )


def run_lomo_incremental_prediction(
    frames: Sequence[MapPredictionFrame],
    *,
    target: str,
) -> IncrementalPredictionSummary:
    """Leave-one-map-out comparison of baseline vs baseline+V."""
    if len(frames) < 3:
        raise ValueError("need at least three maps for leave-one-map-out CV")

    fold_rows: list[LomOFoldResult] = []
    for frame in frames:
        base = lomo_fold(frames, frame.emdb_id, target=target, model="baseline")
        full = lomo_fold(frames, frame.emdb_id, target=target, model="full")
        delta_r2 = full.r2_full - base.r2_baseline
        # Recompute loglik for delta (full model has one extra parameter).
        train = [f for f in frames if f.emdb_id != frame.emdb_id]
        Xb = np.vstack([f.X_baseline for f in train])
        yb = np.concatenate([f.y for f in train])
        Xf = np.vstack([f.X_full for f in train])
        coef_b = ols_fit(Xb, yb)
        coef_f = ols_fit(Xf, yb)
        y_hat_b = ols_predict(coef_b, frame.X_baseline)
        y_hat_f = ols_predict(coef_f, frame.X_full)
        ll_b = ols_gaussian_loglik(frame.y, y_hat_b, n_params=frame.X_baseline.shape[1] + 1)
        ll_f = ols_gaussian_loglik(frame.y, y_hat_f, n_params=frame.X_full.shape[1] + 1)
        delta_ll = ll_f - ll_b
        fold_rows.append(
            LomOFoldResult(
                emdb_id=frame.emdb_id,
                target=target,
                n_residues=frame.n_residues,
                r2_baseline=base.r2_baseline,
                r2_full=full.r2_full,
                delta_r2=delta_r2,
                delta_loglik=delta_ll,
                delta_loglik_per_residue=delta_ll / frame.n_residues if frame.n_residues else float("nan"),
            )
        )

    deltas = np.array([r.delta_r2 for r in fold_rows], dtype=np.float64)
    finite = deltas[np.isfinite(deltas)]
    n_pos = int(np.sum(finite > 0))
    n_folds = len(finite)
    # Two-sided sign test against median delta = 0.
    if n_folds == 0:
        p_sign = float("nan")
    else:
        p_sign = float(stats.binomtest(n_pos, n_folds, p=0.5, alternative="two-sided").pvalue)

    delta_ll = np.array([r.delta_loglik for r in fold_rows], dtype=np.float64)
    return IncrementalPredictionSummary(
        target=target,
        n_maps=len(frames),
        fold_results=tuple(fold_rows),
        median_delta_r2=float(np.nanmedian(deltas)),
        mean_delta_r2=float(np.nanmean(deltas)),
        n_positive_delta_r2=n_pos,
        sign_test_p_value=p_sign,
        median_delta_loglik=float(np.nanmedian(delta_ll)),
    )
