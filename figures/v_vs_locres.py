"""Per-structure scatter: V metric vs BlocRes local resolution at Cα."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from style.nature import PALETTES, WORD_PNG_DPI, apply, savefig as save_nature

from cryoem_mrc.metric_comparison import load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST

logger = logging.getLogger(__name__)
ANCHOR_EMD_ID = "49450"
OUTPUT_STEM = Path("figures/output/v_vs_locres")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--max-cols", type=int, default=4)
    return p.parse_args(argv)


def _entries_with_locres(manifest: Path) -> list[str]:
    """Manifest EMDB IDs that have both a PDB and finite local_resolution after load."""
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            emdb_id = str(row.get("emdb_id", "")).strip()
            pdb = row.get("flexibility_path_or_pdb", "").strip()
            if not emdb_id or not pdb or not Path(pdb).is_file():
                continue
            try:
                df = load_all_metrics(emdb_id, manifest=manifest)
            except FileNotFoundError as exc:
                logger.warning("skip EMD-%s: %s", emdb_id, exc)
                continue
            sub = df[df["in_contour_mask"].astype(bool)]
            if sub["local_resolution"].notna().sum() < 30:
                continue
            if sub["v_metric"].notna().sum() < 30:
                continue
            ids.append(emdb_id)
    return ids


def _spearman_in_mask(df) -> tuple[float, int]:
    sub = df[df["in_contour_mask"].astype(bool)]
    m = sub["v_metric"].notna() & sub["local_resolution"].notna()
    n = int(m.sum())
    if n < 10:
        return float("nan"), n
    rho, _ = stats.spearmanr(sub.loc[m, "v_metric"], sub.loc[m, "local_resolution"])
    return float(rho), n


def _plot_entry(ax, df, *, emdb_id: str, display_name: str) -> None:
    sub = df[df["in_contour_mask"].astype(bool)].copy()
    m = sub["v_metric"].notna() & sub["local_resolution"].notna()
    sub = sub.loc[m]
    chains = sub["chain"].astype(str)
    unique_chains = sorted(chains.unique())
    colors = PALETTES["categorical"]
    chain_color = {ch: colors[i % len(colors)] for i, ch in enumerate(unique_chains)}

    for ch in unique_chains:
        block = sub[chains == ch]
        ax.scatter(
            block["local_resolution"],
            block["v_metric"],
            s=6,
            alpha=0.45,
            c=chain_color[ch],
            edgecolors="none",
            label=ch,
        )

    rho, n = _spearman_in_mask(df)
    apply(ax)
    ax.set_xlabel("Local resolution (Å)")
    ax.set_ylabel("V metric")
    title = f"EMD-{emdb_id}"
    if display_name:
        title += f"\n{display_name}"
    ax.set_title(title, fontsize=7)
    ax.text(
        0.98,
        0.02,
        f"ρ = {rho:+.2f}\nn = {n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    if emdb_id == ANCHOR_EMD_ID:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("black")
        ax.set_title(title, fontsize=7, fontweight="bold")


def build_figure(
    entries: list[tuple[str, str]],
    *,
    manifest: Path = COHORT_MANIFEST,
    max_cols: int = 4,
) -> plt.Figure:
    if not entries:
        raise ValueError("No entries with both V metric and local resolution")

    n = len(entries)
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)

    for idx, (emdb_id, display_name) in enumerate(entries):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df = load_all_metrics(emdb_id, manifest=manifest)
        _plot_entry(ax, df, emdb_id=emdb_id, display_name=display_name)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig


def save_v_vs_locres_figure(fig: plt.Figure, stem: Path = OUTPUT_STEM) -> tuple[Path, Path]:
    """Write PDF (via Nature helper) and TIFF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_nature(fig, stem.with_suffix(".pdf"))
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=WORD_PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    return stem.with_suffix(".pdf"), tiff_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    display_names: dict[str, str] = {}
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                display_names[eid] = row.get("display_name", "").strip()

    ids = _entries_with_locres(args.manifest)
    if not ids:
        print("[v_vs_locres] no entries with V metric and local resolution", file=sys.stderr)
        return 1

    entries = [(eid, display_names.get(eid, "")) for eid in ids]
    if ANCHOR_EMD_ID in ids:
        entries = [(ANCHOR_EMD_ID, display_names.get(ANCHOR_EMD_ID, ""))] + [
            e for e in entries if e[0] != ANCHOR_EMD_ID
        ]

    fig = build_figure(entries, manifest=args.manifest, max_cols=args.max_cols)
    pdf_path, tiff_path = save_v_vs_locres_figure(fig)
    plt.close(fig)
    print(f"[v_vs_locres] wrote {pdf_path} and {tiff_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
