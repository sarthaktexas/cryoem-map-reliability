"""Copy thesis figures from ``outputs/`` into ``docs/figures/`` for self-contained prose.

``docs/THESIS_NARRATIVE.md`` embeds only ``figures/<name>.png`` paths (relative to
``docs/``). Numeric CSV/JSON exports stay in ``outputs/cohort_summary/``.

Example::

    .venv/bin/python scripts/sync_thesis_narrative_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryoem_mrc.repo_paths import (
    OUTPUTS_ROOT,
    sync_thesis_appendix_b1_from_anchor,
    sync_thesis_narrative_cohort_figures,
    sync_thesis_doc_figure,
    THESIS_APPENDIX_B_FIGURES,
)

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    return p.parse_args()


def _sync_appendix_b_extras() -> list[Path]:
    synced: list[Path] = []
    b1 = sync_thesis_appendix_b1_from_anchor()
    if b1 is not None:
        synced.append(b1)
    sens = OUTPUTS_ROOT / "sensitivity"
    pairs = [
        (sens / "local_fsc" / "figures" / "spearman_vs_cc_bar.png", "fig_b2_local_fsc_sensitivity_bar.png"),
        (sens / "contour" / "figures" / "emd_49450_contour_sensitivity.png", "fig_b3_contour_sensitivity.png"),
    ]
    for src, dest_name in pairs:
        if src.is_file():
            synced.append(sync_thesis_doc_figure(src, dest_name))
    return synced


def main() -> int:
    args = _parse_args()
    synced = sync_thesis_narrative_cohort_figures(args.cohort_dir)
    synced.extend(_sync_appendix_b_extras())
    if not synced:
        print("[sync_thesis] no source figures found", file=sys.stderr)
        return 2
    print(f"[sync_thesis] {len(synced)} figures → docs/figures/", flush=True)
    for path in synced:
        try:
            rel = path.relative_to(REPO)
        except ValueError:
            rel = path
        print(f"  {rel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
