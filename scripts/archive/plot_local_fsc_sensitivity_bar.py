"""Plot local FSC sensitivity bar chart from Decision 007 anchor numbers.

Use when full ``run_local_fsc_sensitivity.py`` recomputation is unavailable.
Full 2×3 midplane panel requires the six MRC outputs from that driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style.nature import apply, label_panel, savefig as save_nature

from cryoem_mrc.repo_paths import sensitivity_local_fsc_dir, sync_thesis_appendix_b_figure

# EMD-49450, contour 0.116, stride 4 (Decision 007 validation panel).
RECORDS = [
    {"patch_size": 13, "fsc_threshold": 0.143, "spearman_vs_local_cc": -0.8728},
    {"patch_size": 17, "fsc_threshold": 0.143, "spearman_vs_local_cc": -0.9006},
    {"patch_size": 25, "fsc_threshold": 0.143, "spearman_vs_local_cc": -0.9004},
    {"patch_size": 13, "fsc_threshold": 0.5, "spearman_vs_local_cc": -0.6512},
    {"patch_size": 17, "fsc_threshold": 0.5, "spearman_vs_local_cc": -0.6841},
    {"patch_size": 25, "fsc_threshold": 0.5, "spearman_vs_local_cc": -0.7210},
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=sensitivity_local_fsc_dir())
    args = p.parse_args()
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"P={r['patch_size']} t={r['fsc_threshold']}" for r in RECORDS]
    rhos = [r["spearman_vs_local_cc"] for r in RECORDS]
    colors = ["#4C72B0" if r["fsc_threshold"] == 0.143 else "#DD8452" for r in RECORDS]

    fig, ax = plt.subplots(figsize=(8, 4))
    apply(ax)
    ax.bar(range(len(labels)), rhos, color=colors)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Spearman(local_CC, local_FSC Å)")
    ax.set_title("EMD-49450 local FSC sensitivity (Decision 007)")
    label_panel(ax, "a")
    fig.tight_layout()
    out = fig_dir / "spearman_vs_cc_bar.png"
    save_nature(fig, out)
    plt.close(fig)
    thesis_path = sync_thesis_appendix_b_figure(out, "fig_b2_local_fsc_sensitivity_bar.png")
    print(f"[plot_lfsc_sens] wrote {out}")
    print(f"[plot_lfsc_sens] synced → {thesis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
