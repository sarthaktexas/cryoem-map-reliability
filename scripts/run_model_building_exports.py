"""Export ChimeraX / Coot build-zone coloring for anchor maps.

Default maps: MgtA (EMD-49450) and ClpB decoupling case (EMD-4941).

Example::

    python scripts/run_model_building_exports.py
    python scripts/run_model_building_exports.py --emd-id 49450 --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryoem_mrc.chimerax_figures import find_chimerax_executable, run_chimerax_script
from cryoem_mrc.model_building_export import export_model_building_assets
from cryoem_mrc.repo_paths import COHORT_MANIFEST

DEFAULT_EMD_IDS = ("49450", "4941")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", action="append", dest="emd_ids", help="EMDB ID (repeatable)")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--render",
        action="store_true",
        help="Invoke ChimeraX to render build_zones_preview.png when available",
    )
    p.add_argument("--chimerax", type=Path, default=None, help="ChimeraX executable path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    emd_ids = args.emd_ids or list(DEFAULT_EMD_IDS)
    exe = find_chimerax_executable(args.chimerax)

    for emdb_id in emd_ids:
        try:
            outputs = export_model_building_assets(emdb_id, manifest=args.manifest)
        except FileNotFoundError as exc:
            print(f"[model_building] ERROR EMD-{emdb_id}: {exc}", file=sys.stderr)
            continue
        print(f"[model_building] EMD-{emdb_id}:", flush=True)
        for key, path in outputs.items():
            print(f"  {key}: {path}", flush=True)

        if args.render and exe is not None:
            script = outputs["chimerax_script"]
            ok = run_chimerax_script(script, executable=exe, timeout_s=240)
            if ok:
                print(f"  rendered: {outputs['chimerax_script'].parent / 'build_zones_preview.png'}", flush=True)
            else:
                print("  render failed (see ChimeraX log)", file=sys.stderr)
        elif args.render:
            print("  --render skipped: ChimeraX not found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
