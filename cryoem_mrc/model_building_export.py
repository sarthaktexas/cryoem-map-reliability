"""Export build-zone coloring for ChimeraX and Coot during model building."""

from __future__ import annotations

from pathlib import Path

from .repo_paths import COHORT_MANIFEST, emd_output_dir, lh_map_reliability_dir
from .structure_validation import ResidueValidationRow, read_residue_validation_csv

ZONE_COLORS = {
    0: "#e74c3c",  # omit
    1: "#f1c40f",  # caution
    2: "#27ae60",  # build
}

ZONE_LABELS = {
    0: "omit",
    1: "caution",
    2: "build",
}


def _quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _residue_select_token(row: ResidueValidationRow) -> str:
    chain = (row.auth_chain or row.chain).strip()
    seq = row.auth_seq_num or row.seq_num
    if row.seq_icode and row.seq_icode.strip():
        return f"/{chain}:{seq}{row.seq_icode.strip()}"
    return f"/{chain}:{seq}"


def write_chimerax_build_zone_script(
    *,
    structure_path: Path,
    reference_mrc: Path,
    rows: list[ResidueValidationRow],
    out_script: Path,
    contour: float,
    out_png: Path | None = None,
    width: int = 900,
    height: int = 900,
) -> Path:
    """ChimeraX script: density shell + cartoon colored by omit/caution/build zones."""
    model = _quote(structure_path)
    ref = _quote(reference_mrc)
    lines = [
        f'open "{ref}" name density',
        f'open "{model}" name model',
        f"volume #1 style surface level {contour:g} step 1 color #bbbbbb transparency 82",
        "cartoon #2",
        "hide #2 atoms",
    ]
    masked = [r for r in rows if r.in_contour_mask]
    for zone in (0, 1, 2):
        tokens = [_residue_select_token(r) for r in masked if r.build_zone == zone]
        if not tokens:
            continue
        # ChimeraX accepts comma-separated residue lists; chunk to avoid command limits.
        chunk_size = 80
        for start in range(0, len(tokens), chunk_size):
            chunk = ",".join(tokens[start : start + chunk_size])
            lines.append(f"select {chunk}")
            lines.append(f"color {ZONE_COLORS[zone]} sel")
            lines.append("select clear")
    lines.extend(
        [
            "set bgcolor white",
            "lighting soft",
            "view orient",
            "turn y 135",
            "turn x 15",
        ]
    )
    if out_png is not None:
        png = _quote(out_png)
        lines.append(f'save "{png}" width {width} height {height} supersample 2')
    lines.append("exit")
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_script


def write_coot_zone_python(
    rows: list[ResidueValidationRow],
    out_path: Path,
    *,
    structure_basename: str = "deposited.cif",
) -> Path:
    """Write a Coot Python snippet to color residues by build zone."""
    masked = [r for r in rows if r.in_contour_mask]
    lines = [
        "# Paste into Coot Python console after loading the deposited model.",
        f"# Structure: {structure_basename}",
        "# Colors: red=omit, yellow=caution, green=build",
        "",
        "zone_rgb = {",
        "    0: (1.0, 0.2, 0.2),",
        "    1: (1.0, 0.85, 0.2),",
        "    2: (0.2, 0.75, 0.35),",
        "}",
        "",
        "def color_build_zones():",
        "    imol = coot_utils.get_first_molecule_with_coordinates()",
        "    if imol < 0:",
        "        print('No coordinates loaded')",
        "        return",
    ]
    for zone in (0, 1, 2):
        residues = [r for r in masked if r.build_zone == zone]
        if not residues:
            continue
        lines.append(f"    # {ZONE_LABELS[zone]} zone ({len(residues)} residues)")
        for r in residues:
            chain = (r.auth_chain or r.chain).replace("'", "\\'")
            seq = r.auth_seq_num or r.seq_num
            lines.append(
                f"    set_residue_colour(imol, '{chain}', {seq}, '', "
                f"zone_rgb[{zone}][0], zone_rgb[{zone}][1], zone_rgb[{zone}][2])"
            )
    lines.extend(
        [
            "    graphics_draw()",
            "",
            "color_build_zones()",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_reliability_bfactor_pdb(
    structure_path: Path,
    rows: list[ResidueValidationRow],
    out_path: Path,
) -> Path:
    """Copy deposited structure with B_iso = reliability_score * 100 for coloring."""
    import gemmi

    st = gemmi.read_structure(str(structure_path))
    key_to_rel = {
        (r.chain, r.seq_num, r.seq_icode): r.reliability_score for r in rows
    }
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.entity_type == gemmi.EntityType.Water:
                    continue
                key = (
                    chain.name,
                    int(residue.seqid.num),
                    residue.seqid.icode.strip(),
                )
                rel = key_to_rel.get(key)
                if rel is None:
                    continue
                for atom in residue:
                    atom.b_iso = float(rel) * 100.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_path))
    return out_path


def export_model_building_assets(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    contour: float | None = None,
) -> dict[str, Path]:
    """Write ChimeraX script, Coot snippet, and reliability-colored PDB for one map."""
    from .chimerax_figures import resolve_protein_bundle
    from .structure_validation import load_cohort_manifest_row

    row = load_cohort_manifest_row(manifest, emdb_id)
    bundle = resolve_protein_bundle(emdb_id, manifest=manifest)
    rv_path = lh_map_reliability_dir(emdb_id) / "residue_validation.csv"
    if not rv_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id}: missing {rv_path}")
    residues = read_residue_validation_csv(rv_path)

    out_root = emd_output_dir(emdb_id) / "model_building"
    out_root.mkdir(parents=True, exist_ok=True)
    c_level = contour if contour is not None else bundle.contour

    outputs: dict[str, Path] = {}
    outputs["chimerax_script"] = write_chimerax_build_zone_script(
        structure_path=bundle.structure_path,
        reference_mrc=bundle.reference_mrc,
        rows=residues,
        out_script=out_root / "color_build_zones.cxc",
        contour=c_level,
        out_png=out_root / "build_zones_preview.png",
    )
    outputs["coot_script"] = write_coot_zone_python(
        residues,
        out_root / "color_build_zones_coot.py",
        structure_basename=bundle.structure_path.name,
    )
    outputs["reliability_pdb"] = write_reliability_bfactor_pdb(
        bundle.structure_path,
        residues,
        out_root / f"emd_{emdb_id}_reliability_bfactor.pdb",
    )
    readme = out_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                f"Model-building exports for EMD-{emdb_id} ({row.get('display_name', '')})",
                "",
                "ChimeraX:",
                f"  ChimeraX --script {outputs['chimerax_script'].name}",
                "  (or open color_build_zones.cxc from the File menu)",
                "",
                "Coot:",
                f"  File -> Run script -> {outputs['coot_script'].name}",
                "",
                "Reliability B-factor PDB:",
                f"  Load {outputs['reliability_pdb'].name} and color by B-factor",
                "  (100 * reliability_score; higher = more reliable)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs["readme"] = readme
    return outputs
