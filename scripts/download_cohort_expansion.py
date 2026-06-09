"""Download cohort expansion maps and PDB models from EMDB / RCSB.

Manifests (FTP-verified 2026-06-07):
  - ``cohort/expansion_downloads.json`` — 7 conformation pairs (14 maps)
  - ``cohort/publication_downloads.json`` — 12 single-map benchmark entries

Example::

    source .venv/bin/activate
    cd /Users/sarthakmohanty/Developer/thesis

    python scripts/download_cohort_expansion.py --verify-only --everything
    python scripts/download_cohort_expansion.py --everything
    python scripts/download_cohort_expansion.py --publication --all
    python scripts/download_cohort_expansion.py --pair ribosome_ecoli_pre_post
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from cryoem_mrc.repo_paths import DATA_ROOT, PDB_ROOT

REPO = Path(__file__).resolve().parents[1]
EXPANSION_MANIFEST = REPO / "cohort" / "expansion_downloads.json"
PUBLICATION_MANIFEST = REPO / "cohort" / "publication_downloads.json"
EXPANSION_REPORT = REPO / "cohort" / "expansion_download_report.json"
PUBLICATION_REPORT = REPO / "cohort" / "publication_download_report.json"

EMDB_MIRRORS = (
    "https://files.wwpdb.org/pub/emdb/structures",
    "https://ftp.ebi.ac.uk/pub/emdb/structures",
)
RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif"

MAP_FILES = (
    ("primary", "map/emd_{emdb_id}.map.gz"),
    ("half1", "other/emd_{emdb_id}_half_map_1.map.gz"),
    ("half2", "other/emd_{emdb_id}_half_map_2.map.gz"),
)


@dataclass
class FileResult:
    name: str
    url: str
    path: Path
    status: str
    bytes: int = 0
    error: str = ""


@dataclass
class EntryResult:
    emdb_id: str
    folder: str
    files: list[FileResult] = field(default_factory=list)
    pdb: FileResult | None = None

    @property
    def ok(self) -> bool:
        maps_ok = all(f.status in ("ok", "skipped", "available", "would_download") for f in self.files)
        pdb_ok = self.pdb is None or self.pdb.status in ("ok", "skipped", "available", "would_download")
        return maps_ok and pdb_ok and not any(f.status in ("failed", "missing") for f in self.files)


def _url_exists(url: str, timeout: int = 30) -> tuple[bool, int]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            if resp.status not in (200, 206):
                return False, 0
            total = resp.headers.get("Content-Range", "")
            if total and "/" in total:
                try:
                    return True, int(total.rsplit("/", 1)[-1])
                except ValueError:
                    pass
            cl = resp.headers.get("Content-Length")
            if cl:
                try:
                    return True, int(cl)
                except ValueError:
                    pass
            return True, 0
    except urllib.error.HTTPError as exc:
        if exc.code in (200, 206):
            return True, 0
        return False, 0
    except OSError:
        return False, 0


def _resolve_emdb_url(emdb_id: str, rel: str) -> tuple[str | None, int]:
    emdb_id = str(emdb_id).strip()
    rel = rel.format(emdb_id=emdb_id)
    for mirror in EMDB_MIRRORS:
        url = f"{mirror}/EMD-{emdb_id}/{rel}"
        ok, size = _url_exists(url)
        if ok:
            return url, size
    return None, 0


def _download(url: str, dest: Path, timeout: int = 600) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)
    return dest.stat().st_size


def _gunzip(path: Path) -> Path:
    out = path.with_suffix("")
    if out.is_file() and out.stat().st_size > 0:
        return out
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out


def _load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _select_pairs(data: dict, *, pair_id: str | None, all_pairs: bool) -> list[dict]:
    pairs = data.get("pairs", [])
    if pair_id:
        out = [p for p in pairs if p.get("id") == pair_id]
        if not out:
            raise SystemExit(f"Unknown pair id: {pair_id!r}. Choices: {[p['id'] for p in pairs]}")
        return out
    if all_pairs:
        return pairs
    raise SystemExit("Specify --all or --pair <id>")


def _select_publication(data: dict, *, emdb_id: str | None, all_entries: bool) -> list[dict]:
    entries = data.get("entries", [])
    if emdb_id:
        out = [e for e in entries if str(e["emdb_id"]) == emdb_id.strip()]
        if not out:
            raise SystemExit(f"Unknown emdb_id: {emdb_id!r}")
        return out
    if all_entries:
        return entries
    raise SystemExit("Specify --all or --emdb-id <id>")


def verify_entry(state: dict) -> EntryResult:
    emdb_id = str(state["emdb_id"])
    folder = state["folder"]
    result = EntryResult(emdb_id=emdb_id, folder=folder)

    for name, rel in MAP_FILES:
        url, _size = _resolve_emdb_url(emdb_id, rel)
        path = DATA_ROOT / folder / rel.format(emdb_id=emdb_id).split("/")[-1]
        if url:
            result.files.append(FileResult(name=name, url=url, path=path, status="available"))
        else:
            result.files.append(
                FileResult(name=name, url="", path=path, status="missing", error="not on EMDB FTP mirrors")
            )

    pdb_id = state.get("pdb_id")
    if pdb_id:
        pdb_id = str(pdb_id).lower()
        url = RCSB_CIF.format(pdb_id=pdb_id)
        ok, _ = _url_exists(url)
        path = PDB_ROOT / f"{pdb_id}.cif"
        result.pdb = FileResult(
            name="pdb",
            url=url if ok else "",
            path=path,
            status="available" if ok else "missing",
            error="" if ok else "PDB not found on RCSB",
        )
    return result


def download_entry(state: dict, *, gunzip: bool, skip_existing: bool, dry_run: bool) -> EntryResult:
    emdb_id = str(state["emdb_id"])
    folder = state["folder"]
    data_dir = DATA_ROOT / folder
    result = EntryResult(emdb_id=emdb_id, folder=folder)

    for name, rel in MAP_FILES:
        fname = rel.format(emdb_id=emdb_id).split("/")[-1]
        dest_gz = data_dir / fname
        dest_map = dest_gz.with_suffix("")
        final_path = dest_map if gunzip else dest_gz

        if skip_existing and final_path.is_file() and final_path.stat().st_size > 0:
            result.files.append(
                FileResult(name=name, url="", path=final_path, status="skipped", bytes=final_path.stat().st_size)
            )
            continue

        url, _ = _resolve_emdb_url(emdb_id, rel)
        if not url:
            result.files.append(
                FileResult(name=name, url="", path=dest_gz, status="failed", error="not on EMDB FTP mirrors")
            )
            continue

        if dry_run:
            result.files.append(FileResult(name=name, url=url, path=dest_gz, status="would_download"))
            continue

        try:
            nbytes = _download(url, dest_gz)
            out_path = _gunzip(dest_gz) if gunzip else dest_gz
            result.files.append(FileResult(name=name, url=url, path=out_path, status="ok", bytes=nbytes))
            print(f"  [ok] EMD-{emdb_id} {name} -> {out_path}", flush=True)
        except OSError as exc:
            result.files.append(FileResult(name=name, url=url, path=dest_gz, status="failed", error=str(exc)))
            print(f"  [FAIL] EMD-{emdb_id} {name}: {exc}", flush=True)

    pdb_id = state.get("pdb_id")
    if pdb_id:
        pdb_id = str(pdb_id).lower()
        dest = PDB_ROOT / f"{pdb_id}.cif"
        url = RCSB_CIF.format(pdb_id=pdb_id)

        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            result.pdb = FileResult(name="pdb", url=url, path=dest, status="skipped", bytes=dest.stat().st_size)
        elif dry_run:
            result.pdb = FileResult(name="pdb", url=url, path=dest, status="would_download")
        else:
            try:
                nbytes = _download(url, dest)
                result.pdb = FileResult(name="pdb", url=url, path=dest, status="ok", bytes=nbytes)
                print(f"  [ok] PDB {pdb_id} -> {dest}", flush=True)
            except OSError as exc:
                result.pdb = FileResult(name="pdb", url=url, path=dest, status="failed", error=str(exc))
                print(f"  [FAIL] PDB {pdb_id}: {exc}", flush=True)

    return result


def _serialize_result(entry: EntryResult) -> dict:
    def fr(f: FileResult | None) -> dict | None:
        if f is None:
            return None
        return {
            "name": f.name,
            "status": f.status,
            "path": str(f.path),
            "url": f.url,
            "bytes": f.bytes,
            "error": f.error,
        }

    return {
        "emdb_id": entry.emdb_id,
        "folder": entry.folder,
        "ok": entry.ok,
        "files": [fr(f) for f in entry.files],
        "pdb": fr(entry.pdb),
    }


def _tally(entry: EntryResult) -> tuple[int, int, int]:
    ok = fail = skip = 0
    for f in entry.files:
        if f.status in ("ok", "available", "would_download"):
            ok += 1
        elif f.status == "skipped":
            skip += 1
        elif f.status in ("failed", "missing"):
            fail += 1
    if entry.pdb:
        if entry.pdb.status in ("ok", "available", "would_download"):
            ok += 1
        elif entry.pdb.status == "skipped":
            skip += 1
        elif entry.pdb.status in ("failed", "missing"):
            fail += 1
    return ok, skip, fail


def _run_expansion(args: argparse.Namespace) -> tuple[dict, int, int, int]:
    data = _load_manifest(args.manifest or EXPANSION_MANIFEST)
    pairs = _select_pairs(data, pair_id=args.pair, all_pairs=args.all or args.everything)

    print(f"Expansion manifest: {args.manifest or EXPANSION_MANIFEST}", flush=True)
    print(f"Pairs selected: {len(pairs)}", flush=True)

    report: dict = {"pairs": [], "summary": {}}
    n_ok = n_fail = n_skip = 0
    dry_run = args.dry_run or args.verify_only
    skip_existing = not args.no_skip_existing
    gunzip = not args.no_gunzip

    for pair in pairs:
        pid = pair["id"]
        print(f"\n=== Pair: {pid} — {pair.get('description', '')}", flush=True)
        pair_report = {"id": pid, "entries": []}

        for state in pair["states"]:
            emdb_id = state["emdb_id"]
            print(f"EMD-{emdb_id} -> data/{state['folder']}/", flush=True)
            entry = (
                verify_entry(state)
                if args.verify_only
                else download_entry(state, gunzip=gunzip, skip_existing=skip_existing, dry_run=dry_run)
            )
            pair_report["entries"].append(_serialize_result(entry))
            o, s, f = _tally(entry)
            n_ok += o
            n_skip += s
            n_fail += f

        report["pairs"].append(pair_report)

    report["summary"] = {"ok_or_available": n_ok, "skipped": n_skip, "failed_or_missing": n_fail}
    return report, n_ok, n_skip, n_fail


def _run_publication(args: argparse.Namespace) -> tuple[dict, int, int, int]:
    path = args.publication_manifest or PUBLICATION_MANIFEST
    data = _load_manifest(path)
    entries = _select_publication(
        data,
        emdb_id=args.emdb_id,
        all_entries=args.all or args.everything,
    )

    print(f"Publication manifest: {path}", flush=True)
    print(f"Entries selected: {len(entries)}", flush=True)

    report: dict = {"entries": [], "summary": {}}
    n_ok = n_fail = n_skip = 0
    dry_run = args.dry_run or args.verify_only
    skip_existing = not args.no_skip_existing
    gunzip = not args.no_gunzip

    for state in entries:
        emdb_id = state["emdb_id"]
        print(f"\n=== EMD-{emdb_id} ({state.get('gap', '')}) — {state.get('label', '')}", flush=True)
        print(f"data/{state['folder']}/", flush=True)
        entry = (
            verify_entry(state)
            if args.verify_only
            else download_entry(state, gunzip=gunzip, skip_existing=skip_existing, dry_run=dry_run)
        )
        report["entries"].append({**state, "download": _serialize_result(entry)})
        o, s, f = _tally(entry)
        n_ok += o
        n_skip += s
        n_fail += f

    report["summary"] = {"ok_or_available": n_ok, "skipped": n_skip, "failed_or_missing": n_fail}
    return report, n_ok, n_skip, n_fail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download cohort expansion / publication maps and PDB models.")
    p.add_argument("--manifest", type=Path, default=None, help="Conformation-pair manifest (default: expansion_downloads.json)")
    p.add_argument("--publication-manifest", type=Path, default=None)
    p.add_argument("--expansion", action="store_true", help="Conformation-pair expansion (default if neither flag set with --all)")
    p.add_argument("--publication", action="store_true", help="Single-map publication cohort")
    p.add_argument("--everything", action="store_true", help="Both expansion pairs and publication singles")
    p.add_argument("--all", action="store_true", help="All items in the selected manifest(s)")
    p.add_argument("--pair", type=str, help="Expansion: one pair id")
    p.add_argument("--emdb-id", type=str, help="Publication: one EMDB id")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--no-gunzip", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.expansion and not args.publication and not args.everything:
        if args.pair:
            args.expansion = True
        elif args.emdb_id:
            args.publication = True
        elif args.all:
            args.expansion = True
        else:
            raise SystemExit("Specify --expansion, --publication, or --everything (with --all or filters).")

    total_fail = 0
    if args.expansion or args.everything:
        report, ok, skip, fail = _run_expansion(args)
        total_fail += fail
        if not args.verify_only and not args.dry_run:
            EXPANSION_REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(f"\nExpansion report: {EXPANSION_REPORT}", flush=True)
        print(f"Expansion: {ok} ok/available, {skip} skipped, {fail} failed/missing", flush=True)

    if args.publication or args.everything:
        report, ok, skip, fail = _run_publication(args)
        total_fail += fail
        if not args.verify_only and not args.dry_run:
            PUBLICATION_REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(f"\nPublication report: {PUBLICATION_REPORT}", flush=True)
        print(f"Publication: {ok} ok/available, {skip} skipped, {fail} failed/missing", flush=True)

    if total_fail:
        print("Some files missing — see cohort/*_downloads.json for ID notes.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
