"""Move work between computers, or save it and pick it up later.

export_project:   zip EVERYTHING of one or more datasets (species list, name check, downloads, sequences, QC,
                  alignments, exclusions, trees, every PCR / qPCR / design / resolution run, reports, the choices
                  made in the app, and the primer library) -> results/exports/<name>_project_<date>.zip
import_project:   restore such a zip. The app then opens the dataset exactly as it was.
export_downloads: zip only the name check and the GenBank downloads of a dataset (to work offline).
import_downloads: unpack such a zip.
import_any:       import either kind of zip.
import_sequences: add your own GenBank or FASTA files to the download cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from . import paths, taxonomy
from .utils import get_logger, read_fasta, write_fasta

ALLOWED = ("data/taxonomy/", "data/genbank/", "data/imported/", "data/species_lists/")

PROJECT_FORMAT = "ispcr-project"
PROJECT_VERSION = 1
MANIFEST = "ispcr_project.json"
# folders that belong to one dataset: <base>/<dataset>/...
DATASET_DIRS = ("data/taxonomy", "data/sequences", "data/qc", "data/alignments",
                "results/pcr", "results/qpcr", "results/qpcr_design", "results/resolution", "results/design", "results/reports")
LIBRARY_FILES = ("primers.yaml", "qpcr_assays.yaml")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dataset_files(dataset: str, downloads: bool = True) -> list[Path]:
    """Every file in the data folder that belongs to `dataset`."""
    root, ds = paths.ROOT, paths.safe(dataset)
    files = []
    for base in DATASET_DIRS:
        d = root / base / ds
        if d.exists():
            files += [f for f in d.rglob("*") if f.is_file() and f.name != ".DS_Store"]
    files += [f for f in (root / "data" / "species_lists").glob(f"{ds}.*") if f.is_file()]
    if downloads:
        try:
            species = {paths.safe(s) for s in taxonomy.load(dataset)["accepted_name"]}
        except Exception:
            species = set()
        for base in ("genbank", "imported"):
            files += [f for f in (root / "data" / base).glob("*/*") if f.is_file() and f.stem in species]
    return sorted(set(files))


def export_project(datasets: str | list[str], downloads: bool = True, out: Path | None = None) -> Path:
    """One zip with everything needed to reopen these datasets as they are now (here or on another computer).
    downloads=False leaves out the GenBank cache (smaller; it can be downloaded again)."""
    datasets = [datasets] if isinstance(datasets, str) else list(datasets)
    if not datasets:
        raise ValueError("No dataset to export")
    root = paths.ROOT
    stem = paths.safe(datasets[0]) if len(datasets) == 1 else f"{len(datasets)}_datasets"
    out = out or paths.exports_dir() / f"{stem}_project{'' if downloads else '_light'}_{time.strftime('%Y%m%d_%H%M')}.zip"
    files = sorted({f for ds in datasets for f in _dataset_files(ds, downloads)})
    checksums = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            b = f.read_bytes()
            name = f.relative_to(root).as_posix()
            z.writestr(name, b)
            checksums[name] = _sha(b)
        for lib in LIBRARY_FILES:
            f = paths.config_dir() / lib
            if f.exists():
                b = f.read_bytes()
                z.writestr(f"library/{lib}", b)
                checksums[f"library/{lib}"] = _sha(b)
        from . import __version__
        z.writestr(MANIFEST, json.dumps({
            "format": PROJECT_FORMAT, "version": PROJECT_VERSION, "app_version": __version__,
            "created": time.strftime("%Y-%m-%d %H:%M"), "datasets": [paths.safe(d) for d in datasets],
            "downloads": downloads, "files": checksums}, indent=1))
    get_logger("bundle").info("Exported project %s (%d files)", out, len(files))
    return out


def read_manifest(data: bytes | Path) -> dict | None:
    """The manifest of a project zip, or None for a downloads-only bundle."""
    zf = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    if MANIFEST not in zf.namelist():
        return None
    return json.loads(zf.read(MANIFEST))


def _merge_library(name: str, b: bytes) -> list[str]:
    """Add primer pairs / assays from a project to your library. Names you already have are kept as they are."""
    import yaml
    incoming = yaml.safe_load(b.decode()) or {}
    dst = paths.config_dir() / name
    text = dst.read_text() if dst.exists() else b.decode()
    have = (yaml.safe_load(text) or {}) if dst.exists() else {}
    added = [k for k in incoming if k not in have]
    if added or not dst.exists():
        head = "".join(line for line in text.splitlines(True) if line.startswith("#"))
        have.update({k: incoming[k] for k in added})
        dst.write_text(head + (yaml.safe_dump(have, sort_keys=False) if have else ""))
    return added


def import_project(data: bytes | Path, replace: bool = True) -> dict:
    """Restore a zip made by export_project. replace=True first removes what is here for the same datasets
    (after saving it to results/exports/<dataset>_before_import_<date>.zip), so the result is exactly what was exported."""
    log = get_logger("bundle")
    zf = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    man = read_manifest(data)
    if not man or man.get("format") != PROJECT_FORMAT:
        raise ValueError("This is not an ispcr project file")
    if int(man.get("version", 0)) > PROJECT_VERSION:
        raise ValueError(f"This project was made by a newer ispcr (format {man['version']}). Update the app first.")
    datasets = [paths.safe(d) for d in man.get("datasets", [])]
    backups = []
    if replace:
        for ds in datasets:
            if _dataset_files(ds, downloads=False):
                backups.append(str(export_project(ds, downloads=False,
                                                  out=paths.exports_dir() / f"{ds}_before_import_{time.strftime('%Y%m%d_%H%M%S')}.zip")))
                for base in DATASET_DIRS:
                    shutil.rmtree(paths.ROOT / base / ds, ignore_errors=True)
    written, skipped, corrupt, added = 0, [], [], []
    sums = man.get("files", {})
    for m in zf.infolist():
        name = m.filename
        if m.is_dir() or name == MANIFEST:
            continue
        parts = Path(name).parts
        if ".." in parts or name.startswith("/") or "\\" in name:
            skipped.append(name)
            continue
        b = zf.read(m)
        if name in sums and _sha(b) != sums[name]:
            corrupt.append(name)
            continue
        if name.startswith("library/") and parts[-1] in LIBRARY_FILES:
            added += _merge_library(parts[-1], b)
            continue
        ok = name.startswith(("data/genbank/", "data/imported/", "data/species_lists/")) or (
            len(parts) >= 3 and "/".join(parts[:2]) in DATASET_DIRS and parts[2] in datasets)
        if not ok:
            skipped.append(name)
            continue
        dest = paths.ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b)
        written += 1
    log.info("Imported project %s: %d files, %d skipped, %d corrupt, %d library entries added",
             ", ".join(datasets), written, len(skipped), len(corrupt), len(added))
    return {"files": written, "skipped": skipped, "corrupt": corrupt, "datasets": datasets,
            "library_added": added, "backups": backups, "created": man.get("created"), "kind": "project"}


def import_any(data: bytes | Path, replace: bool = True) -> dict:
    """Import a project zip (export_project) or a downloads-only bundle (export_downloads)."""
    if read_manifest(data):
        return import_project(data, replace=replace)
    res = import_downloads(data)
    res.update(kind="downloads", corrupt=[], library_added=[], backups=[])
    return res


def export_downloads(dataset: str) -> Path:
    tax = taxonomy.load(dataset)
    species = {paths.safe(s) for s in tax["accepted_name"]}
    out = paths.exports_dir() / f"{paths.safe(dataset)}_downloads_{time.strftime('%Y%m%d_%H%M')}.zip"
    root = paths.ROOT
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (root / "data" / "taxonomy" / paths.safe(dataset)).glob("*"):
            z.write(f, f.relative_to(root))
        for lst in (root / "data" / "species_lists").glob(f"{paths.safe(dataset)}.*"):
            z.write(lst, lst.relative_to(root))
        for base in ("genbank", "imported"):
            for f in (root / "data" / base).glob("*/*"):
                if f.stem in species:
                    z.write(f, f.relative_to(root))
        z.writestr("ispcr_bundle.json", json.dumps({"dataset": dataset, "created": time.strftime("%Y-%m-%d %H:%M")}))
    get_logger("bundle").info("Exported %s", out)
    return out


def import_downloads(data: bytes | Path) -> dict:
    """Unpack a bundle made by export_downloads. Existing files with the same name are replaced."""
    log = get_logger("bundle")
    zf = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    written, skipped = 0, []
    datasets = set()
    for m in zf.infolist():
        name = m.filename
        if m.is_dir() or name == "ispcr_bundle.json":
            continue
        if ".." in Path(name).parts or name.startswith("/") or not name.startswith(ALLOWED):
            skipped.append(name)
            continue
        dest = paths.ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(m))
        written += 1
        if name.startswith("data/taxonomy/"):
            datasets.add(Path(name).parts[2])
    log.info("Imported %d files (%d skipped)", written, len(skipped))
    return {"files": written, "skipped": skipped, "datasets": sorted(datasets)}


def _match_species(text: str, names: dict[str, str]) -> str | None:
    t = re.sub(r"[_\s]+", " ", text).lower()
    for key, sp in names.items():
        if key in t:
            return sp
    return None


def import_sequences(dataset: str, target: str, filename: str, data: bytes) -> pd.DataFrame:
    """Add GenBank (.gb/.gbk) or FASTA records to the cache for `target`. Each record is assigned to a
    species of the dataset by its organism (GenBank) or by a species name in its header (FASTA)."""
    log = get_logger("bundle")
    tax = taxonomy.load(dataset)
    names = {}
    for r in tax.to_dict("records"):
        for k in ("accepted_name", "input_name", "ncbi_name"):
            if isinstance(r.get(k), str):
                names[r[k].lower()] = r["accepted_name"]
    names = dict(sorted(names.items(), key=lambda kv: -len(kv[0])))
    text = data.decode(errors="ignore")
    is_gb = filename.lower().endswith((".gb", ".gbk", ".genbank", ".gbff")) or text.lstrip().startswith("LOCUS")
    rows, by_sp = [], {}
    if is_gb:
        for rec in SeqIO.parse(io.StringIO(text), "genbank"):
            sp = _match_species(rec.annotations.get("organism", "") + " " + rec.description, names)
            rows.append({"record": rec.id, "species": sp, "length": len(rec.seq)})
            if sp:
                by_sp.setdefault(sp, []).append(rec)
        for sp, recs in by_sp.items():
            gb = paths.genbank_dir(target) / f"{paths.safe(sp)}.gb"
            meta = gb.with_suffix(".json")
            old = list(SeqIO.parse(str(gb), "genbank")) if gb.exists() and gb.stat().st_size else []
            have = {r.id for r in old}
            merged = old + [r for r in recs if r.id not in have]
            with open(gb, "w") as fh:
                SeqIO.write(merged, fh, "genbank")
            m = json.loads(meta.read_text()) if meta.exists() else {"species": sp, "query": None, "max": 0}
            m.update(accessions=[r.id for r in merged], imported=True, date=time.strftime("%Y-%m-%d %H:%M"))
            meta.write_text(json.dumps(m, indent=1))
    else:
        tmp = paths.exports_dir() / "_import.fasta"
        tmp.write_text(text)
        fasta = read_fasta(tmp)
        tmp.unlink()
        headers = {n: line for n, line in zip([n for n, _ in fasta], [l[1:] for l in text.splitlines() if l.startswith(">")])}
        for n, s in fasta:
            sp = _match_species(headers.get(n, n), names)
            rows.append({"record": n, "species": sp, "length": len(s)})
            if sp:
                by_sp.setdefault(sp, []).append((n, s.upper()))
        for sp, recs in by_sp.items():
            f = paths.imported_dir(target) / f"{paths.safe(sp)}.fasta"
            old = read_fasta(f) if f.exists() else []
            have = {n for n, _ in old}
            write_fasta(f, old + [r for r in recs if r[0] not in have])
    df = pd.DataFrame(rows)
    log.info("Imported %d of %d records from %s into %s", int(df["species"].notna().sum()) if len(df) else 0, len(df), filename, target)
    return df
