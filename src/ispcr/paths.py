"""Folder layout. The code (this repository) and your data live in separate places.

CODE_ROOT  the repository: app/, src/, config/ (default settings and the published primer library)
ROOT       the data folder. Every file ispcr writes goes somewhere below it:

ROOT/
  config/                         your primer library (primers.yaml, qpcr_assays.yaml), optional settings.yaml overrides
  data/
    species_lists/                your input lists (one Latin name per line)
    taxonomy/<dataset>/           species_resolved.csv
    genbank/<target>/             cached GenBank downloads, one .gb + .json per species
    imported/<target>/            FASTA files you imported, one per species
    sequences/<dataset>/<marker>/ sequences.fasta, metadata.csv (extracted, before QC)
    qc/<dataset>/<marker>/        clean.fasta, clean_metadata.csv, removed.csv, species_counts.csv
    alignments/<dataset>/<marker>/aligned.fasta, columns.csv, misid_flags.csv, excluded.csv, trees/
  results/
    pcr/<dataset>/<marker>/<primer>/         hits.csv, species_summary.csv, mismatch_*.csv, amplicons*.fasta
    qpcr/<dataset>/<marker>/<assay>/         species-specific assay check: hits.csv, species_summary.csv, checks.csv
    qpcr_design/<dataset>/<marker>/<run_id>/ designed species-specific assays (assays.csv)
    resolution/<dataset>/<marker>/<primer>/  barcode_gap.csv, distances, tree.nwk
    design/<dataset>/<marker>/<run_id>/      candidates and evaluated pairs
    reports/<dataset>/<marker>/              HTML and PDF reports (selections.json: choices made in the app)
    exports/                                 project and download bundles (.zip)
  logs/                           one log file per command

Where the data folder is (first match wins):
  1. environment variable ISPCR_DATA (or the older ISPCR_ROOT)
  2. data_dir in ispcr.local.yaml next to this repository (written by the app's "Data folder" setting)
  3. workspace/ inside the repository
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_FILE = CODE_ROOT / "ispcr.local.yaml"


def _local() -> dict:
    if not LOCAL_FILE.exists():
        return {}
    import yaml
    try:
        return yaml.safe_load(LOCAL_FILE.read_text()) or {}
    except Exception:
        return {}


def data_root_source() -> str:
    """Where the data folder setting comes from: 'env', 'local' (ispcr.local.yaml) or 'default'."""
    if os.environ.get("ISPCR_DATA") or os.environ.get("ISPCR_ROOT"):
        return "env"
    return "local" if _local().get("data_dir") else "default"


def _resolve_root() -> Path:
    env = os.environ.get("ISPCR_DATA") or os.environ.get("ISPCR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    d = _local().get("data_dir")
    if d:
        return Path(str(d)).expanduser().resolve()
    return CODE_ROOT / "workspace"


ROOT = _resolve_root()


def is_hosted() -> bool:
    """True on a shared server (Streamlit Community Cloud, or ISPCR_HOSTED=1): there the data folder can't be changed."""
    return bool(os.environ.get("ISPCR_HOSTED")) or str(CODE_ROOT).startswith("/mount/src")


def set_root(folder: str | Path, remember: bool = True) -> Path:
    """Use another data folder from now on. remember=True writes it to ispcr.local.yaml for the next start."""
    global ROOT
    ROOT = Path(folder).expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    if remember:
        import yaml
        cfg = _local()
        cfg["data_dir"] = str(ROOT)
        LOCAL_FILE.write_text("# Local settings for this computer (not in git).\n" + yaml.safe_dump(cfg, sort_keys=False))
    init_layout()
    import logging  # log files are opened per logger: reopen them in the new folder
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("ispcr.") and isinstance(lg, logging.Logger):
            for h in list(lg.handlers):
                lg.removeHandler(h)
                h.close()
    from . import config
    config.settings.cache_clear()
    return ROOT


def safe(name: str) -> str:
    """File-system friendly version of a name ('Gadus morhua' -> 'Gadus_morhua')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")


def _d(*parts: str) -> Path:
    p = ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_dir() -> Path:
    """Your primer library and setting overrides (in the data folder)."""
    return _d("config")


def default_config_dir() -> Path:
    """Default settings and the published primer library (in the repository)."""
    return CODE_ROOT / "config"


def species_lists_dir() -> Path:
    return _d("data", "species_lists")


def taxonomy_dir(dataset: str) -> Path:
    return _d("data", "taxonomy", safe(dataset))


def genbank_dir(target: str) -> Path:
    return _d("data", "genbank", safe(target))


def sequences_dir(dataset: str, marker: str) -> Path:
    return _d("data", "sequences", safe(dataset), safe(marker))


def qc_dir(dataset: str, marker: str) -> Path:
    return _d("data", "qc", safe(dataset), safe(marker))


def alignment_dir(dataset: str, marker: str) -> Path:
    return _d("data", "alignments", safe(dataset), safe(marker))


def pcr_dir(dataset: str, marker: str, primer: str) -> Path:
    return _d("results", "pcr", safe(dataset), safe(marker), safe(primer))


def resolution_dir(dataset: str, marker: str, primer: str) -> Path:
    return _d("results", "resolution", safe(dataset), safe(marker), safe(primer))


def design_dir(dataset: str, marker: str, run_id: str) -> Path:
    return _d("results", "design", safe(dataset), safe(marker), safe(run_id))


def qpcr_dir(dataset: str, marker: str, assay: str) -> Path:
    return _d("results", "qpcr", safe(dataset), safe(marker), safe(assay))


def qpcr_design_dir(dataset: str, marker: str, run_id: str) -> Path:
    return _d("results", "qpcr_design", safe(dataset), safe(marker), safe(run_id))


def selections_file(dataset: str) -> Path:
    """What was chosen in each step of the app (used by the PDF report)."""
    return _d("results", "reports", safe(dataset)) / "selections.json"


def report_dir(dataset: str, marker: str) -> Path:
    return _d("results", "reports", safe(dataset), safe(marker))


def imported_dir(target: str) -> Path:
    """FASTA files you imported yourself (no GenBank annotation), one file per species."""
    return _d("data", "imported", safe(target))


def exports_dir() -> Path:
    return _d("results", "exports")


def trees_dir(dataset: str, marker: str) -> Path:
    return _d("data", "alignments", safe(dataset), safe(marker), "trees")


def logs_dir() -> Path:
    return _d("logs")


def list_datasets() -> list[str]:
    p = ROOT / "data" / "taxonomy"
    return sorted(x.name for x in p.iterdir() if x.is_dir()) if p.exists() else []


def list_markers(dataset: str, stage: str = "qc") -> list[str]:
    p = ROOT / "data" / stage / safe(dataset)
    return sorted(x.name for x in p.iterdir() if x.is_dir()) if p.exists() else []


def list_pcr_runs(dataset: str, marker: str) -> list[str]:
    p = ROOT / "results" / "pcr" / safe(dataset) / safe(marker)
    return sorted(x.name for x in p.iterdir() if (x / "hits.csv").exists()) if p.exists() else []


def list_design_runs(dataset: str, marker: str) -> list[str]:
    p = ROOT / "results" / "design" / safe(dataset) / safe(marker)
    return sorted((x.name for x in p.iterdir() if x.is_dir()), reverse=True) if p.exists() else []


def list_qpcr_runs(dataset: str, marker: str) -> list[str]:
    p = ROOT / "results" / "qpcr" / safe(dataset) / safe(marker)
    return sorted(x.name for x in p.iterdir() if (x / "summary.json").exists()) if p.exists() else []


def list_qpcr_design_runs(dataset: str, marker: str) -> list[str]:
    p = ROOT / "results" / "qpcr_design" / safe(dataset) / safe(marker)
    return sorted((x.name for x in p.iterdir() if (x / "assays.csv").exists()), reverse=True) if p.exists() else []


def init_layout() -> None:
    """Create the top-level folders so the structure is visible from the start."""
    for parts in [
        ("config",), ("data", "species_lists"), ("data", "taxonomy"), ("data", "genbank"),
        ("data", "sequences"), ("data", "qc"), ("data", "alignments"), ("data", "imported"),
        ("results", "pcr"), ("results", "qpcr"), ("results", "qpcr_design"), ("results", "resolution"), ("results", "design"), ("results", "reports"),
        ("results", "exports"),
        ("logs",),
    ]:
        _d(*parts)
    # start your primer library from the published one in the repository
    for name in ("primers.yaml", "qpcr_assays.yaml"):
        src, dst = default_config_dir() / name, ROOT / "config" / name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
