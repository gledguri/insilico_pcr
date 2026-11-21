"""What was chosen in each step of the app, saved per dataset so the PDF report (step 10) and the next
session start from the same choices. File: results/reports/<dataset>/selections.json

Layout:
  {"mode": "pcr" | "qpcr", "marker": "12S", "mito_species": ..., ...,
   "markers": {"12S": {"pcr_run": ..., "compare_runs": [...], ...}, "COI": {...}}}
"""
from __future__ import annotations

import json

from . import paths

GLOBAL_KEYS = {"mode", "marker", "page", "mito_species", "mito_record", "mito_pairs", "mito_assays", "report_sections",
               "mito_view", "mito_highlight", "mito_gc", "mito_trna"}


def load(dataset: str) -> dict:
    p = paths.selections_file(dataset)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def save(dataset: str, data: dict) -> None:
    p = paths.selections_file(dataset)
    txt = json.dumps(data, indent=1, sort_keys=True, default=str)
    if not p.exists() or p.read_text() != txt:
        p.write_text(txt)


def get(dataset: str, key: str, marker: str | None = None, default=None):
    d = load(dataset)
    if key in GLOBAL_KEYS or marker is None:
        return d.get(key, default)
    return d.get("markers", {}).get(marker, {}).get(key, default)


def put(dataset: str, key: str, value, marker: str | None = None) -> None:
    d = load(dataset)
    if key in GLOBAL_KEYS or marker is None:
        d[key] = value
    else:
        d.setdefault("markers", {}).setdefault(marker, {})[key] = value
    save(dataset, d)


def mode(dataset: str) -> str:
    return get(dataset, "mode", default="pcr") or "pcr"


def _state_file():
    return paths.config_dir() / "app_state.json"


def last_dataset() -> str | None:
    """The dataset that was open last time (stored in the data folder)."""
    try:
        return json.loads(_state_file().read_text()).get("dataset")
    except (OSError, ValueError):
        return None


def set_last_dataset(dataset: str) -> None:
    if dataset and dataset != last_dataset():
        _state_file().write_text(json.dumps({"dataset": dataset}))
