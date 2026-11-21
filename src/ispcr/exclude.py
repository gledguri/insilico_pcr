"""Records you exclude by hand (e.g. misannotated sequences).

Stored per dataset and marker in data/alignments/<dataset>/<marker>/excluded.csv.
Excluded records stay in the alignment file but are left out of the alignment view
(unless shown on purpose), the tree, in silico PCR, resolution statistics and primer design.
"""
from __future__ import annotations

import time

import pandas as pd

from . import paths

COLS = ["seq_id", "species", "reason", "excluded_on"]


def _path(dataset: str, marker: str):
    return paths.alignment_dir(dataset, marker) / "excluded.csv"


def load(dataset: str, marker: str) -> pd.DataFrame:
    p = _path(dataset, marker)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLS)


def ids(dataset: str, marker: str) -> set[str]:
    return set(load(dataset, marker)["seq_id"])


def add(dataset: str, marker: str, rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with seq_id, species, reason."""
    df = load(dataset, marker)
    new = pd.DataFrame([{**r, "excluded_on": time.strftime("%Y-%m-%d %H:%M")} for r in rows], columns=COLS)
    df = pd.concat([df[~df["seq_id"].isin(new["seq_id"])], new], ignore_index=True)
    df.to_csv(_path(dataset, marker), index=False)
    _refresh_columns(dataset, marker)
    return df


def remove(dataset: str, marker: str, seq_ids: list[str]) -> pd.DataFrame:
    df = load(dataset, marker)
    df = df[~df["seq_id"].isin(seq_ids)]
    df.to_csv(_path(dataset, marker), index=False)
    _refresh_columns(dataset, marker)
    return df


def _refresh_columns(dataset: str, marker: str) -> None:
    """Recompute per-column variability without the excluded records."""
    from .stats import alignment_matrix, column_profile
    from .utils import read_fasta
    a = paths.alignment_dir(dataset, marker)
    if not (a / "aligned.fasta").exists():
        return
    skip = ids(dataset, marker)
    recs = [s for n, s in read_fasta(a / "aligned.fasta") if n not in skip]
    if recs:
        column_profile(alignment_matrix(recs)).to_csv(a / "columns.csv", index=False)
