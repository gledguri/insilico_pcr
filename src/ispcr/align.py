"""Multiple alignment with MAFFT (installed in ./env)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import paths
from .config import settings
from .qc import flag_misidentified
from .stats import alignment_matrix, column_profile
from .utils import get_logger, read_fasta, revcomp, write_fasta


def mafft_bin() -> str:
    local = Path(sys.executable).parent / "mafft"
    if local.exists():
        return str(local)
    found = shutil.which("mafft")
    if not found:
        raise FileNotFoundError("MAFFT not found. Activate the project environment: ./env/bin/...")
    return found


def run_mafft(inp: Path, out: Path, extra: list[str] | None = None, threads: int | None = None) -> list[tuple[str, str]]:
    cfg = settings()["align"]
    args = [mafft_bin(), *(extra if extra is not None else cfg["mafft_args"]),
            "--thread", str(threads if threads is not None else cfg.get("threads", -1)), "--quiet", str(inp)]
    with open(out, "w") as fh:
        subprocess.run(args, stdout=fh, stderr=subprocess.PIPE, check=True, text=True)
    raw = read_fasta(out)
    rev = {n[3:] for n, _ in raw if n.startswith("_R_")}
    recs = [(n[3:] if n.startswith("_R_") else n, s.upper()) for n, s in raw]
    if len(rev) > len(recs) / 2:  # MAFFT follows the first record; keep the majority orientation instead
        recs = [(n, revcomp(s)) for n, s in recs]
        rev = {n for n, _ in recs} - rev
    run_mafft.reversed = rev
    write_fasta(out, recs)
    return recs


def align(dataset: str, marker: str, threads: int | None = None) -> pd.DataFrame:
    log = get_logger("align")
    qdir, adir = paths.qc_dir(dataset, marker), paths.alignment_dir(dataset, marker)
    inp = qdir / "clean.fasta"
    meta = pd.read_csv(qdir / "clean_metadata.csv", dtype={"taxid": str})
    log.info("Aligning %d sequences with MAFFT (%s/%s)...", len(meta), dataset, marker)
    raw_names = [n for n, _ in read_fasta(inp)]
    recs = run_mafft(inp, adir / "aligned.fasta", threads=threads)
    meta = meta.set_index("seq_id").loc[[n for n, _ in recs if n in set(raw_names)]].reset_index()
    meta["reverse_complemented"] = meta["seq_id"].isin(run_mafft.reversed)
    meta.to_csv(adir / "aligned_metadata.csv", index=False)
    if meta["reverse_complemented"].any():
        log.info("%d records were reverse-complemented by MAFFT to match the others", int(meta["reverse_complemented"].sum()))
    X = alignment_matrix([s for _, s in recs])
    from .exclude import ids as excluded
    skip = excluded(dataset, marker)
    keep = [i for i, (n, _) in enumerate(recs) if n not in skip]
    prof = column_profile(X[keep] if skip else X)
    prof.to_csv(adir / "columns.csv", index=False)
    log.info("Alignment: %d sequences x %d columns, %d variable columns", X.shape[0], X.shape[1],
             int((prof["entropy"] > 0).sum()))
    if settings()["qc"].get("flag_misidentified", True) and len(meta) > 2:
        flags = flag_misidentified(X, meta, settings()["qc"].get("min_overlap", 50))
        flags.to_csv(adir / "misid_flags.csv", index=False)
        log.info("Flagged %d records as possibly misidentified -> misid_flags.csv", len(flags))
    return prof
