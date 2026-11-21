"""Small shared helpers: FASTA I/O, sequence utilities, logging."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from . import paths

COMP = str.maketrans("ACGTRYSWKMBDHVNIacgtryswkmbdhvni-", "TGCAYRSWMKVHDBNNtgcayrswmkvhdbnn-")

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT", "I": "ACGT",
}
IUPAC_FROM_SET = {frozenset(v): k for k, v in IUPAC.items() if k not in ("U", "I")}
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def clean_primer(seq: str) -> str:
    return seq.strip().upper().replace(" ", "").replace("U", "T").replace("I", "N")


def degeneracy(primer: str) -> int:
    d = 1
    for b in clean_primer(primer):
        d *= len(IUPAC.get(b, "ACGT"))
    return d


def expand(primer: str, limit: int = 256) -> list[str]:
    out = [""]
    for b in clean_primer(primer):
        out = [o + x for o in out for x in IUPAC.get(b, "ACGT")]
        if len(out) > limit:
            return out[:limit]
    return out


def encode(seq: str) -> np.ndarray:
    """A,C,G,T -> 0..3; anything else (gap, N, IUPAC) -> 4."""
    lut = np.full(256, 4, dtype=np.uint8)
    for k, v in CODE.items():
        lut[ord(k)] = v
        lut[ord(k.lower())] = v
    return lut[np.frombuffer(seq.encode(), dtype=np.uint8)]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    recs, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line.strip())
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def write_fasta(path: Path, recs: Iterable[tuple[str, str]], width: int = 0) -> None:
    with open(path, "w") as fh:
        for name, seq in recs:
            fh.write(f">{name}\n")
            if width:
                for i in range(0, len(seq), width):
                    fh.write(seq[i:i + width] + "\n")
            else:
                fh.write(seq + "\n")


def get_logger(command: str) -> logging.Logger:
    log = logging.getLogger(f"ispcr.{command}")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(paths.logs_dir() / f"{time.strftime('%Y%m%d')}_{command}.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def read_species_list(path: Path) -> list[str]:
    names = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip().strip(",;\t")
            if line:
                names.append(" ".join(line.split()))
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out
