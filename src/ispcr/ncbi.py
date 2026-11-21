"""Thin, polite wrapper around NCBI E-utilities (via Biopython Entrez)."""
from __future__ import annotations

import time
import warnings
from typing import Any
from urllib.error import HTTPError, URLError

from Bio import Entrez

from .config import settings

_last_call = [0.0]


def online(timeout: float = 2.0) -> bool:
    """True when NCBI can be reached. Only opens a connection (~10 ms); asking the server for a page took ~3 s."""
    import socket
    try:
        socket.create_connection(("eutils.ncbi.nlm.nih.gov", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _setup() -> float:
    cfg = settings()["ncbi"]
    Entrez.email = cfg.get("email") or None
    Entrez.tool = cfg.get("tool", "ispcr")
    Entrez.api_key = cfg.get("api_key") or None
    return 0.11 if Entrez.api_key else 0.35


def _call(fn, retries: int = 5, **kw) -> Any:
    delay = _setup()
    for attempt in range(retries):
        wait = delay - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                handle = fn(**kw)
            return handle
        except (HTTPError, URLError, ConnectionError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1) + (3 if getattr(e, "code", 0) == 429 else 0))
    raise RuntimeError("unreachable")


def esearch(db: str, term: str, retmax: int = 100, **kw) -> list[str]:
    h = _call(Entrez.esearch, db=db, term=term, retmax=retmax, **kw)
    res = Entrez.read(h)
    h.close()
    return list(res.get("IdList", []))


def esearch_count(db: str, term: str) -> int:
    h = _call(Entrez.esearch, db=db, term=term, retmax=0)
    res = Entrez.read(h)
    h.close()
    return int(res.get("Count", 0))


def efetch_text(db: str, ids: list[str], rettype: str, retmode: str = "text") -> str:
    h = _call(Entrez.efetch, db=db, id=",".join(ids), rettype=rettype, retmode=retmode)
    txt = h.read()
    h.close()
    return txt.decode() if isinstance(txt, bytes) else txt


def efetch_xml(db: str, ids: list[str]) -> Any:
    h = _call(Entrez.efetch, db=db, id=",".join(ids), retmode="xml")
    res = Entrez.read(h)
    h.close()
    return res
