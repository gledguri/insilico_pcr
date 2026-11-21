"""Download GenBank records from NCBI Nucleotide for every species in a dataset.

target = "mitogenome" (complete mitochondrial genomes) or a marker name from
settings.yaml (12S, 16S, COI, CYTB, ...). Downloads are cached per species in
data/genbank/<target>/<Genus_species>.gb (+ .json with the query), so re-running
only fetches what is missing.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd

from . import ncbi, paths, taxonomy
from .config import marker_cfg, settings
from .utils import get_logger


def build_query(taxid: str, target: str, refseq: bool | None = None) -> str:
    dl = settings()["download"]
    base = f"txid{taxid}[Organism:exp]"
    if target.lower() == "mitogenome":
        q = (f'{base} AND mitochondrion[filter] AND "complete genome"[Title] '
             f'AND {dl["mitogenome_min_len"]}:{dl["mitogenome_max_len"]}[SLEN]')
        if refseq is True:
            q += " AND refseq[filter]"
        elif refseq is False:
            q += " NOT refseq[filter]"
        return q
    m = marker_cfg(target)
    return (f'{base} AND {m["query"]} AND mitochondrion[filter] AND 1:{m.get("query_max_len", 6000)}[SLEN] '
            f'NOT "complete genome"[Title] NOT PREDICTED[Title]')


def search_ids(taxid: str, target: str, retmax: int) -> tuple[list[str], int, str]:
    if target.lower() == "mitogenome" and settings()["download"].get("prefer_refseq", True):
        q_ref, q_oth = build_query(taxid, target, True), build_query(taxid, target, False)
        ids = ncbi.esearch("nucleotide", q_ref, retmax=retmax, idtype="acc")
        if len(ids) < retmax:
            ids += ncbi.esearch("nucleotide", q_oth, retmax=retmax - len(ids), idtype="acc")
        total = ncbi.esearch_count("nucleotide", build_query(taxid, target))
        return ids, total, build_query(taxid, target)
    q = build_query(taxid, target)
    ids = ncbi.esearch("nucleotide", q, retmax=retmax, idtype="acc")
    total = ncbi.esearch_count("nucleotide", q) if len(ids) >= retmax else len(ids)
    return ids, total, q


def fetch_dataset(dataset: str, target: str, max_per_species: int | None = None, force: bool = False,
                  progress: Callable[[int, int, str], None] | None = None, offline: bool | None = None) -> pd.DataFrame:
    """offline=None: check whether NCBI is reachable. Offline, only files already in data/genbank/ are used."""
    log = get_logger("fetch")
    if offline is None:
        offline = not ncbi.online()
    if offline:
        log.info("Offline: using files already in data/genbank/%s (nothing is downloaded)", target)
    dl = settings()["download"]
    if max_per_species is None:
        max_per_species = dl["max_mitogenomes_per_species"] if target.lower() == "mitogenome" \
            else dl["max_gene_records_per_species"]
    tax = taxonomy.load(dataset)
    tax = tax[tax["ncbi_taxid"].notna()]
    gdir = paths.genbank_dir(target)
    batch = settings()["ncbi"].get("batch_size", 50)
    rows = []
    log.info("Fetching %s for %d species (max %d each) -> %s", target, len(tax), max_per_species, gdir)
    for i, r in enumerate(tax.to_dict("records")):
        sp, taxid = r["accepted_name"], r["ncbi_taxid"]
        gb, meta = gdir / f"{paths.safe(sp)}.gb", gdir / f"{paths.safe(sp)}.json"
        if progress:
            progress(i, len(tax), sp)
        query = build_query(taxid, target)
        if offline:
            m = json.loads(meta.read_text()) if meta.exists() else {}
            n = len(m.get("accessions", [])) if gb.exists() else 0
            rows.append({"species": sp, "taxid": taxid, "downloaded": n, "available": m.get("available"),
                         "cached": gb.exists(), "error": None if gb.exists() else "not downloaded yet (offline)"})
            continue
        if not force and gb.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if m.get("query") == query and m.get("max", 0) >= max_per_species or m.get("imported"):
                rows.append({"species": sp, "taxid": taxid, "downloaded": len(m["accessions"]),
                             "available": m.get("available"), "cached": True})
                continue
        try:
            ids, total, q = search_ids(taxid, target, max_per_species)
            text = ""
            for j in range(0, len(ids), batch):
                text += ncbi.efetch_text("nucleotide", ids[j:j + batch], rettype="gb")
            gb.write_text(text)
            meta.write_text(json.dumps({"species": sp, "taxid": taxid, "query": q, "max": max_per_species,
                                        "available": total, "accessions": ids,
                                        "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
            rows.append({"species": sp, "taxid": taxid, "downloaded": len(ids), "available": total,
                         "cached": False})
            log.info("  %-32s %3d of %d records", sp, len(ids), total)
        except Exception as e:  # keep going; report at the end
            log.error("  %-32s failed: %s", sp, e)
            rows.append({"species": sp, "taxid": taxid, "downloaded": 0, "available": None,
                         "cached": False, "error": str(e)})
    if progress:
        progress(len(tax), len(tax), "done")
    df = pd.DataFrame(rows)
    out = paths.sequences_dir(dataset, target) / f"download_summary_{paths.safe(target)}.csv"
    df.to_csv(out, index=False)
    log.info("Saved %s (%d species with no records)", out, int((df["downloaded"] == 0).sum()))
    return df
