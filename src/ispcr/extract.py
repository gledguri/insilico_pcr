"""Cut a gene region out of downloaded GenBank records.

Reads data/genbank/<marker>/ (gene records) and, if enabled, data/genbank/mitogenome/
(complete mitogenomes), and writes data/sequences/<dataset>/<marker>/sequences.fasta
plus metadata.csv. Use marker "mitogenome" to keep whole mitogenomes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from . import paths, taxonomy
from .config import marker_cfg, settings
from .utils import get_logger, read_fasta, write_fasta

REFSEQ_IDENT = re.compile(r"identical to ([A-Z]{1,2}\d{5,8})", re.I)


def _qual_text(feat) -> str:
    vals = []
    for q in ("gene", "product", "note", "standard_name", "gene_synonym"):
        vals += feat.qualifiers.get(q, [])
    return " | ".join(vals).lower()


def _source_info(rec) -> dict:
    src = next((f for f in rec.features if f.type == "source"), None)
    q = src.qualifiers if src else {}
    g = lambda k: (q.get(k) or [None])[0]
    return {"organism": rec.annotations.get("organism"), "country": g("country") or g("geo_loc_name"),
            "voucher": g("specimen_voucher") or g("isolate"), "collection_date": g("collection_date")}


def extract_record(rec, mcfg: dict, allow_whole: bool) -> tuple[str, str, str] | None:
    """Return (sequence, how, feature_type) or None."""
    rx = re.compile(mcfg["feature_regex"], re.I)
    types = set(mcfg.get("feature_types", ["rRNA", "gene", "CDS"]))
    best = None
    for f in rec.features:
        if f.type in types and rx.search(_qual_text(f)):
            if best is None or len(f.location) > len(best.location):
                best = f
    if best is not None:
        try:
            return str(best.extract(rec.seq)).upper(), "feature", best.type
        except Exception:
            return None
    annotated = any(f.type in ("gene", "CDS", "rRNA", "tRNA") for f in rec.features)
    if (allow_whole and not annotated and rx.search(rec.description.lower())
            and mcfg["min_len"] <= len(rec.seq) <= mcfg["max_len"]):
        return str(rec.seq).upper(), "whole_record", ""
    return None


class _FastaRec:
    """Minimal stand-in for a SeqRecord built from an imported FASTA entry."""
    def __init__(self, name: str, seq: str):
        self.id, self.seq, self.description = name, seq, f"{name} (imported FASTA)"
        self.annotations, self.features = {}, []


def _iter_records(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return list(SeqIO.parse(str(path), "genbank"))
    except Exception:
        return []


def extract(dataset: str, marker: str) -> pd.DataFrame:
    log = get_logger("extract")
    tax = taxonomy.load(dataset)
    tax = tax[tax["ncbi_taxid"].notna()]
    is_mito = marker.lower() == "mitogenome"
    mcfg = None if is_mito else marker_cfg(marker)
    use_mito = is_mito or settings()["download"].get("include_mitogenomes_for_genes", True)
    rows, recs, seen = [], [], set()
    for r in tax.to_dict("records"):
        sp = r["accepted_name"]
        sources = []
        if not is_mito:
            sources.append(("gene_record", _iter_records(paths.genbank_dir(marker) / f"{paths.safe(sp)}.gb")))
        if use_mito:
            sources.append(("mitogenome", _iter_records(paths.genbank_dir("mitogenome") / f"{paths.safe(sp)}.gb")))
        # RefSeq mitogenomes duplicate an INSDC record: skip the duplicate
        dup_of_refseq = set()
        for _, rr in sources:
            for rec in rr:
                if rec.id.startswith("NC_"):
                    dup_of_refseq.update(m.upper() for m in REFSEQ_IDENT.findall(rec.annotations.get("comment", "")))
        imp = paths.imported_dir(marker) / f"{paths.safe(sp)}.fasta"
        if imp.exists():
            sources.append(("imported", [_FastaRec(n, q) for n, q in read_fasta(imp)]))
        n_sp = 0
        for source, rr in sources:
            for rec in rr:
                acc = rec.id
                if rec.annotations.get("molecule_type", "DNA") == "mRNA" or rec.description.startswith("PREDICTED"):
                    continue
                if acc in seen or acc.split(".")[0].upper() in dup_of_refseq:
                    continue
                if is_mito or source == "imported":
                    res = (str(rec.seq).upper().replace("-", ""), "whole_record" if source != "imported" else "imported_fasta", "")
                else:
                    res = extract_record(rec, mcfg, allow_whole=(source == "gene_record"))
                if not res:
                    continue
                seq, how, ftype = res
                seen.add(acc)
                n_sp += 1
                recs.append((acc, seq))
                rows.append({"seq_id": acc, "species": sp, "input_name": r["input_name"], "taxid": r["ncbi_taxid"],
                             "genus": r["genus"], "family": r["family"], "order": r["order"], "class": r["class"],
                             "source": source, "extraction": how, "feature_type": ftype, "length": len(seq),
                             "definition": rec.description, **_source_info(rec)})
        log.info("  %-32s %3d sequences", sp, n_sp)
    out = paths.sequences_dir(dataset, marker)
    write_fasta(out / "sequences.fasta", recs)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    log.info("Saved %d sequences for %d species -> %s", len(df), df["species"].nunique() if len(df) else 0, out)
    return df
