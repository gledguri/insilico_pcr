"""Gene map of a mitochondrial genome from its GenBank record (for the circular map in step 8)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from . import paths

# standard names, in the order they usually appear in vertebrate mitogenomes
PATTERNS = [
    ("ND4L", r"\b(nd4l|nad4l|mt-nd4l|nadh dehydrogenase subunit 4l)\b"),
    ("ND1", r"\b(nd1|nad1|mt-nd1|nadh dehydrogenase subunit (1|i))\b"),
    ("ND2", r"\b(nd2|nad2|mt-nd2|nadh dehydrogenase subunit (2|ii))\b"),
    ("ND3", r"\b(nd3|nad3|mt-nd3|nadh dehydrogenase subunit (3|iii))\b"),
    ("ND4", r"\b(nd4|nad4|mt-nd4|nadh dehydrogenase subunit (4|iv))\b"),
    ("ND5", r"\b(nd5|nad5|mt-nd5|nadh dehydrogenase subunit (5|v))\b"),
    ("ND6", r"\b(nd6|nad6|mt-nd6|nadh dehydrogenase subunit (6|vi))\b"),
    ("COI", r"\b(coi|cox1|co1|coxi|mt-co1|cytochrome c oxidase subunit (1|i))\b"),
    ("COII", r"\b(coii|cox2|co2|coxii|mt-co2|cytochrome c oxidase subunit (2|ii))\b"),
    ("COIII", r"\b(coiii|cox3|co3|coxiii|mt-co3|cytochrome c oxidase subunit (3|iii))\b"),
    ("ATP8", r"\b(atp8|atpase8|mt-atp8|atp synthase f0 subunit 8|atpase subunit 8)\b"),
    ("ATP6", r"\b(atp6|atpase6|mt-atp6|atp synthase f0 subunit 6|atpase subunit 6)\b"),
    ("CYTB", r"\b(cytb|cob|mt-cyb|cytochrome b)\b"),
    ("12S", r"\b(12s|rrns|rrn12|s-rrna|small subunit ribosomal rna)\b"),
    ("16S", r"\b(16s|rrnl|rrn16|l-rrna|large subunit ribosomal rna)\b"),
]
AA = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
      "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
      "Tyr": "Y", "Val": "V"}
CATEGORY = {"ND1": "Complex I (NADH dehydrogenase)", "ND2": "Complex I (NADH dehydrogenase)", "ND3": "Complex I (NADH dehydrogenase)",
            "ND4": "Complex I (NADH dehydrogenase)", "ND4L": "Complex I (NADH dehydrogenase)", "ND5": "Complex I (NADH dehydrogenase)",
            "ND6": "Complex I (NADH dehydrogenase)", "CYTB": "Complex III (cytochrome b)",
            "COI": "Complex IV (cytochrome c oxidase)", "COII": "Complex IV (cytochrome c oxidase)", "COIII": "Complex IV (cytochrome c oxidase)",
            "ATP6": "Complex V (ATP synthase)", "ATP8": "Complex V (ATP synthase)", "12S": "rRNA", "16S": "rRNA"}


def _text(f) -> str:
    v = []
    for q in ("gene", "product", "note", "standard_name", "gene_synonym"):
        v += f.qualifiers.get(q, [])
    return " | ".join(v)


def standard_name(f) -> tuple[str, str]:
    """(short name, category) for a GenBank feature."""
    t = _text(f)
    tl = t.lower()
    if f.type == "D-loop" or re.search(r"control region|d-loop|dloop", tl):
        return "Control region", "Control region"
    m1 = re.match(r"^trn([A-Z])\d?\b", " ".join(f.qualifiers.get("gene", [])))
    if f.type != "tRNA" and m1:
        three = {v: k for k, v in AA.items()}.get(m1.group(1), "?")
        return f"tRNA-{three}", "tRNA"
    if f.type == "tRNA" or tl.startswith("trna") or "trna-" in tl:
        m = re.search(r"tRNA-([A-Z][a-z]{2})", t)
        aa = AA.get(m.group(1), "?") if m else "?"
        cod = re.search(r"\(([ACGTU]{3})\)|anticodon[:\s]*([ACGTU]{3})", t, re.I)
        suffix = f" ({(cod.group(1) or cod.group(2)).upper()})" if cod else ""
        return f"tRNA-{m.group(1) if m else '?'}{suffix}", "tRNA"
    for name, rx in PATTERNS:
        if re.search(rx, tl):
            return name, CATEGORY[name]
    return (f.qualifiers.get("gene") or f.qualifiers.get("product") or [f.type])[0], "Other"


def features(rec) -> pd.DataFrame:
    """One row per gene/part. Positions are 1-based inclusive; parts crossing the origin are split."""
    rows, seen = [], set()
    typed = [f for f in rec.features if f.type in ("CDS", "rRNA", "tRNA", "D-loop")]
    misc = [f for f in rec.features if f.type in ("misc_feature", "misc_RNA", "rep_origin")
            and re.search(r"control region|d-loop|dloop", _text(f).lower())]
    genes = [f for f in rec.features if f.type == "gene"]
    covered = {(int(f.location.start), int(f.location.end)) for f in typed + misc}
    genes = [f for f in genes if (int(f.location.start), int(f.location.end)) not in covered]
    for f in typed + misc + genes:
        name, cat = standard_name(f)
        if cat == "Other" and f.type == "gene" and not f.qualifiers.get("gene"):
            continue
        key = (name, int(f.location.start), int(f.location.end))
        if key in seen:
            continue
        seen.add(key)
        strand = f.location.strand or 1
        total = sum(len(p) for p in f.location.parts)
        for k, part in enumerate(f.location.parts):
            rows.append({"name": name, "category": cat, "type": f.type, "start": int(part.start) + 1, "end": int(part.end),
                         "strand": "+" if strand >= 0 else "−", "length": total,
                         "product": (f.qualifiers.get("product") or [""])[0], "part": k})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["start", "end"]).reset_index(drop=True)
        # the same gene annotated twice (e.g. gene + CDS with slightly different ends): keep the longest
        keep = []
        for name, g in df[df["part"] == 0].groupby("name"):
            if name.startswith("tRNA"):
                keep += list(g.index)
                continue
            g = g.sort_values("length", ascending=False)
            chosen = []
            for i, r in g.iterrows():
                overl = any(not (r.end < df.at[j, "start"] or r.start > df.at[j, "end"]) for j in chosen)
                if not overl and (name != "Control region" or not chosen):
                    chosen.append(i)
            keep += chosen
        df = df[df.apply(lambda r: r["part"] > 0 or r.name in set(keep), axis=1)].reset_index(drop=True)
        # tRNA one-letter labels (L1/L2, S1/S2 for the two copies)
        df["label"] = df["name"]
        tr = df["category"] == "tRNA"
        aa1 = df.loc[tr, "name"].str.extract(r"tRNA-([A-Z][a-z]{2})")[0].map(AA).fillna("?")
        df.loc[tr, "label"] = aa1
        for aa in ("L", "S"):
            idx = df.index[tr & (df["label"] == aa) & (df["part"] == 0)]
            if len(idx) == 2:
                df.loc[idx[0], "label"], df.loc[idx[1], "label"] = f"{aa}1", f"{aa}2"
    return df


def gene_order(df: pd.DataFrame) -> str:
    """Order of protein-coding and rRNA genes, read from 12S onwards (so records that start at a
    different point of the circle compare equal). The control region is left out."""
    g = df[(df["category"] != "tRNA") & (df["category"] != "Control region") & (df["part"] == 0)].sort_values("start")
    names = list(g["name"])
    if "12S" in names:
        k = names.index("12S")
        names = names[k:] + names[:k]
    return " ".join(names)


def gc_windows(seq: str, window: int = 200) -> pd.DataFrame:
    s = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    gc = np.isin(s, [ord("G"), ord("C")]).astype(float)
    g, c = (s == ord("G")).astype(float), (s == ord("C")).astype(float)
    starts = np.arange(0, len(s), window)
    rows = []
    for a in starts:
        b = min(len(s), a + window)
        gs, cs = g[a:b].sum(), c[a:b].sum()
        rows.append({"start": a + 1, "end": b, "gc": gc[a:b].mean(), "gc_skew": (gs - cs) / max(gs + cs, 1)})
    return pd.DataFrame(rows)


def cached_species(dataset: str | None = None) -> list[str]:
    files = sorted(paths.genbank_dir("mitogenome").glob("*.gb"))
    have = {f.stem.replace("_", " "): f for f in files if f.stat().st_size > 0}
    if dataset:
        try:
            from .taxonomy import load
            names = set(load(dataset)["accepted_name"])
            have = {k: v for k, v in have.items() if k in names}
        except FileNotFoundError:
            pass
    return sorted(have)


def records(species: str) -> list:
    p: Path = paths.genbank_dir("mitogenome") / f"{paths.safe(species)}.gb"
    return list(SeqIO.parse(str(p), "genbank")) if p.exists() else []


def primer_sites(seq: str, pairs: dict[str, dict]) -> pd.DataFrame:
    """Where each primer pair binds in this genome (uses the in silico PCR engine)."""
    from .pcr import PrimerPair, run_on_sequences
    rows = []
    L = len(seq)
    for name, e in pairs.items():
        pair = PrimerPair(name, e["forward"], e["reverse"], int(e.get("product_min", 0)), int(e.get("product_max", 100000)))
        h = run_on_sequences([("g", seq)], pair, thermo_binding=False).iloc[0]
        if h["f"] is None or pd.isna(h["f"]) or h["product_len"] is None or pd.isna(h["product_len"]):
            rows.append({"pair": name, "status": h["status"], "found": False})
            continue
        f, plen = int(h["f"]), int(h["product_len"])
        start = f + 1 if h["strand"] == "+" else L - (f + plen) + 1
        rows.append({"pair": name, "status": h["status"], "found": True, "start": start, "end": start + plen - 1,
                     "strand": h["strand"], "product_len": plen, "insert_len": int(h["insert_len"]),
                     "f_mm": int(h["f_mm"]), "r_mm": int(h["r_mm"]), "f_mismatches": h["f_mismatches"] or "",
                     "r_mismatches": h["r_mismatches"] or "", "marker": e.get("marker", "")})
    return pd.DataFrame(rows)
