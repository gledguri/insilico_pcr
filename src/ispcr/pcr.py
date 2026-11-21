"""In silico PCR: find primer binding sites (with mismatches) in every sequence.

For each sequence both strands are searched. A forward site and a downstream
reverse site (reverse complement of the reverse primer) form a product. The best
product is the one with the fewest mismatches, weighting mismatches at the 3' end.

Status per sequence:
  amplified          both sites within the mismatch rules and product length in range
  fail_mismatch      sites found but too many mismatches (total or at the 3' end)
  fail_length        sites found but product length outside the expected range
  no_site            no plausible binding site for one or both primers
  site_not_covered   the record ends before a primer site (partial sequence), so it
                     cannot be tested; decided from the alignment
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import paths, qc, thermo
from .config import primers as primer_library
from .config import settings
from .utils import CODE, IUPAC, clean_primer, encode, get_logger, read_fasta, revcomp, write_fasta


@dataclass
class PrimerPair:
    name: str
    forward: str
    reverse: str
    product_min: int = 0
    product_max: int = 100000
    marker: str = ""
    reference: str = ""

    def __post_init__(self):
        self.forward, self.reverse = clean_primer(self.forward), clean_primer(self.reverse)


def get_pair(name: str) -> PrimerPair:
    lib = primer_library()
    if name not in lib:
        raise KeyError(f"Primer pair '{name}' not in config/primers.yaml (known: {', '.join(lib)})")
    e = lib[name]
    return PrimerPair(name, e["forward"], e["reverse"], int(e.get("product_min", 0)),
                      int(e.get("product_max", 100000)), e.get("marker", ""), e.get("reference", ""))


def primer_mask(primer: str) -> np.ndarray:
    m = np.zeros((len(primer), 5), dtype=bool)
    for k, b in enumerate(primer):
        for x in IUPAC.get(b, "ACGT"):
            m[k, CODE[x]] = True
    m[:, 4] = True  # ambiguous template base: not counted as mismatch
    return m


def mismatch_windows(codes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = mask.shape[0]
    if len(codes) < p:
        return np.zeros((0, p), dtype=bool)
    W = sliding_window_view(codes, p)
    return ~mask[np.arange(p)[None, :], W]


def _search_strand(seq: str, pair: PrimerPair, masks, cfg) -> dict:
    fmask, rmask = masks
    codes = encode(seq)
    pF, pR, w = len(pair.forward), len(pair.reverse), cfg["three_prime_window"]
    mF, mR = mismatch_windows(codes, fmask), mismatch_windows(codes, rmask)
    if not len(mF):
        return {"found_f": False, "found_r": False, "penalty": 1e9}
    totF, f3 = mF.sum(1), mF[:, -w:].sum(1)
    totR, r3 = mR.sum(1), mR[:, :w].sum(1)
    penF, penR = totF + 2 * f3, totR + 2 * r3
    smax = cfg["search_max_mismatches"]
    fc = np.flatnonzero(totF <= smax)
    rc = np.flatnonzero(totR <= smax)
    fc = fc[np.argsort(penF[fc])][:200]
    rc = rc[np.argsort(penR[rc])][:200]
    best = None
    for f in fc:
        rs = rc[(rc >= f + pF) & (rc + pR - f <= cfg["max_product_search"])]
        if not len(rs):
            continue
        prod = rs + pR - f
        pen = penF[f] + penR[rs] + np.where((prod >= pair.product_min) & (prod <= pair.product_max), 0, 6)
        k = int(np.argmin(pen))
        if best is None or pen[k] < best[0]:
            best = (float(pen[k]), int(f), int(rs[k]))
    out = {"found_f": len(fc) > 0, "found_r": len(rc) > 0}
    if best is None:
        f = int(np.argmin(penF)) if len(fc) else None
        r = int(np.argmin(penR)) if len(rc) else None
        out.update(penalty=1e6 + (penF[f] if f is not None else 50) + (penR[r] if r is not None else 50), f=f, r=r)
    else:
        out.update(penalty=best[0], f=best[1], r=best[2])
    f, r = out.get("f"), out.get("r")
    if f is not None:
        out.update(f_mm=int(totF[f]), f_mm3=int(f3[f]), f_vec=mF[f], f_site=seq[f:f + pF])
    if r is not None:
        site_rc = seq[r:r + pR]
        out.update(r_mm=int(totR[r]), r_mm3=int(r3[r]), r_vec=mR[r][::-1], r_site=revcomp(site_rc))
    if best is not None:
        out.update(product=seq[f:r + pR], insert=seq[f + pF:r], product_len=r + pR - f, insert_len=r - f - pF)
    return out


def _mm_list(vec, site, primer) -> str:
    if vec is None or site is None:
        return ""
    return ";".join(f"{k + 1}{primer[k]}>{site[k]}" for k in np.flatnonzero(vec))


def run_on_sequences(records: list[tuple[str, str]], pair: PrimerPair, aligned: dict[str, str] | None = None,
                     cfg: dict | None = None, thermo_binding: bool = True) -> pd.DataFrame:
    """records: (seq_id, ungapped sequence). aligned: optional seq_id -> aligned row (same orientation)."""
    cfg = {**settings()["pcr"], **(cfg or {})}
    masks = (primer_mask(pair.forward), primer_mask(revcomp(pair.reverse)))
    pF, pR = len(pair.forward), len(pair.reverse)
    rows = []
    for sid, seq in records:
        seq = seq.upper().replace("-", "")
        plus = _search_strand(seq, pair, masks, cfg)
        minus = _search_strand(revcomp(seq), pair, masks, cfg)
        res, strand = (plus, "+") if plus["penalty"] <= minus["penalty"] else (minus, "-")
        row = {"seq_id": sid, "strand": strand, "seq_len": len(seq)}
        for k in ("f", "r", "f_mm", "f_mm3", "r_mm", "r_mm3", "f_site", "r_site", "product_len", "insert_len"):
            row[k] = res.get(k)
        row["f_mismatches"] = _mm_list(res.get("f_vec"), res.get("f_site"), pair.forward)
        row["r_mismatches"] = _mm_list(res.get("r_vec"), res.get("r_site"), pair.reverse)
        row["_f_vec"], row["_r_vec"] = res.get("f_vec"), res.get("r_vec")
        row["insert"], row["product"] = res.get("insert"), res.get("product")
        if "product" not in res:
            status = "no_site"
        elif (row["f_mm"] > cfg["max_mismatches"] or row["r_mm"] > cfg["max_mismatches"]
              or row["f_mm3"] > cfg["max_3prime_mismatches"] or row["r_mm3"] > cfg["max_3prime_mismatches"]):
            status = "fail_mismatch"
        elif not (pair.product_min <= row["product_len"] <= pair.product_max):
            status = "fail_length"
        else:
            status = "amplified"
        row["status"] = status
        if thermo_binding:
            row["f_bind_tm"], row["f_bind_dg"] = thermo.binding(pair.forward, row["f_site"]) if row["f_site"] else (np.nan, np.nan)
            row["r_bind_tm"], row["r_bind_dg"] = thermo.binding(pair.reverse, row["r_site"]) if row["r_site"] else (np.nan, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if aligned is not None and len(df):
        df = _mark_not_covered(df, aligned, pF, pR, cfg.get("coverage_slack", 3))
    return df


def _mark_not_covered(df: pd.DataFrame, aligned: dict[str, str], pF: int, pR: int, slack: int) -> pd.DataFrame:
    """Use alignment columns of the primer sites (from amplified records) to tell partial records apart."""
    pos2col = {}
    for sid, row in aligned.items():
        pos2col[sid] = np.flatnonzero(np.frombuffer(row.encode(), dtype=np.uint8) != ord("-"))
    ok = df[(df["status"] != "no_site") & (df["strand"] == "+") & df["f"].notna() & df["r"].notna()]
    ok = ok[ok["status"].isin(["amplified", "fail_mismatch", "fail_length"])]
    if not len(ok):
        df["site_columns"] = ""
        return df
    fs, fe, rs, re_ = [], [], [], []
    for r in ok.itertuples():
        pc = pos2col.get(r.seq_id)
        if pc is None or len(pc) < (r.r + pR):
            continue
        fs.append(pc[int(r.f)]); fe.append(pc[int(r.f) + pF - 1]); rs.append(pc[int(r.r)]); re_.append(pc[int(r.r) + pR - 1])
    if not fs:
        df["site_columns"] = ""
        return df
    cfs, cfe, crs, cre = (int(np.median(x)) for x in (fs, fe, rs, re_))
    df["site_columns"] = f"F {cfs + 1}-{cfe + 1}; R {crs + 1}-{cre + 1}"
    for i, r in df.iterrows():
        if r["status"] == "amplified":
            continue
        pc = pos2col.get(r["seq_id"])
        if pc is None or not len(pc):
            continue
        first, last = pc[0], pc[-1]
        cov_f = first <= cfs + slack and last >= cfe - slack
        cov_r = first <= crs + slack and last >= cre - slack
        if not (cov_f and cov_r):
            df.at[i, "status"] = "site_not_covered"
            df.at[i, "not_covered"] = ("F" if not cov_f else "") + ("R" if not cov_r else "")
    return df


def mismatch_matrix(hits: pd.DataFrame, vec_col: str, plen: int) -> pd.DataFrame:
    """Fraction of tested records per species with a mismatch at each primer position (5'->3')."""
    tested = hits[hits["status"].isin(["amplified", "fail_mismatch", "fail_length"]) & hits[vec_col].notna()]
    out = {}
    for sp, g in tested.groupby("species"):
        out[sp] = np.mean(np.stack(g[vec_col].values), axis=0)
    return pd.DataFrame.from_dict(out, orient="index", columns=[str(i) for i in range(1, plen + 1)]).sort_index()


def run_pcr(dataset: str, marker: str, pair: PrimerPair, use_alignment: bool = True, rules: dict | None = None) -> dict:
    log = get_logger("pcr")
    qdir, adir = paths.qc_dir(dataset, marker), paths.alignment_dir(dataset, marker)
    meta = pd.read_csv(qdir / "clean_metadata.csv", dtype={"taxid": str})
    aligned = None
    if use_alignment and (adir / "aligned.fasta").exists():
        aligned = dict(read_fasta(adir / "aligned.fasta"))
        records = [(k, v.replace("-", "")) for k, v in aligned.items()]
    else:
        records = read_fasta(qdir / "clean.fasta")
    excl = qc.excluded_ids(dataset, marker)
    records = [r for r in records if r[0] not in excl]
    log.info("In silico PCR %s on %s/%s: %d sequences", pair.name, dataset, marker, len(records))
    hits = run_on_sequences(records, pair, aligned, cfg=rules)
    cols = ["seq_id", "species", "genus", "family", "order", "source", "country"]
    hits = hits.merge(meta[[c for c in cols if c in meta.columns]], on="seq_id", how="left")
    out = paths.pcr_dir(dataset, marker, pair.name)
    write_fasta(out / "amplicons.fasta", [(r.seq_id, r.insert) for r in hits.itertuples() if r.status == "amplified"])
    write_fasta(out / "amplicons_with_primers.fasta",
                [(r.seq_id, r.product) for r in hits.itertuples() if r.status == "amplified"])
    mmF = mismatch_matrix(hits, "_f_vec", len(pair.forward))
    mmR = mismatch_matrix(hits, "_r_vec", len(pair.reverse))
    mmF.to_csv(out / "mismatch_matrix_forward.csv")
    mmR.to_csv(out / "mismatch_matrix_reverse.csv")
    hits.drop(columns=["_f_vec", "_r_vec", "product"]).to_csv(out / "hits.csv", index=False)
    sp = species_summary(hits, dataset, marker)
    sp.to_csv(out / "species_summary.csv", index=False)
    summary = make_summary(hits, sp, pair, rules)
    summary["excluded_records"] = sorted(excl)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("  species amplified %d/%d testable; sequences amplified %d/%d -> %s",
             summary["species_amplified"], summary["species_testable"], summary["seqs_amplified"],
             summary["seqs_testable"], out)
    return summary


def species_summary(hits: pd.DataFrame, dataset: str | None = None, marker: str | None = None) -> pd.DataFrame:
    st = hits.pivot_table(index="species", columns="status", values="seq_id", aggfunc="count", fill_value=0)
    for c in ["amplified", "fail_mismatch", "fail_length", "no_site", "site_not_covered"]:
        if c not in st:
            st[c] = 0
    st = st.reset_index()
    st["n_seqs"] = st[["amplified", "fail_mismatch", "fail_length", "no_site", "site_not_covered"]].sum(1)
    st["n_testable"] = st["n_seqs"] - st["site_not_covered"]
    st["frac_amplified"] = (st["amplified"] / st["n_testable"].replace(0, np.nan)).round(3)
    g = hits.groupby("species")
    st["min_f_mm"] = st["species"].map(g["f_mm"].min())
    st["min_r_mm"] = st["species"].map(g["r_mm"].min())
    st["median_insert_len"] = st["species"].map(hits[hits["status"] == "amplified"].groupby("species")["insert_len"].median())
    tax = hits.drop_duplicates("species").set_index("species")
    for c in ("genus", "family", "order"):
        if c in tax:
            st[c] = st["species"].map(tax[c])
    st["result"] = np.select([st["amplified"] > 0, st["n_testable"] == 0], ["amplified", "not_testable"], "fails")
    if dataset and marker:
        counts = pd.read_csv(paths.qc_dir(dataset, marker) / "species_counts.csv")
        missing = counts[~counts["species"].isin(st["species"])]
        extra = pd.DataFrame({"species": missing["species"], "family": missing["family"],
                              "order": missing["order"], "n_seqs": 0, "result": "no_sequences"})
        st = pd.concat([st, extra], ignore_index=True)
    return st


def make_summary(hits: pd.DataFrame, sp: pd.DataFrame, pair: PrimerPair, rules: dict | None = None) -> dict:
    amp = hits[hits["status"] == "amplified"]
    testable = hits[hits["status"] != "site_not_covered"]
    return {
        "primer_pair": asdict(pair), "rules": {**settings()["pcr"], **(rules or {})},
        "forward": thermo.properties(pair.forward), "reverse": thermo.properties(pair.reverse),
        "heterodimer_dg": thermo.heterodimer_dg(pair.forward, pair.reverse),
        "species_total": int(len(sp)), "species_with_sequences": int((sp["n_seqs"] > 0).sum()),
        "species_testable": int((sp["result"].isin(["amplified", "fails"])).sum()),
        "species_amplified": int((sp["result"] == "amplified").sum()),
        "species_fail": sp.loc[sp["result"] == "fails", "species"].tolist(),
        "species_not_testable": sp.loc[sp["result"] == "not_testable", "species"].tolist(),
        "species_no_sequences": sp.loc[sp["result"] == "no_sequences", "species"].tolist(),
        "seqs_total": int(len(hits)), "seqs_testable": int(len(testable)), "seqs_amplified": int(len(amp)),
        "status_counts": hits["status"].value_counts().to_dict(),
        "median_product_len": float(amp["product_len"].median()) if len(amp) else None,
        "median_insert_len": float(amp["insert_len"].median()) if len(amp) else None,
    }
