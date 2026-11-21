"""Primer design from an alignment.

1. Score every primer-length window of the alignment: build a (degenerate) primer
   from the bases present, and compute how many sequences / species it matches
   under the PCR rules (total and 3'-end mismatches).
2. Filter by melting temperature, GC, homopolymers; keep the best non-overlapping
   forward and reverse candidates.
3. Pair them by product length and Tm difference, and score pairs by joint species
   coverage and by how many species get a unique amplicon (resolution).
4. Optionally evaluate the top pairs with the full in silico PCR engine and the
   barcode-gap statistics.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd

from . import paths, qc, thermo
from .config import settings
from .pcr import PrimerPair, run_on_sequences
from .stats import alignment_matrix, column_profile
from .utils import IUPAC, IUPAC_FROM_SET, get_logger, read_fasta, revcomp

BASES = np.array(list("ACGT"))


def _degenerate(freqs: np.ndarray, min_freq: float, max_deg: int, three_prime_right: bool, w: int) -> list[set]:
    """freqs: k x 4 base frequencies. Returns per-position base sets within the degeneracy budget."""
    k = len(freqs)
    sets = [set(np.flatnonzero(f >= min_freq)) or {int(np.argmax(f))} for f in freqs]
    # positions near the 3' end keep their degeneracy longest
    weight = np.ones(k)
    if three_prime_right:
        weight[-w:] = 3
    else:
        weight[:w] = 3
    while np.prod([len(s) for s in sets]) > max_deg:
        best = None
        for i, s in enumerate(sets):
            if len(s) < 2:
                continue
            j = min(s, key=lambda b: freqs[i, b])
            val = freqs[i, j] * weight[i]
            if best is None or val < best[0]:
                best = (val, i, j)
        sets[best[1]].discard(best[2])
    return sets


def _to_iupac(sets: list[set]) -> str:
    return "".join(IUPAC_FROM_SET[frozenset(BASES[list(s)])] for s in sets)


def design(dataset: str, marker: str, params: dict | None = None, evaluate_top: int = 10,
           progress: Callable[[str, float], None] | None = None) -> pd.DataFrame:
    log = get_logger("design")
    p = {**settings()["design"], **(params or {})}
    pc = settings()["pcr"]
    w, max_mm, max_mm3 = pc["three_prime_window"], pc["max_mismatches"], pc["max_3prime_mismatches"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = paths.design_dir(dataset, marker, run_id)
    adir = paths.alignment_dir(dataset, marker)
    recs = read_fasta(adir / "aligned.fasta")
    meta = pd.read_csv(adir / "aligned_metadata.csv").set_index("seq_id")
    excl = qc.excluded_ids(dataset, marker)
    recs = [r for r in recs if r[0] not in excl and r[0] in meta.index]
    ids = [r[0] for r in recs]
    species = meta.loc[ids, "species"].values
    sp_names, sp_idx = np.unique(species, return_inverse=True)
    S = np.zeros((len(ids), len(sp_names)), dtype=np.float32)
    S[np.arange(len(ids)), sp_idx] = 1
    X = alignment_matrix([s for _, s in recs])
    n, L = X.shape
    log.info("Design on %s/%s: %d sequences, %d species, %d columns (run %s)", dataset, marker, n, len(sp_names), L, run_id)
    nongap = X != 5
    first = np.argmax(nongap, 1)
    last = L - 1 - np.argmax(nongap[:, ::-1], 1)
    prof = column_profile(X)
    covering = np.array([((first <= c) & (last >= c)).sum() for c in range(L)])
    internal_gap = (prof["gap"].values - (n - covering)) / np.maximum(covering, 1)
    prof["covering_seqs"] = covering
    prof["internal_gap_fraction"] = np.round(internal_gap, 4)
    prof.to_csv(out / "conservation.csv", index=False)
    freqs = prof[["A", "C", "G", "T"]].values.astype(float)
    freqs = freqs / np.maximum(freqs.sum(1, keepdims=True), 1)
    nongap_frac = nongap.sum(0) / np.maximum(covering, 1)  # among records that span the column
    cum_len = np.concatenate([[0], np.cumsum(nongap_frac)])
    core = np.flatnonzero((internal_gap < p["max_gap_fraction"]) & (covering >= max(3, 0.2 * n)))
    lens = range(p["primer_min_len"], p["primer_max_len"] + 1)
    cands = {"F": [], "R": []}
    matches = {"F": [], "R": []}
    covers = {"F": [], "R": []}
    total = len(core) * len(lens)
    done = 0
    for k in lens:
        for j in range(len(core) - k + 1):
            done += 1
            if progress and done % 2000 == 0:
                progress("Scanning primer sites", done / total)
            cols = core[j:j + k]
            if cols[-1] - cols[0] > k + 2:
                continue
            rows_cov = (first <= cols[0]) & (last >= cols[-1])
            if rows_cov.sum() < 3:
                continue
            Xw = X[np.ix_(rows_cov, cols)]
            for d in ("F", "R"):
                sets = _degenerate(freqs[cols], p["min_base_freq"], p["max_degeneracy"], d == "F", w)
                mask = np.zeros((k, 6), dtype=bool)
                for i, s in enumerate(sets):
                    mask[i, list(s)] = True
                mask[:, 4] = True
                mm = ~mask[np.arange(k)[None, :], Xw]
                tot = mm.sum(1)
                end = mm[:, -w:].sum(1) if d == "F" else mm[:, :w].sum(1)
                ok = (tot <= max_mm) & (end <= max_mm3)
                cov = ok.mean()
                if cov < 0.5:
                    continue
                plus = _to_iupac(sets)
                primer = plus if d == "F" else revcomp(plus)
                cons = "".join(BASES[np.argmax(freqs[cols], 1)])
                cons = cons if d == "F" else revcomp(cons)
                tm = thermo.tm(cons)
                gcv = thermo.gc(primer)
                if not (p["tm_min"] <= tm <= p["tm_max"] and p["gc_min"] <= gcv <= p["gc_max"]):
                    continue
                if thermo.max_homopolymer(cons) > p["max_homopolymer"]:
                    continue
                full = np.zeros(n, dtype=bool)
                full[rows_cov] = ok
                sp_cov = (S[rows_cov].T @ ok.astype(np.float32) > 0).sum() / max(1, (S[rows_cov].sum(0) > 0).sum())
                clamp = sum(b in "GCS" for b in primer[-5:])
                deg = int(np.prod([len(s) for s in sets]))
                score = sp_cov + 0.1 * cov - 0.02 * np.log2(deg) - (0.03 if clamp in (0, 5) else 0)
                cands[d].append({"dir": d, "primer": primer, "start_col": int(cols[0]) + 1, "end_col": int(cols[-1]) + 1,
                                 "length": k, "degeneracy": deg, "tm_consensus": round(tm, 1), "gc": round(gcv, 1),
                                 "seq_coverage": round(float(cov), 4), "species_coverage": round(float(sp_cov), 4),
                                 "gc_clamp": clamp, "score": round(float(score), 4)})
                matches[d].append(full)
                covers[d].append(rows_cov)
    # keep best non-overlapping candidates per direction
    kept = {}
    for d in ("F", "R"):
        if not cands[d]:
            raise ValueError(f"No {'forward' if d == 'F' else 'reverse'} primer candidates. Relax the settings "
                             "(Tm/GC range, degeneracy, mismatch rules).")
        df = pd.DataFrame(cands[d])
        df["cand_idx"] = range(len(df))
        df = df.sort_values("score", ascending=False)
        sel, starts = [], []
        for r in df.itertuples():
            if all(abs(r.start_col - s) >= 4 for s in starts):
                sel.append(r.cand_idx)
                starts.append(r.start_col)
            if len(sel) >= p["keep_per_direction"]:
                break
        kept[d] = df.set_index("cand_idx").loc[sel].reset_index()
        kept[d].to_csv(out / f"{'forward' if d == 'F' else 'reverse'}_candidates.csv", index=False)
    log.info("  %d forward and %d reverse candidates kept", len(kept["F"]), len(kept["R"]))
    MF = np.stack([matches["F"][i] for i in kept["F"]["cand_idx"]]).astype(np.float32)
    MR = np.stack([matches["R"][i] for i in kept["R"]["cand_idx"]]).astype(np.float32)
    CF = np.stack([covers["F"][i] for i in kept["F"]["cand_idx"]]).astype(np.float32)
    CR = np.stack([covers["R"][i] for i in kept["R"]["cand_idx"]]).astype(np.float32)
    pairs = []
    Fd, Rd = kept["F"], kept["R"]
    for i, f in enumerate(Fd.itertuples()):
        if progress and i % 20 == 0:
            progress("Pairing primers", i / len(Fd))
        prod = cum_len[Rd["end_col"].values] - cum_len[f.start_col - 1]
        valid = np.flatnonzero((Rd["start_col"].values > f.end_col) & (prod >= p["product_min"]) & (prod <= p["product_max"])
                               & (np.abs(Rd["tm_consensus"].values - f.tm_consensus) <= p["max_tm_diff"]))
        if not len(valid):
            continue
        both = MF[i][None, :] * MR[valid]
        testable = CF[i][None, :] * CR[valid]
        sp_amp = (both @ S > 0).sum(1)
        sp_test = (testable @ S > 0).sum(1)
        for v, a, t, nb, nt in zip(valid, sp_amp, sp_test, both.sum(1), testable.sum(1)):
            r = Rd.iloc[v]
            pairs.append({"f": i, "r": int(v), "forward": f.primer, "reverse": r["primer"],
                          "f_cols": f"{f.start_col}-{f.end_col}", "r_cols": f"{r['start_col']}-{r['end_col']}",
                          "product_len_est": int(round(prod[v])), "tm_f": f.tm_consensus, "tm_r": r["tm_consensus"],
                          "deg_f": f.degeneracy, "deg_r": int(r["degeneracy"]),
                          "species_amplified": int(a), "species_testable": int(t),
                          "species_coverage": round(a / max(t, 1), 4), "seq_coverage": round(nb / max(nt, 1), 4),
                          "f_end_col": f.end_col, "r_start_col": int(r["start_col"])})
    if not pairs:
        raise ValueError("No primer pairs fit the product length and Tm rules. Widen product_min/max or max_tm_diff.")
    pairs = pd.DataFrame(pairs).sort_values(["species_coverage", "seq_coverage"], ascending=False)
    pairs = pairs.head(max(3 * p["max_pairs"], 100)).copy()
    # resolution proxy: species whose amplicon (insert) sequences are not shared with any other species
    chars = np.array(list("ACGTN-"))
    res = []
    for r in pairs.itertuples():
        both = (MF[r.f] * MR[r.r]) > 0
        cols = slice(r.f_end_col, r.r_start_col - 1)
        ins = {}
        for row in np.flatnonzero(both):
            s = "".join(chars[X[row, cols]]).replace("-", "")
            ins.setdefault(species[row], set()).add(s)
        owner = {}
        for s, seqs in ins.items():
            for q in seqs:
                owner.setdefault(q, set()).add(s)
        unique = sum(all(len(owner[q]) == 1 for q in seqs) for seqs in ins.values())
        res.append(unique / max(1, len(ins)))
    pairs["resolution_unique"] = np.round(res, 4)
    pairs["heterodimer_dg"] = [thermo.heterodimer_dg(a, b) for a, b in zip(pairs["forward"], pairs["reverse"])]
    pairs["score"] = (0.5 * pairs["species_coverage"] + 0.35 * pairs["resolution_unique"] + 0.1 * pairs["seq_coverage"]
                      + 0.05 * (1 - (pairs["tm_f"] - pairs["tm_r"]).abs() / p["max_tm_diff"])
                      - 0.02 * np.log2(pairs["deg_f"] * pairs["deg_r"]) - 0.02 * (pairs["heterodimer_dg"] < -9))
    pairs = pairs.sort_values("score", ascending=False).head(p["max_pairs"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"D{i + 1:02d}" for i in range(len(pairs))])
    pairs = pairs.drop(columns=["f", "r", "f_end_col", "r_start_col"])
    for side in ("forward", "reverse"):
        props = pairs[side].map(thermo.properties)
        pairs[f"{side[0]}_hairpin_dg"] = props.map(lambda x: x["hairpin_dg"])
        pairs[f"{side[0]}_self_dimer_dg"] = props.map(lambda x: x["self_dimer_dg"])
    pairs["score"] = pairs["score"].round(4)
    pairs.to_csv(out / "pairs.csv", index=False)
    (out / "params.json").write_text(json.dumps({"dataset": dataset, "marker": marker, "run_id": run_id,
                                                  "design": p, "pcr_rules": pc, "n_sequences": n,
                                                  "n_species": int(len(sp_names))}, indent=1))
    log.info("  %d pairs saved -> %s", len(pairs), out / "pairs.csv")
    if evaluate_top:
        evaluate_pairs(dataset, marker, pairs.head(evaluate_top), out, recs, progress)
    return pairs


def evaluate_pairs(dataset: str, marker: str, pairs: pd.DataFrame, out, recs=None, progress=None) -> pd.DataFrame:
    """Run the full PCR engine and barcode-gap statistics on candidate pairs."""
    from .align import run_mafft
    from .stats import barcode_gap, distance_matrix
    from .utils import write_fasta
    log = get_logger("design")
    adir = paths.alignment_dir(dataset, marker)
    if recs is None:
        excl = qc.excluded_ids(dataset, marker)
        recs = [r for r in read_fasta(adir / "aligned.fasta") if r[0] not in excl]
    aligned = dict(recs)
    meta = pd.read_csv(adir / "aligned_metadata.csv").set_index("seq_id")
    ungapped = [(k, v.replace("-", "")) for k, v in recs]
    rows = []
    for i, r in enumerate(pairs.itertuples()):
        if progress:
            progress("Evaluating top pairs", i / len(pairs))
        lo = int(r.product_len_est * 0.7)
        pair = PrimerPair(r.pair_id, r.forward, r.reverse, lo, int(r.product_len_est * 1.4) + 20, marker)
        hits = run_on_sequences(ungapped, pair, aligned, thermo_binding=False)
        hits = hits.join(meta[["species", "genus", "family"]], on="seq_id")
        amp = hits[hits["status"] == "amplified"]
        testable_sp = hits.loc[hits["status"] != "site_not_covered", "species"].nunique()
        row = {"pair_id": r.pair_id, "species_amplified": amp["species"].nunique(), "species_testable": testable_sp,
               "seqs_amplified": len(amp), "seqs_testable": int((hits["status"] != "site_not_covered").sum()),
               "median_insert_len": amp["insert_len"].median() if len(amp) else None}
        if amp["species"].nunique() >= 2:
            tmp = out / f"_tmp_{r.pair_id}.fasta"
            write_fasta(tmp, zip(amp["seq_id"], amp["insert"]))
            al = run_mafft(tmp, out / f"_tmp_{r.pair_id}_aln.fasta", extra=["--auto"])
            tmp.unlink(missing_ok=True)
            (out / f"_tmp_{r.pair_id}_aln.fasta").unlink(missing_ok=True)
            m = amp.set_index("seq_id").loc[[a for a, _ in al], ["species", "genus", "family"]].reset_index()
            Xa = alignment_matrix([s for _, s in al])
            D = distance_matrix(Xa, settings()["resolution"].get("distance_model", "k2p"), min_overlap=20)
            _, summ = barcode_gap(D, m)
            row.update(resolved_species=summ["resolved_species"], resolved_genus=summ["resolved_genus"],
                       unique_amplicon=summ["unique_amplicon"], median_gap=summ["median_gap"])
        rows.append(row)
        log.info("  %s: %d/%d species amplified", r.pair_id, row["species_amplified"], testable_sp)
    ev = pairs[["pair_id", "forward", "reverse"]].merge(pd.DataFrame(rows), on="pair_id")
    ev.to_csv(out / "evaluated_pairs.csv", index=False)
    return ev
