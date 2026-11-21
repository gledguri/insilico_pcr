"""In silico qPCR: is a primer (+ probe) assay specific to ONE target species?

The assay is placed on the alignment with the target species' consensus, and every record is
then read at exactly those alignment columns. Mismatches are therefore counted however many
there are, and records that end before an oligo site are recognised as not covered.

Outcome per record (limits in settings.yaml -> qpcr.rules):
  detected     amplifies and the probe binds: few mismatches, none at the primer 3' ends
  at_risk      between the two: late or weak amplification is possible
  blocked      no amplification or no probe signal: enough mismatches at a primer 3' end,
               in one primer, in both primers together, or in the probe
  not_covered  the record does not reach every oligo site (partial sequence)

For the target species "detected" is what you want (sensitivity); for every other species
"blocked" is what you want (specificity).

Design (design_assays) finds oligos that match every record of the target and differ from the
other species, with the mismatches placed at the primer 3' ends where they stop amplification.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from . import paths, qc, thermo
from .config import settings
from .pcr import mismatch_windows, primer_mask
from .stats import alignment_matrix
from .utils import IUPAC, clean_primer, degeneracy, encode, get_logger, read_fasta, revcomp, write_fasta

GAP, UNK = 5, 6
CHARS = np.array(list("ACGTN-?"))
OUTCOMES = ["detected", "at_risk", "blocked", "not_covered"]
REFERENCES = [
    "Bustin et al. 2009, Clin Chem 55:611 (MIQE guidelines)",
    "Kwok et al. 1990, Nucleic Acids Res 18:999 (effect of primer-template mismatches, 3' end)",
    "Stadhouders et al. 2010, J Mol Diagn 12:109 (mismatches in 5' nuclease assays)",
    "Wilcox et al. 2013, PLoS ONE 8:e59520 (primer specificity for eDNA species detection)",
    "Thalinger et al. 2021, Environmental DNA 3:823 (validation scale for eDNA assays)",
]


# ---------------------------------------------------------------- assays and settings
@dataclass
class Assay:
    name: str
    forward: str
    reverse: str
    probe: str = ""
    target: str = ""
    marker: str = ""
    probe_type: str = "TaqMan"  # TaqMan, MGB or none (SYBR Green)
    reference: str = ""

    def __post_init__(self):
        self.forward, self.reverse = clean_primer(self.forward), clean_primer(self.reverse)
        self.probe = clean_primer(self.probe) if self.probe else ""
        if not self.probe:
            self.probe_type = "none"


def library() -> dict:
    p = paths.config_dir() / "qpcr_assays.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def save_assay(name: str, entry: dict) -> None:
    p = paths.config_dir() / "qpcr_assays.yaml"
    head = "".join(line for line in (p.read_text().splitlines(True) if p.exists() else []) if line.startswith("#"))
    lib = library()
    lib[name] = entry
    p.write_text(head + yaml.safe_dump(lib, sort_keys=False))


def delete_assays(names: list[str]) -> list[str]:
    """Remove assays from qpcr_assays.yaml (keeps the comment header). Old file kept as qpcr_assays.yaml.bak."""
    p = paths.config_dir() / "qpcr_assays.yaml"
    lib = library()
    gone = [n for n in names if n in lib]
    if gone:
        text = p.read_text()
        p.with_suffix(".yaml.bak").write_text(text)
        head = "".join(line for line in text.splitlines(True) if line.startswith("#"))
        for n in gone:
            del lib[n]
        p.write_text(head + (yaml.safe_dump(lib, sort_keys=False) if lib else ""))
    return gone


def get_assay(name: str) -> Assay:
    e = library()[name]
    return Assay(name, e["forward"], e["reverse"], e.get("probe", "") or "", e.get("target", ""), e.get("marker", ""),
                 e.get("probe_type", "TaqMan"), e.get("reference", ""))


def rules(over: dict | None = None) -> dict:
    r = {"three_prime_window": 5, "coverage_slack": 2, "detect_primer_mm": 2, "detect_primer_3prime": 0,
         "detect_probe_mm": 1, "block_primer_3prime": 2, "block_primer_mm": 4, "block_pair_mm": 5, "block_probe_mm": 3}
    r.update(settings().get("qpcr", {}).get("rules", {}))
    r.update(over or {})
    return r


def checks_cfg() -> dict:
    return settings().get("qpcr", {}).get("checks", {})


def classify(fm, f3, rm, r3, pm, has_probe: bool, R: dict) -> np.ndarray:
    fm, f3, rm, r3, pm = (np.asarray(x) for x in (fm, f3, rm, r3, pm))
    blocked = ((f3 >= R["block_primer_3prime"]) | (r3 >= R["block_primer_3prime"]) | (fm >= R["block_primer_mm"])
               | (rm >= R["block_primer_mm"]) | (fm + rm >= R["block_pair_mm"]))
    detected = (fm <= R["detect_primer_mm"]) & (rm <= R["detect_primer_mm"]) & (f3 <= R["detect_primer_3prime"]) \
        & (r3 <= R["detect_primer_3prime"])
    if has_probe:
        blocked = blocked | (pm >= R["block_probe_mm"])
        detected = detected & (pm <= R["detect_probe_mm"])
    return np.where(blocked, "blocked", np.where(detected, "detected", "at_risk"))


# ---------------------------------------------------------------- alignment context
_CACHE: dict = {}


def _alignment(dataset: str, marker: str):
    a = paths.alignment_dir(dataset, marker)
    excl = qc.excluded_ids(dataset, marker)
    key = (dataset, marker, (a / "aligned.fasta").stat().st_mtime, frozenset(excl))
    if key not in _CACHE:
        recs = read_fasta(a / "aligned.fasta")
        meta = pd.read_csv(a / "aligned_metadata.csv").set_index("seq_id")
        recs = [(n, s) for n, s in recs if n in meta.index and n not in excl]
        X = alignment_matrix([s for _, s in recs])
        nongap = X != GAP
        L = X.shape[1]
        first = np.argmax(nongap, 1)
        last = L - 1 - np.argmax(nongap[:, ::-1], 1)
        _CACHE.clear()
        _CACHE[key] = {"ids": [n for n, _ in recs], "X": X, "meta": meta.loc[[n for n, _ in recs]],
                       "first": first, "last": last, "excluded": sorted(excl)}
    return _CACHE[key]


def _context(dataset: str, marker: str, target: str) -> dict:
    al = _alignment(dataset, marker)
    X = al["X"]
    sp = al["meta"]["species"].values
    trow = np.flatnonzero(sp == target)
    if not len(trow):
        raise ValueError(f"No records of {target} in the {marker} alignment (after exclusions).")
    T = X[trow]
    L = X.shape[1]
    cols = np.arange(L)
    cov = (cols[None, :] >= al["first"][trow, None]) & (cols[None, :] <= al["last"][trow, None])
    counts = np.stack([((T == k) & cov).sum(0) for k in range(4)])
    gaps = ((T == GAP) & cov).sum(0)
    ncov = cov.sum(0)
    tcons = np.where(counts.sum(0) > gaps, counts.argmax(0), GAP).astype(np.uint8)
    tcons[ncov == 0] = UNK
    tcols = np.flatnonzero(tcons < 4)
    return {**al, "species": sp, "target": target, "trow": trow, "tcons": tcons, "tcols": tcols,
            "tseq": "".join(CHARS[tcons[tcols]]), "tcov": ncov}


# ---------------------------------------------------------------- placing an assay on the alignment
def _pen(codes, oligo: str, three_right: bool, w: int):
    m = mismatch_windows(codes, primer_mask(oligo))
    if not len(m):
        return None, None
    tot = m.sum(1)
    t3 = m[:, -w:].sum(1) if three_right else m[:, :w].sum(1)
    return tot, tot + 2 * t3


def locate(tseq: str, assay: Assay, w: int = 5, max_len: int = 2000) -> dict:
    """Where the primers (and probe) sit on the target consensus. Positions are 0-based in tseq.
    sense '+': the oligo reads like the alignment; '-': the oligo is the reverse complement."""
    codes = encode(tseq)
    best = None
    # '+': forward primer on the alignment strand; '-': the assay was written for the other strand
    for orient, left, right in (("+", assay.forward, revcomp(assay.reverse)), ("-", assay.reverse, revcomp(assay.forward))):
        tl, pl = _pen(codes, left, True, w)
        tr, pr = _pen(codes, right, False, w)
        if tl is None or tr is None:
            continue
        ci = np.argsort(pl)[:150]
        cj = np.argsort(pr)[:150]
        P = pl[ci][:, None] + pr[cj][None, :]
        ok = (cj[None, :] >= ci[:, None] + len(left)) & (cj[None, :] + len(right) - ci[:, None] <= max_len)
        if not ok.any():
            continue
        P = np.where(ok, P, 10 ** 6)
        a, b = np.unravel_index(np.argmin(P), P.shape)
        if best is None or P[a, b] < best[0]:
            best = (P[a, b], orient, int(ci[a]), len(left), int(cj[b]), len(right), int(tl[ci[a]]), int(tr[cj[b]]))
    if best is None:
        raise ValueError("The primers could not be placed on the target sequence.")
    _, orient, li, ll, ri, rl, lmm, rmm = best
    sites = {"forward": (li, ll, "+") if orient == "+" else (ri, rl, "-"),
             "reverse": (ri, rl, "-") if orient == "+" else (li, ll, "+")}
    out = {"orientation": orient, "left": (li, ll), "right": (ri, rl), "sites": sites}
    if assay.probe:
        lo, hi = li + ll, ri  # between the primers
        if hi - lo < len(assay.probe):
            lo, hi = li, ri + rl
        sub = encode(tseq[lo:hi])
        bestp = None
        for sense, o in (("+", assay.probe), ("-", revcomp(assay.probe))):
            m = mismatch_windows(sub, primer_mask(o))
            if len(m):
                k = int(np.argmin(m.sum(1)))
                if bestp is None or m[k].sum() < bestp[0]:
                    bestp = (int(m[k].sum()), lo + k, sense)
        if bestp:
            sites["probe"] = (bestp[1], len(assay.probe), bestp[2])
    return out


def _read_site(ctx: dict, pos: int, k: int, sense: str, oligo: str, w: int, slack: int) -> dict:
    """Mismatches of every record against one oligo, read at the oligo's alignment columns."""
    X, tcons, first, last = ctx["X"], ctx["tcons"], ctx["first"], ctx["last"]
    n = X.shape[0]
    cols = ctx["tcols"][pos:pos + k]
    a, b = int(cols[0]), int(cols[-1])
    seg, exp = X[:, a:b + 1], tcons[a:b + 1]
    is_base = exp < 4
    pidx = np.maximum(np.cumsum(is_base) - 1, 0)  # columns where the target has a gap count for the base before
    mask = primer_mask(oligo if sense == "+" else revcomp(oligo))
    span = np.arange(a, b + 1)
    inside = (span[None, :] >= first[:, None]) & (span[None, :] <= last[:, None])
    vec = np.zeros((n, k), dtype=bool)
    for j in range(seg.shape[1]):
        x = seg[:, j]
        if is_base[j]:
            ok = np.where(x == GAP, False, mask[pidx[j], np.minimum(x, 4)])
        else:
            ok = (x == GAP) | (exp[j] == UNK)
        vec[:, pidx[j]] |= ~ok & inside[:, j]
    missing = (~inside[:, is_base]).sum(1)
    templ = CHARS[seg[:, is_base]]
    if sense == "-":
        vec = vec[:, ::-1]
        templ = np.vectorize(lambda c: {"A": "T", "C": "G", "G": "C", "T": "A"}.get(c, c))(templ)[:, ::-1]
    sites = ["".join(r) for r in templ]
    notes = []
    for i in range(n):
        idx = np.flatnonzero(vec[i])
        notes.append(";".join(f"{q + 1}{oligo[q]}>{sites[i][q] if sites[i][q] not in IUPAC.get(oligo[q], 'ACGT') else 'ins'}"
                              for q in idx))
    return {"cols": (a + 1, b + 1), "tot": vec.sum(1), "t3": vec[:, -w:].sum(1), "covered": missing <= slack,
            "site": sites, "notes": notes, "vec": vec}


def _dist_to_target(Xs: np.ndarray, trow_mask: np.ndarray, model: str = "k2p") -> tuple[np.ndarray, float]:
    """Smallest distance of every record to any target record, over the given columns; and the
    largest distance among target records."""
    valid = Xs < 4
    purine = (Xs == 0) | (Xs == 2)
    n = Xs.shape[0]
    best = np.full(n, np.nan)
    tidx = np.flatnonzero(trow_mask)
    intra = 0.0
    for t in tidx:
        v = valid & valid[t]
        nv = v.sum(1)
        neq = (Xs != Xs[t]) & v
        with np.errstate(divide="ignore", invalid="ignore"):
            if model == "p":
                d = neq.sum(1) / nv
            else:
                ts = (neq & (purine == purine[t])).sum(1) / nv
                tv = (neq & (purine != purine[t])).sum(1) / nv
                A, B = 1 - 2 * ts - tv, 1 - 2 * tv
                d = np.where((A > 0) & (B > 0), -0.5 * np.log(np.where(A > 0, A, 1)) - 0.25 * np.log(np.where(B > 0, B, 1)), np.nan)
                d = np.where(neq.sum(1) == 0, 0.0, d)
        d = np.where(nv >= 20, d, np.nan)
        best = np.fmin(best, d)
        dt = d[tidx]
        if np.isfinite(dt).any():
            intra = max(intra, float(np.nanmax(dt)))
    return best, intra


# ---------------------------------------------------------------- oligo design checks
def oligo_checks(assay: Assay, amplicon_len: float | None = None, probe_overlaps: bool | None = None) -> pd.DataFrame:
    c = checks_cfg()
    rows = []

    def add(oligo, check, value, guide, ok):
        rows.append({"oligo": oligo, "check": check, "value": value, "guideline": guide, "ok": bool(ok)})

    props = {}
    for lab, s in (("forward", assay.forward), ("reverse", assay.reverse)):
        p = thermo.properties(s)
        props[lab] = p
        lo, hi = c.get("primer_len", [18, 25])
        add(lab, "length", f"{p['length']} nt", f"{lo}–{hi} nt", lo <= p["length"] <= hi)
        lo, hi = c.get("primer_tm", [57, 63])
        add(lab, "Tm", f"{p['tm_min']:.1f} °C", f"{lo}–{hi} °C", lo <= p["tm_min"] and p["tm_max"] <= hi)
        lo, hi = c.get("gc", [35, 65])
        add(lab, "GC", f"{p['gc']:.0f} %", f"{lo}–{hi} %", lo <= p["gc"] <= hi)
        g3 = p["gc_clamp_3prime"]
        add(lab, "G/C in last 5 bases (3′ end)", g3, f"1–{c.get('max_3prime_gc', 3)}", 1 <= g3 <= c.get("max_3prime_gc", 3))
        add(lab, "longest run of one base", p["max_homopolymer"], f"≤ {c.get('max_run', 3)}", p["max_homopolymer"] <= c.get("max_run", 3))
        add(lab, "hairpin ΔG", f"{p['hairpin_dg']} kcal/mol", f"≥ {c.get('min_hairpin_dg', -3)}", p["hairpin_dg"] >= c.get("min_hairpin_dg", -3))
        add(lab, "self-dimer ΔG", f"{p['self_dimer_dg']} kcal/mol", f"≥ {c.get('min_dimer_dg', -9)}", p["self_dimer_dg"] >= c.get("min_dimer_dg", -9))
        add(lab, "degenerate bases", p["degeneracy"], "none (1-fold)", p["degeneracy"] == 1)
    dtm = abs(props["forward"]["tm_min"] - props["reverse"]["tm_min"])
    add("pair", "Tm difference F vs R", f"{dtm:.1f} °C", f"≤ {c.get('max_tm_diff', 2)} °C", dtm <= c.get("max_tm_diff", 2))
    hd = thermo.heterodimer_dg(assay.forward, assay.reverse)
    add("pair", "primer-dimer ΔG (F × R)", f"{hd} kcal/mol", f"≥ {c.get('min_dimer_dg', -9)}", hd >= c.get("min_dimer_dg", -9))
    if amplicon_len:
        lo, hi = c.get("amplicon", [50, 150])
        add("pair", "amplicon length (target)", f"{amplicon_len:.0f} bp", f"{lo}–{hi} bp", lo <= amplicon_len <= hi)
    if assay.probe:
        p = thermo.properties(assay.probe)
        mgb = assay.probe_type.upper() == "MGB"
        lo, hi = c.get("probe_len_mgb" if mgb else "probe_len_taqman", [13, 20] if mgb else [18, 30])
        add("probe", "length", f"{p['length']} nt", f"{lo}–{hi} nt ({'MGB' if mgb else 'TaqMan'})", lo <= p["length"] <= hi)
        mean_p = (props["forward"]["tm_min"] + props["reverse"]["tm_min"]) / 2
        d = p["tm_min"] - mean_p
        lo, hi = c.get("probe_tm_above_primers", [5, 10])
        if mgb:
            add("probe", "Tm above primers", f"{d:+.1f} °C (without MGB)", "MGB raises Tm by 10–20 °C; not modelled", True)
        else:
            add("probe", "Tm above primers", f"{d:+.1f} °C", f"+{lo} to +{hi} °C", lo <= d <= hi)
        lo, hi = c.get("probe_gc", [30, 80])
        add("probe", "GC", f"{p['gc']:.0f} %", f"{lo}–{hi} %", lo <= p["gc"] <= hi)
        add("probe", "no G at the 5′ end", assay.probe[0], "not G (quenches the reporter dye)", assay.probe[0] != "G")
        nc, ng = assay.probe.count("C"), assay.probe.count("G")
        add("probe", "more C than G", f"C {nc} / G {ng}", "C ≥ G (else use the other strand)", nc >= ng)
        add("probe", "longest run of one base", p["max_homopolymer"], f"≤ {c.get('max_run', 3)}", p["max_homopolymer"] <= c.get("max_run", 3))
        add("probe", "hairpin ΔG", f"{p['hairpin_dg']} kcal/mol", f"≥ {c.get('min_hairpin_dg', -3)}", p["hairpin_dg"] >= c.get("min_hairpin_dg", -3))
        worst = min(thermo.heterodimer_dg(assay.probe, assay.forward), thermo.heterodimer_dg(assay.probe, assay.reverse))
        add("probe", "probe-primer dimer ΔG", f"{worst} kcal/mol", f"≥ {c.get('min_dimer_dg', -9)}", worst >= c.get("min_dimer_dg", -9))
        if probe_overlaps is not None:
            add("probe", "does not overlap a primer", "overlaps" if probe_overlaps else "no", "between the primers", not probe_overlaps)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluate one assay
def evaluate(dataset: str, marker: str, assay: Assay, target: str | None = None, rule_over: dict | None = None,
             save: bool = True) -> dict:
    """Check an assay against every record: is the target detected, and is every other species blocked?"""
    log = get_logger("qpcr")
    target = target or assay.target
    if not target:
        raise ValueError("Choose the target species.")
    assay.target, assay.marker = target, marker
    R = rules(rule_over)
    w = R["three_prime_window"]
    ctx = _context(dataset, marker, target)
    loc = locate(ctx["tseq"], assay, w)
    oligos = {"forward": assay.forward, "reverse": assay.reverse, "probe": assay.probe}
    res = {}
    for k, (pos, ln, sense) in loc["sites"].items():
        res[k] = _read_site(ctx, pos, ln, sense, oligos[k], w, R["coverage_slack"])
    has_probe = "probe" in res
    n = len(ctx["ids"])
    zero = np.zeros(n, dtype=int)
    covered = res["forward"]["covered"] & res["reverse"]["covered"] & (res["probe"]["covered"] if has_probe else True)
    pm = res["probe"]["tot"] if has_probe else zero
    out = classify(res["forward"]["tot"], res["forward"]["t3"], res["reverse"]["tot"], res["reverse"]["t3"], pm, has_probe, R)
    out = np.where(covered, out, "not_covered")
    # product length per record (alignment columns from the left site to the right site)
    tc = ctx["tcols"]
    (li, ll), (ri, rl) = loc["left"], loc["right"]
    A, B = int(tc[li]), int(tc[ri + rl - 1])
    X = ctx["X"]
    plen = (X[:, A:B + 1] != GAP).sum(1)
    ins_a, ins_b = int(tc[li + ll - 1]) + 1, int(tc[ri])
    is_t = ctx["species"] == target
    dist, intra = _dist_to_target(X[:, ins_a:ins_b], is_t & covered) if ins_b - ins_a >= 20 and (is_t & covered).any() else (np.full(n, np.nan), np.nan)
    meta = ctx["meta"]
    hits = pd.DataFrame({
        "seq_id": ctx["ids"], "species": ctx["species"], "genus": meta["genus"].values, "family": meta["family"].values,
        "is_target": is_t, "outcome": out, "covered": covered, "product_len": np.where(covered, plen, np.nan),
        "f_mm": res["forward"]["tot"], "f_mm3": res["forward"]["t3"], "f_mismatches": res["forward"]["notes"],
        "r_mm": res["reverse"]["tot"], "r_mm3": res["reverse"]["t3"], "r_mismatches": res["reverse"]["notes"],
        "p_mm": pm if has_probe else np.nan, "p_mismatches": res["probe"]["notes"] if has_probe else "",
        "f_site": res["forward"]["site"], "r_site": res["reverse"]["site"],
        "p_site": res["probe"]["site"] if has_probe else "",
        "distance_to_target": np.round(dist, 5)})
    hits["total_mm"] = hits["f_mm"] + hits["r_mm"] + (hits["p_mm"].fillna(0) if has_probe else 0)
    sp = species_table(hits, dataset, marker, target)
    tgt = hits[hits["is_target"] & hits["covered"]]
    nt = sp[~sp["is_target"]]
    tested = nt[nt["result"].isin(["detected", "at_risk", "blocked"])]
    closest = tested.sort_values(["min_total_mm", "min_distance_pct"]).head(1)
    det_t = tgt[tgt["outcome"] == "detected"]
    amp_len = float(det_t["product_len"].median()) if len(det_t) else (float(tgt["product_len"].median()) if len(tgt) else None)
    pos = {k: f"{v['cols'][0]}-{v['cols'][1]}" for k, v in res.items()}
    overl = None
    if has_probe:
        pa, pb = res["probe"]["cols"]
        overl = any(not (pb < v["cols"][0] or pa > v["cols"][1]) for k, v in res.items() if k != "probe")
    checks = oligo_checks(assay, amp_len, overl)
    sens = len(det_t) / len(tgt) if len(tgt) else 0.0
    nt_det, nt_risk = nt.loc[nt["result"] == "detected", "species"].tolist(), nt.loc[nt["result"] == "at_risk", "species"].tolist()
    if not len(tgt):
        verdict = "target not testable"
    elif sens < 0.9:
        verdict = "target not reliably detected"
    elif nt_det:
        verdict = "not specific"
    elif nt_risk:
        verdict = "specific, with species at risk"
    else:
        verdict = "specific"
    summary = {
        "assay": asdict(assay), "target": target, "marker": marker, "rules": R, "orientation": loc["orientation"],
        "site_columns": "; ".join(f"{k[0].upper()} {v}" for k, v in pos.items()),
        "target_records": int(is_t.sum()), "target_covered": int(len(tgt)), "target_detected": int(len(det_t)),
        "target_perfect": int((tgt["total_mm"] == 0).sum()), "target_at_risk": int((tgt["outcome"] == "at_risk").sum()),
        "target_blocked": int((tgt["outcome"] == "blocked").sum()), "sensitivity": round(sens, 4),
        "target_max_intra_pct": round(100 * intra, 3) if intra == intra else None,
        "amplicon_len": amp_len,
        "nontarget_species_tested": int(len(tested)), "nontarget_detected": nt_det, "nontarget_at_risk": nt_risk,
        "nontarget_blocked": int((nt["result"] == "blocked").sum()),
        "nontarget_not_testable": nt.loc[nt["result"] == "not_testable", "species"].tolist(),
        "species_no_sequences": nt.loc[nt["result"] == "no_sequences", "species"].tolist(),
        "closest_species": closest["species"].iloc[0] if len(closest) else None,
        "closest_total_mm": int(closest["min_total_mm"].iloc[0]) if len(closest) else None,
        "closest_distance_pct": float(closest["min_distance_pct"].iloc[0]) if len(closest) and closest["min_distance_pct"].notna().iloc[0] else None,
        "checks_passed": int(checks["ok"].sum()), "checks_total": int(len(checks)),
        "verdict": verdict, "excluded_records": ctx["excluded"], "built": time.strftime("%Y-%m-%d %H:%M"),
    }
    if save:
        d = paths.qpcr_dir(dataset, marker, assay.name)
        hits.to_csv(d / "hits.csv", index=False)
        sp.to_csv(d / "species_summary.csv", index=False)
        checks.to_csv(d / "checks.csv", index=False)
        recs = dict(read_fasta(paths.alignment_dir(dataset, marker) / "aligned.fasta"))
        write_fasta(d / "target_amplicons.fasta", [(i, recs[i][A:B + 1].replace("-", "")) for i in tgt["seq_id"]])
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log.info("qPCR %s for %s on %s/%s: target %d/%d detected; non-target detected %d, at risk %d -> %s",
                 assay.name, target, dataset, marker, len(det_t), len(tgt), len(nt_det), len(nt_risk), d)
    return {"summary": summary, "hits": hits, "species": sp, "checks": checks}


def species_table(hits: pd.DataFrame, dataset: str, marker: str, target: str) -> pd.DataFrame:
    rows = []
    for s, g in hits.groupby("species", sort=False):
        c = g[g["covered"]]
        vc = c["outcome"].value_counts()
        res = "detected" if vc.get("detected", 0) else "at_risk" if vc.get("at_risk", 0) else "blocked" if vc.get("blocked", 0) else "not_testable"
        best = c.sort_values(["total_mm", "distance_to_target"]).head(1)
        b = best.iloc[0] if len(best) else None
        rows.append({"species": s, "genus": g["genus"].iloc[0], "family": g["family"].iloc[0], "is_target": s == target,
                     "result": res, "n_records": len(g), "n_covered": len(c), "detected": int(vc.get("detected", 0)),
                     "at_risk": int(vc.get("at_risk", 0)), "blocked": int(vc.get("blocked", 0)),
                     "min_f_mm": c["f_mm"].min() if len(c) else np.nan, "min_r_mm": c["r_mm"].min() if len(c) else np.nan,
                     "min_p_mm": c["p_mm"].min() if len(c) else np.nan,
                     "min_total_mm": c["total_mm"].min() if len(c) else np.nan,
                     "min_distance_pct": round(100 * c["distance_to_target"].min(), 2) if len(c) and c["distance_to_target"].notna().any() else np.nan,
                     "closest_record": b["seq_id"] if b is not None else "",
                     "f_mismatches": b["f_mismatches"] if b is not None else "", "r_mismatches": b["r_mismatches"] if b is not None else "",
                     "p_mismatches": b["p_mismatches"] if b is not None else ""})
    sp = pd.DataFrame(rows)
    cp = paths.qc_dir(dataset, marker) / "species_counts.csv"
    if cp.exists():
        counts = pd.read_csv(cp)
        miss = counts[~counts["species"].isin(sp["species"])]
        sp = pd.concat([sp, pd.DataFrame({"species": miss["species"], "family": miss.get("family"), "is_target": miss["species"] == target,
                                          "result": "no_sequences", "n_records": 0})], ignore_index=True)
    order = {"detected": 0, "at_risk": 1, "blocked": 2, "not_testable": 3, "no_sequences": 4}
    sp["_o"] = sp["result"].map(order)
    sp = sp.sort_values(["is_target", "_o", "min_total_mm"], ascending=[False, True, True]).drop(columns="_o").reset_index(drop=True)
    return sp


def load_run(dataset: str, marker: str, name: str) -> dict:
    d = paths.qpcr_dir(dataset, marker, name)
    return {"name": name, "summary": json.loads((d / "summary.json").read_text()),
            "hits": pd.read_csv(d / "hits.csv", keep_default_na=False, na_values=[""]),
            "species": pd.read_csv(d / "species_summary.csv"), "checks": pd.read_csv(d / "checks.csv")}


def compare_runs(dataset: str, marker: str, names: list[str]) -> pd.DataFrame:
    rows = []
    for nme in names:
        s = load_run(dataset, marker, nme)["summary"]
        rows.append({"assay": nme, "target": s["target"], "verdict": s["verdict"],
                     "target_detected_pct": round(100 * s["sensitivity"], 1),
                     "target_records": f"{s['target_detected']} / {s['target_covered']}",
                     "nontarget_detected": len(s["nontarget_detected"]), "nontarget_at_risk": len(s["nontarget_at_risk"]),
                     "nontarget_blocked": s["nontarget_blocked"], "nontarget_tested": s["nontarget_species_tested"],
                     "closest_species": s["closest_species"], "closest_total_mm": s["closest_total_mm"],
                     "amplicon_len": s["amplicon_len"], "checks": f"{s['checks_passed']}/{s['checks_total']}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- design species-specific assays
def _gc(s: str) -> float:
    return 100 * sum(b in "GC" for b in s) / max(1, len(s))


def _run(s: str) -> int:
    return thermo.max_homopolymer(s)


def design_assays(dataset: str, marker: str, target: str, params: dict | None = None, top: int = 20,
                  progress: Callable[[str, float], None] | None = None) -> pd.DataFrame:
    """Find primer pairs (+ probe) that match every record of the target and are blocked in the other species."""
    log = get_logger("qpcr")
    p = {**settings().get("qpcr", {}).get("design", {}), **(params or {})}
    c = checks_cfg()
    R = rules()
    w = R["three_prime_window"]
    prog = progress or (lambda m, f: None)
    ctx = _context(dataset, marker, target)
    X, sp, trow = ctx["X"], ctx["species"], ctx["trow"]
    tcons, tcols, tseq = ctx["tcons"], ctx["tcols"], ctx["tseq"]
    first, last = ctx["first"], ctx["last"]
    n, L = X.shape
    log.info("qPCR design for %s on %s/%s: %d records (%d of the target)", target, dataset, marker, n, len(trow))
    prog("Comparing every record with the target", 0.02)
    colidx = np.arange(L)
    inside = (colidx[None, :] >= first[:, None]) & (colidx[None, :] <= last[:, None])
    MM = (X != tcons[None, :]) & (X != 4) & (tcons[None, :] != UNK) & inside
    cs = np.zeros((n, L + 1), dtype=np.int16)
    np.cumsum(MM, 1, out=cs[:, 1:])
    unk = np.concatenate([[0], np.cumsum(tcons == UNK)])
    nt = np.flatnonzero(sp != target)
    order = nt[np.argsort(sp[nt], kind="stable")]
    sps = sp[order]
    starts = np.flatnonzero(np.r_[True, sps[1:] != sps[:-1]]) if len(order) else np.array([], dtype=int)
    csN, firstN, lastN = cs[order], first[order], last[order]
    csT = cs[trow]

    def by_species(x):  # rows of non-target records (sorted) -> per species sums
        return np.add.reduceat(x.astype(np.int32), starts, axis=0) if len(starts) else np.zeros((0,) + x.shape[1:])

    # ---- candidate windows on the target consensus (positions in tseq)
    lo_len, hi_len = int(p.get("primer_min_len", 18)), int(p.get("primer_max_len", 25))
    tmax_mm = int(p.get("target_max_mismatches", 0))
    probe_type = str(p.get("probe_type", "TaqMan"))
    use_probe = probe_type.lower() != "none"
    plen_lo, plen_hi = c.get("probe_len_mgb" if probe_type.upper() == "MGB" else "probe_len_taqman", [18, 30])
    win = []
    for k in range(min(lo_len, plen_lo if use_probe else lo_len), max(hi_len, plen_hi if use_probe else hi_len) + 1):
        i = np.arange(0, len(tcols) - k + 1)
        if not len(i):
            continue
        A, B = tcols[i], tcols[i + k - 1]
        ok = (B - A + 1 <= k + 2) & (unk[B + 1] - unk[A] == 0)
        tot = csT[:, B + 1] - csT[:, A]
        cov = (first[trow][:, None] <= A[None, :]) & (last[trow][:, None] >= B[None, :])
        ntc = cov.sum(0)
        tmx = np.where(cov, tot, 0).max(0)
        ok &= (tmx <= tmax_mm) & (ntc > 0)
        win.append(pd.DataFrame({"i": i[ok], "k": k, "A": A[ok], "B": B[ok], "ntc": ntc[ok]}))
    W = pd.concat(win, ignore_index=True) if win else pd.DataFrame(columns=["i", "k", "A", "B", "ntc"])
    if not len(W):
        raise ValueError("No site is identical in every record of the target. Exclude odd target records or allow mismatches.")
    W = W[W["ntc"] >= max(1, float(p.get("min_target_coverage", 0.5)) * W["ntc"].max())].reset_index(drop=True)
    W["seq"] = [tseq[a:a + b] for a, b in zip(W["i"], W["k"])]
    W["gc"] = W["seq"].map(_gc)
    W["run"] = W["seq"].map(_run)
    prog("Melting temperatures", 0.1)
    tm_lo, tm_hi = c.get("primer_tm", [57, 63])
    prim = W[W["k"].between(lo_len, hi_len) & W["gc"].between(*c.get("gc", [35, 65])) & (W["run"] <= c.get("max_run", 3))].copy()
    prim["tm"] = prim["seq"].map(thermo.tm)
    prim = prim[prim["tm"].between(tm_lo, tm_hi)]
    max3 = c.get("max_3prime_gc", 3)
    Fc = prim[prim["seq"].map(lambda s: 1 <= sum(b in "GC" for b in s[-5:]) <= max3)].reset_index(drop=True)
    Rc = prim[prim["seq"].map(lambda s: 1 <= sum(b in "GC" for b in s[:5]) <= max3)].reset_index(drop=True)
    if not len(Fc) or not len(Rc):
        raise ValueError("No primer site passes the Tm/GC rules on the target. Widen them in settings.yaml (qpcr.checks).")

    def primer_arrays(C: pd.DataFrame, three_right: bool):
        i, k = C["i"].values, C["k"].values
        A, B = C["A"].values, C["B"].values
        tot = csN[:, B + 1] - csN[:, A]
        if three_right:
            t3 = csN[:, B + 1] - csN[:, tcols[i + k - w]]
        else:
            t3 = csN[:, tcols[i + w - 1] + 1] - csN[:, A]
        cov = (firstN[:, None] <= A[None, :]) & (lastN[:, None] >= B[None, :])
        blk = cov & ((t3 >= R["block_primer_3prime"]) | (tot >= R["block_primer_mm"]))
        return tot.astype(np.int16), t3.astype(np.int16), cov, blk

    prog("Scoring primer specificity", 0.2)
    keep = int(p.get("keep_per_direction", 300))
    arr = {}
    for lab, C, right in (("F", Fc, True), ("R", Rc, False)):
        tot, t3, cov, blk = primer_arrays(C, right)
        ncov, nblk = by_species(cov), by_species(blk)
        C["species_blocked"] = ((ncov > 0) & (nblk == ncov)).sum(0)
        C["margin"] = np.where(cov, np.minimum(tot + t3, 12), 12).min(0) if len(order) else 12
        C["rank"] = C["species_blocked"] * 100 + C["margin"]
        top_idx = np.argsort(-C["rank"].values, kind="stable")[:keep]
        arr[lab] = (C.iloc[top_idx].reset_index(drop=True), tot[:, top_idx], t3[:, top_idx], cov[:, top_idx], blk[:, top_idx])
    F, totF, f3F, covF, blkF = arr["F"]
    Rr, totR, r3R, covR, blkR = arr["R"]
    gap_min = plen_lo + 2 if use_probe else 0
    pmin, pmax = int(p.get("product_min", 60)), int(p.get("product_max", 150))
    prog("Pairing primers", 0.35)
    pairs = []
    for j in range(len(F)):
        fi, fk, ftm = F.at[j, "i"], F.at[j, "k"], F.at[j, "tm"]
        prod = Rr["i"].values + Rr["k"].values - fi
        ok = (Rr["i"].values >= fi + fk + gap_min) & (prod >= pmin) & (prod <= pmax) & \
             (np.abs(Rr["tm"].values - ftm) <= c.get("max_tm_diff", 2))
        jr = np.flatnonzero(ok)
        if not len(jr):
            continue
        tf = totF[:, j][:, None] + totR[:, jr]
        blk = blkF[:, j][:, None] | blkR[:, jr] | (tf >= R["block_pair_mm"])
        cov = covF[:, j][:, None] & covR[:, jr]
        amp = cov & (totF[:, j][:, None] <= R["detect_primer_mm"]) & (totR[:, jr] <= R["detect_primer_mm"]) & \
            (f3F[:, j][:, None] <= R["detect_primer_3prime"]) & (r3R[:, jr] <= R["detect_primer_3prime"])
        ncov, nblk, namp = by_species(cov), by_species(blk & cov), by_species(amp)
        remaining = ((ncov > 0) & (nblk < ncov)).sum(0)
        amplify = (namp > 0).sum(0)
        margin = np.where(cov, tf + f3F[:, j][:, None] + r3R[:, jr], 99).min(0) if len(order) else np.full(len(jr), 99)
        for q, r in enumerate(jr):
            pairs.append((j, int(r), int(remaining[q]), int(amplify[q]), int(margin[q]), int(prod[r])))
    if not pairs:
        raise ValueError(f"No primer pair gives a {pmin}–{pmax} bp product on the target. Widen the product length.")
    P = pd.DataFrame(pairs, columns=["jf", "jr", "species_not_blocked", "species_amplify", "margin", "product_len"])
    P["len_pen"] = (P["product_len"] - 100).abs()
    P = P.sort_values(["species_not_blocked", "species_amplify", "margin", "len_pen"], ascending=[True, True, False, True])
    # diversity: at most 3 pairs sharing a forward or reverse primer
    P = P[(P.groupby("jf").cumcount() < 3) & (P.groupby("jr").cumcount() < 3)].head(int(p.get("keep_pairs", 80))).reset_index(drop=True)
    # ---- probes inside each pair
    prog("Placing probes", 0.5)
    probes = W[W["k"].between(plen_lo, plen_hi) & W["gc"].between(*c.get("probe_gc", [30, 80])) & (W["run"] <= c.get("max_run", 3))].copy()
    tmcache: dict[str, float] = {}
    rows = []
    for q, pr in enumerate(P.itertuples()):
        f, r = F.iloc[pr.jf], Rr.iloc[pr.jr]
        base = {"forward": f["seq"], "reverse": revcomp(r["seq"]), "f_pos": int(f["i"]), "r_pos": int(r["i"]),
                "tm_f": round(f["tm"], 1), "tm_r": round(r["tm"], 1), "product_len": pr.product_len,
                "species_not_blocked_by_primers": pr.species_not_blocked, "margin_primers": pr.margin}
        fm, f3 = totF[:, pr.jf], f3F[:, pr.jf]
        rm, r3 = totR[:, pr.jr], r3R[:, pr.jr]
        cov = covF[:, pr.jf] & covR[:, pr.jr]
        best = None
        if use_probe:
            mean_tm = (f["tm"] + r["tm"]) / 2
            dlo, dhi = c.get("probe_tm_above_primers", [5, 10])
            cand = probes[(probes["i"] >= f["i"] + f["k"]) & (probes["i"] + probes["k"] <= r["i"])]
            for pc in cand.itertuples():
                if pc.seq not in tmcache:
                    tmcache[pc.seq] = thermo.tm(pc.seq)
                ptm = tmcache[pc.seq]
                if probe_type.upper() != "MGB" and not (mean_tm + dlo <= ptm <= mean_tm + dhi):
                    continue
                if probe_type.upper() == "MGB" and ptm < mean_tm - 10:
                    continue
                senses = [(o, s) for o, s in (("+", pc.seq), ("-", revcomp(pc.seq))) if s[0] != "G"]
                if not senses:
                    continue
                o, s = max(senses, key=lambda t: t[1].count("C") - t[1].count("G"))
                pm = csN[:, pc.B + 1] - csN[:, pc.A]
                pcov = cov & (firstN <= pc.A) & (lastN >= pc.B)
                outc = classify(fm, f3, rm, r3, pm, True, R)
                outc = np.where(pcov, outc, "not_covered")
                det = by_species((outc == "detected")[:, None])[:, 0] > 0 if len(order) else np.array([])
                risk = (by_species((outc == "at_risk")[:, None])[:, 0] > 0) & ~det if len(order) else np.array([])
                pmarg = int(np.where(pcov & (outc != "blocked"), pm, 99).min()) if len(order) else 99
                key = (int(det.sum()), int(risk.sum()), -pmarg, abs(ptm - mean_tm - (dlo + dhi) / 2))
                if best is None or key < best[0]:
                    best = (key, s, ptm, int(pc.i), o)
        if use_probe and best is None:
            continue
        if use_probe:
            (ndet, nrisk, negm, _), s, ptm, ppos, o = best
            base.update(probe=s, tm_p=round(ptm, 1), p_pos=ppos, probe_sense=o, species_detected=ndet, species_at_risk=nrisk)
        else:
            outc = np.where(cov, classify(fm, f3, rm, r3, np.zeros_like(fm), False, R), "not_covered")
            det = by_species((outc == "detected")[:, None])[:, 0] > 0
            risk = (by_species((outc == "at_risk")[:, None])[:, 0] > 0) & ~det
            base.update(probe="", tm_p=np.nan, species_detected=int(det.sum()), species_at_risk=int(risk.sum()))
        rows.append(base)
        if q % 10 == 0:
            prog("Placing probes", 0.5 + 0.2 * q / len(P))
    if not rows:
        raise ValueError("No probe fits between the primers with the Tm and sequence rules. Try MGB or a longer product.")
    D = pd.DataFrame(rows).sort_values(["species_detected", "species_at_risk", "species_not_blocked_by_primers", "margin_primers"],
                                       ascending=[True, True, True, False]).head(top).reset_index(drop=True)
    D.insert(0, "assay_id", [f"Q{i + 1:02d}" for i in range(len(D))])
    # ---- full check of each assay
    run_id = f"{paths.safe(target)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = paths.qpcr_design_dir(dataset, marker, run_id)
    ev = []
    for q, r in D.iterrows():
        prog("Checking the best assays", 0.7 + 0.3 * q / len(D))
        a = Assay(r["assay_id"], r["forward"], r["reverse"], r.get("probe", "") or "", target, marker, probe_type)
        e = evaluate(dataset, marker, a, target, save=False)["summary"]
        ev.append({"species_detected": len(e["nontarget_detected"]), "species_at_risk": len(e["nontarget_at_risk"]),
                   "sensitivity": e["sensitivity"], "verdict": e["verdict"], "target_detected": f"{e['target_detected']}/{e['target_covered']}",
                   "nontarget_detected": "; ".join(e["nontarget_detected"]), "nontarget_at_risk": "; ".join(e["nontarget_at_risk"]),
                   "closest_species": e["closest_species"], "closest_total_mm": e["closest_total_mm"],
                   "closest_distance_pct": e["closest_distance_pct"], "checks": f"{e['checks_passed']}/{e['checks_total']}",
                   "site_columns": e["site_columns"]})
    D = pd.concat([D.drop(columns=["species_detected", "species_at_risk"]), pd.DataFrame(ev)], axis=1)
    D = D.sort_values(["sensitivity", "species_detected", "species_at_risk", "closest_total_mm"],
                      ascending=[False, True, True, False]).reset_index(drop=True)
    D["assay_id"] = [f"Q{i + 1:02d}" for i in range(len(D))]
    cols = ["assay_id", "forward", "reverse", "probe", "product_len", "tm_f", "tm_r", "tm_p", "verdict", "target_detected",
            "species_detected", "species_at_risk", "nontarget_detected", "nontarget_at_risk", "closest_species", "closest_total_mm",
            "closest_distance_pct", "checks", "site_columns", "species_not_blocked_by_primers", "margin_primers"]
    D = D[[x for x in cols if x in D.columns]]
    D.to_csv(out / "assays.csv", index=False)
    (out / "params.json").write_text(json.dumps({"dataset": dataset, "marker": marker, "target": target, "run_id": run_id,
                                                  "design": p, "rules": R, "checks": c, "n_records": n,
                                                  "n_target_records": int(len(trow))}, indent=1, default=str))
    prog("Done", 1.0)
    log.info("  %d assays saved -> %s", len(D), out)
    return D
