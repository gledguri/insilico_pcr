"""Primer thermodynamics with primer3 (melting temperature, hairpins, dimers, binding to a site)."""
from __future__ import annotations

import numpy as np
import primer3

from .config import settings
from .utils import IUPAC, clean_primer, degeneracy, expand, revcomp


def _c() -> dict:
    t = settings().get("thermo", {})
    return {"mv_conc": t.get("mv_conc", 50), "dv_conc": t.get("dv_conc", 1.5),
            "dntp_conc": t.get("dntp_conc", 0.6), "dna_conc": t.get("dna_conc", 250)}


def tm(seq: str) -> float:
    return float(primer3.calc_tm(seq, **_c()))


def tm_range(primer: str) -> tuple[float, float]:
    tms = [tm(v) for v in expand(primer, 128)]
    return min(tms), max(tms)


def gc(primer: str) -> float:
    vals = [np.mean([b in "GC" for b in IUPAC.get(x, "ACGT")]) for x in clean_primer(primer)]
    return 100 * float(np.mean(vals))


def max_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def properties(primer: str) -> dict:
    p = clean_primer(primer)
    v = expand(p, 1)[0]
    lo, hi = tm_range(p)
    hp = primer3.calc_hairpin(v, **_c())
    hd = primer3.calc_homodimer(v, **_c())
    return {"sequence": p, "length": len(p), "degeneracy": degeneracy(p), "gc": round(gc(p), 1),
            "tm_min": round(lo, 1), "tm_max": round(hi, 1),
            "hairpin_tm": round(hp.tm, 1), "hairpin_dg": round(hp.dg / 1000, 2),
            "self_dimer_dg": round(hd.dg / 1000, 2),
            "gc_clamp_3prime": sum(b in "GCS" for b in p[-5:]), "max_homopolymer": max_homopolymer(p)}


def heterodimer_dg(a: str, b: str) -> float:
    return round(primer3.calc_heterodimer(expand(a, 1)[0], expand(b, 1)[0], **_c()).dg / 1000, 2)


def resolve_against(primer: str, site: str) -> str:
    """Pick, for each degenerate primer base, the variant that matches the site (if any)."""
    out = []
    for p, s in zip(clean_primer(primer), site.upper()):
        opts = IUPAC.get(p, "ACGT")
        out.append(s if s in opts else opts[0])
    return "".join(out)


def binding(primer: str, site: str) -> tuple[float, float]:
    """Tm and dG (kcal/mol) of the primer annealed to a template site given on the primer's own strand."""
    if not site or len(site) != len(primer):
        return np.nan, np.nan
    v = resolve_against(primer, site)
    r = primer3.calc_heterodimer(v, revcomp(site), **_c())
    return round(r.tm, 1), round(r.dg / 1000, 2)
