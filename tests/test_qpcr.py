import numpy as np

from ispcr import qpcr
from ispcr.utils import revcomp


def test_classify_rules():
    R = qpcr.rules({"detect_primer_mm": 2, "detect_primer_3prime": 0, "detect_probe_mm": 1, "block_primer_3prime": 2,
                    "block_primer_mm": 4, "block_pair_mm": 5, "block_probe_mm": 3})
    out = qpcr.classify(fm=[0, 1, 0, 3, 1, 0], f3=[0, 0, 2, 0, 1, 0], rm=[0, 0, 0, 2, 0, 0], r3=[0, 0, 0, 0, 0, 0],
                        pm=[0, 1, 0, 0, 0, 3], has_probe=True, R=R)
    assert list(out) == ["detected", "detected", "blocked", "blocked", "at_risk", "blocked"]


def test_locate_both_orientations():
    rng = np.random.default_rng(3)
    seq = "".join(rng.choice(list("ACGT"), 300))
    F, Rv, P = seq[20:40], revcomp(seq[150:170]), seq[80:105]
    loc = qpcr.locate(seq, qpcr.Assay("a", F, Rv, P))
    assert loc["orientation"] == "+"
    assert loc["sites"]["forward"] == (20, 20, "+") and loc["sites"]["reverse"] == (150, 20, "-")
    assert loc["sites"]["probe"] == (80, 25, "+")
    # the same assay written for the other strand
    loc = qpcr.locate(seq, qpcr.Assay("b", Rv, F, revcomp(P)))
    assert loc["orientation"] == "-"
    assert loc["sites"]["forward"] == (150, 20, "-") and loc["sites"]["probe"] == (80, 25, "-")


def test_oligo_checks_flags_bad_probe():
    a = qpcr.Assay("c", "TGGTGCACTTCTTGGTGATGAT", "AGGAATGAGTCAGTTCCCAAAGC", "GGGGTGATCGTTACAGCGCACGCTTTCG")
    chk = qpcr.oligo_checks(a, 121).set_index(["oligo", "check"])
    assert not chk.loc[("probe", "no G at the 5′ end"), "ok"]
    assert not chk.loc[("probe", "longest run of one base"), "ok"]
    assert chk.loc[("pair", "amplicon length (target)"), "ok"]
