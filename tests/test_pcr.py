import numpy as np

from ispcr.pcr import PrimerPair, run_on_sequences
from ispcr.stats import alignment_matrix, distance_matrix
from ispcr.utils import degeneracy, expand, revcomp

F = "GTCGGTAAAACTCGTGCCAGC"
R = "CATAGTGGGGTATCTAATCCCAGTTTG"
rng = np.random.default_rng(1)
INSERT = "".join(rng.choice(list("ACGT"), 170))
TEMPLATE = "ACGT" * 10 + F + INSERT + revcomp(R) + "TTGCA" * 8


def run(seq, **kw):
    pair = PrimerPair("t", F, R, 150, 300)
    return run_on_sequences([("s", seq)], pair, cfg=kw or None, thermo_binding=False).iloc[0]


def test_perfect_match_amplifies():
    r = run(TEMPLATE)
    assert r.status == "amplified" and r.f_mm == 0 and r.r_mm == 0
    assert r.insert == INSERT and r.product_len == len(F) + len(INSERT) + len(R)


def test_reverse_strand_found():
    r = run(revcomp(TEMPLATE))
    assert r.status == "amplified" and r.strand == "-"


def test_three_prime_mismatch_fails():
    t = list(TEMPLATE)
    pos = 40 + len(F) - 1  # last base of the forward site
    t[pos] = "A" if t[pos] != "A" else "C"
    t[pos - 2] = "A" if t[pos - 2] != "A" else "C"
    r = run("".join(t))
    assert r.f_mm3 == 2 and r.status == "fail_mismatch"


def test_five_prime_mismatch_tolerated():
    t = list(TEMPLATE)
    t[41] = "A" if t[41] != "A" else "C"  # forward primer position 2
    r = run("".join(t))
    assert r.status == "amplified" and r.f_mm == 1 and r.f_mismatches.startswith("2")


def test_degenerate_primer():
    assert degeneracy("RYN") == 16 and len(expand("RY")) == 4


def test_k2p_identical_zero():
    X = alignment_matrix(["ACGTACGTAC" * 6, "ACGTACGTAC" * 6, "ACGTACGTAA" * 6])
    D = distance_matrix(X, "k2p", min_overlap=10)
    assert D[0, 1] == 0 and D[0, 2] > 0
