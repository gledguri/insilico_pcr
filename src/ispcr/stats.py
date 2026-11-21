"""Distances, alignment column profiles and barcode-gap statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP = ord("-")


def alignment_matrix(seqs: list[str]) -> np.ndarray:
    """Aligned sequences -> uint8 matrix. A,C,G,T = 0..3, gap = 5, anything else (N, IUPAC) = 4."""
    L = max(len(s) for s in seqs)
    lut = np.full(256, 4, dtype=np.uint8)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    lut[GAP] = 5
    lut[ord(".")] = 5
    X = np.full((len(seqs), L), 5, dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = lut[np.frombuffer(s.encode(), dtype=np.uint8)]
    return X


def distance_matrix(X: np.ndarray, model: str = "k2p", min_overlap: int = 50) -> np.ndarray:
    """Pairwise distances with pairwise deletion of gaps/ambiguities. NaN where overlap < min_overlap."""
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    valid = X < 4
    purine = (X == 0) | (X == 2)
    for i in range(n - 1):
        v = valid[i] & valid[i + 1:]
        nv = v.sum(1)
        neq = (X[i] != X[i + 1:]) & v
        with np.errstate(divide="ignore", invalid="ignore"):
            if model == "p":
                d = neq.sum(1) / nv
            else:
                ts = (neq & (purine[i] == purine[i + 1:])).sum(1) / nv
                tv = (neq & (purine[i] != purine[i + 1:])).sum(1) / nv
                a, b = 1 - 2 * ts - tv, 1 - 2 * tv
                d = np.where((a > 0) & (b > 0), -0.5 * np.log(np.where(a > 0, a, 1)) - 0.25 * np.log(np.where(b > 0, b, 1)), np.nan)
                d = np.where(neq.sum(1) == 0, 0.0, d)
        d = np.where(nv >= min_overlap, d, np.nan)
        D[i, i + 1:] = d
        D[i + 1:, i] = d
    return D


def column_profile(X: np.ndarray) -> pd.DataFrame:
    counts = np.stack([(X == k).sum(0) for k in range(6)], axis=1)  # A C G T other gap
    bases = counts[:, :4].astype(float)
    tot = bases.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = bases / tot[:, None]
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=1)
    cons = np.array(list("ACGT"))[bases.argmax(1)]
    cons[tot == 0] = "-"
    return pd.DataFrame({"column": np.arange(1, X.shape[1] + 1), "A": counts[:, 0], "C": counts[:, 1],
                         "G": counts[:, 2], "T": counts[:, 3], "other": counts[:, 4], "gap": counts[:, 5],
                         "gap_fraction": counts[:, 5] / X.shape[0], "entropy": np.round(ent, 4),
                         "consensus": cons})


def species_min_matrix(D: np.ndarray, species: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    sp = pd.Series(species)
    names = list(dict.fromkeys(species))
    idx = {s: np.flatnonzero(sp.values == s) for s in names}
    M = np.full((len(names), len(names)), np.nan)
    intra = {}
    for a, sa in enumerate(names):
        ia = idx[sa]
        sub = D[np.ix_(ia, ia)]
        intra[sa] = np.nanmax(sub[np.triu_indices(len(ia), 1)]) if len(ia) > 1 and np.isfinite(sub[np.triu_indices(len(ia), 1)]).any() else np.nan
        for b in range(a + 1, len(names)):
            sub = D[np.ix_(ia, idx[names[b]])]
            m = np.nanmin(sub) if np.isfinite(sub).any() else np.nan
            M[a, b] = M[b, a] = m
    return pd.DataFrame(M, index=names, columns=names), pd.Series(intra)


def barcode_gap(D: np.ndarray, meta: pd.DataFrame, levels=("species", "genus", "family")) -> tuple[pd.DataFrame, dict]:
    """meta: one row per sequence (same order as D) with species/genus/family columns."""
    M, intra = species_min_matrix(D, meta["species"].tolist())
    names = list(M.index)
    tax = meta.drop_duplicates("species").set_index("species").loc[names]
    counts = meta["species"].value_counts()
    rows = []
    for s in names:
        others = M.loc[s].drop(s)
        nn = others.idxmin() if others.notna().any() else None
        mi = 0.0 if np.isnan(intra.get(s, np.nan)) else float(intra[s])
        identical = sorted(others[others == 0].index)
        row = {"species": s, "genus": tax.loc[s, "genus"], "family": tax.loc[s, "family"], "n_seqs": int(counts[s]),
               "max_intra": None if np.isnan(intra.get(s, np.nan)) else float(intra[s]),
               "min_inter": float(others.min()) if nn else None, "nearest_species": nn,
               "gap": (float(others.min()) - mi) if nn else None,
               "shares_identical_with": "; ".join(identical), "unique_amplicon": len(identical) == 0}
        for lvl in levels:
            key = tax[lvl] if lvl != "species" else pd.Series(names, index=names)
            diff_group = [o for o in others.index if key[o] != key[s] or lvl == "species"]
            vals = others[diff_group]
            row[f"resolved_{lvl}"] = bool((vals.dropna() > mi).all()) if len(vals.dropna()) else True
        rows.append(row)
    df = pd.DataFrame(rows)
    # groups of species sharing identical amplicons (connected components)
    groups, seen = [], set()
    for s in names:
        if s in seen or not df.set_index("species").loc[s, "shares_identical_with"]:
            continue
        comp, stack = set(), [s]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += [y for y in names if y not in comp and M.loc[x, y] == 0]
        seen |= comp
        groups.append(sorted(comp))
    summary = {"n_species": len(names),
               **{f"resolved_{lvl}": int(df[f"resolved_{lvl}"].sum()) for lvl in levels},
               "unique_amplicon": int(df["unique_amplicon"].sum()),
               "median_gap": float(np.nanmedian(df["gap"].astype(float))) if len(df) else None,
               "identical_groups": groups}
    return df, summary
