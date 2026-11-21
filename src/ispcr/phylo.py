"""Neighbour-joining trees from the alignment, with optional bootstrap support.

Trees are saved in data/alignments/<dataset>/<marker>/trees/ as Newick (+ a .json
describing how they were built and which species are not monophyletic).
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd
from Bio import Phylo

from . import paths, qc
from .stats import alignment_matrix, distance_matrix
from .utils import get_logger, read_fasta


def nj_newick(D: np.ndarray, labels: list[str]) -> str:
    """Neighbour joining (Saitou & Nei 1987), vectorised. Returns an unrooted Newick string."""
    D = D.astype(float).copy()
    nodes = list(labels)
    while len(nodes) > 2:
        n = len(nodes)
        r = D.sum(1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
        dij = D[i, j]
        li = 0.5 * dij + (r[i] - r[j]) / (2 * (n - 2)) if n > 2 else dij / 2
        li = min(max(li, 0.0), dij)
        lj = max(dij - li, 0.0)
        new = f"({nodes[i]}:{li:.6f},{nodes[j]}:{lj:.6f})"
        dk = 0.5 * (D[i] + D[j] - dij)
        D[i, :] = dk
        D[:, i] = dk
        D[i, i] = 0
        D = np.delete(np.delete(D, j, 0), j, 1)
        nodes[i] = new
        del nodes[j]
    d = D[0, 1] if len(nodes) == 2 else 0
    return f"({nodes[0]}:{d / 2:.6f},{nodes[1]}:{d / 2:.6f});" if len(nodes) == 2 else f"{nodes[0]};"


def _bipartitions(tree, all_leaves: frozenset, ref: str) -> set[frozenset]:
    out = set()
    for cl in tree.get_nonterminals():
        leaves = frozenset(t.name for t in cl.get_terminals())
        if 1 < len(leaves) < len(all_leaves) - 1:
            out.add(leaves if ref not in leaves else all_leaves - leaves)
    return out


def _fill_nan(D: np.ndarray) -> tuple[np.ndarray, int]:
    bad = int(np.isnan(D).sum() // 2)
    if bad:
        D = np.where(np.isnan(D), np.nanmax(D) if np.isfinite(D).any() else 1.0, D)
    return D, bad


def build_tree(dataset: str, marker: str, mode: str = "species", columns: tuple[int, int] | None = None,
               model: str = "k2p", bootstrap: int = 0, include_excluded: bool = False, min_coverage: float = 0.7,
               species: list[str] | None = None, progress=None) -> dict:
    """mode 'species': one record per species (the one covering most of the region);
    mode 'all': every record. columns: 1-based inclusive alignment region (None = whole alignment)."""
    log = get_logger("tree")
    a = paths.alignment_dir(dataset, marker)
    recs = read_fasta(a / "aligned.fasta")
    meta = pd.read_csv(a / "aligned_metadata.csv").set_index("seq_id")
    excl = set() if include_excluded else qc.excluded_ids(dataset, marker)
    recs = [(n, s) for n, s in recs if n not in excl and n in meta.index]
    if species:  # only these species (e.g. the ones selected in the alignment view)
        keep_sp = set(species)
        recs = [(n, s) for n, s in recs if meta.loc[n, "species"] in keep_sp]
    lo, hi = (columns[0] - 1, columns[1]) if columns else (0, max(len(s) for _, s in recs))
    recs = [(n, s[lo:hi]) for n, s in recs]
    if mode == "species":
        best = {}
        for n, s in recs:
            sp = meta.loc[n, "species"]
            cov = len(s) - s.count("-")
            if sp not in best or cov > best[sp][0]:
                best[sp] = (cov, (n, s))
        recs = [v[1] for v in best.values()]
    # leave out records that cover too little of the region: without overlap their distances are unknown
    covs = {n: len(s) - s.count("-") for n, s in recs}
    top = max(covs.values()) if covs else 0
    short = [n for n, c in covs.items() if c < max(20, min_coverage * top)]
    recs = [(n, s) for n, s in recs if n not in set(short)]
    if len(recs) < 3:
        raise ValueError("Need at least 3 records with sequence in this region to build a tree.")
    ids = [n for n, _ in recs]
    labels = [f"t{i}" for i in range(len(ids))]
    X = alignment_matrix([s for _, s in recs])
    min_ov = min(50, max(10, X.shape[1] // 4))
    D, n_nan = _fill_nan(distance_matrix(X, model, min_ov))
    t0 = time.time()
    main = Phylo.read(io.StringIO(nj_newick(D, labels)), "newick")
    main.root_at_midpoint()
    main.ladderize()
    if bootstrap:
        rng = np.random.default_rng(1)
        all_leaves, ref = frozenset(labels), labels[0]
        counts: dict[frozenset, int] = {}
        for b in range(bootstrap):
            if progress:
                progress(b / bootstrap)
            Xb = X[:, rng.integers(0, X.shape[1], X.shape[1])]
            Db, _ = _fill_nan(distance_matrix(Xb, model, min_ov))
            tb = Phylo.read(io.StringIO(nj_newick(Db, labels)), "newick")
            for bp in _bipartitions(tb, all_leaves, ref):
                counts[bp] = counts.get(bp, 0) + 1
        for cl in main.get_nonterminals():
            leaves = frozenset(t.name for t in cl.get_terminals())
            if 1 < len(leaves) < len(all_leaves) - 1:
                key = leaves if ref not in leaves else all_leaves - leaves
                cl.confidence = round(100 * counts.get(key, 0) / bootstrap)
    # rename leaves to record IDs and find species that are not monophyletic
    lab2id = dict(zip(labels, ids))
    for t in main.get_terminals():
        t.name = lab2id[t.name]
    nonmono = []
    if mode == "all":
        by_sp: dict[str, list] = {}
        for t in main.get_terminals():
            by_sp.setdefault(meta.loc[t.name, "species"], []).append(t)
        for sp, terms in by_sp.items():
            if len(terms) > 1 and not main.is_monophyletic(terms):
                mrca = main.common_ancestor(terms)
                inside = [meta.loc[x.name, "species"] for x in mrca.get_terminals()]
                intr = pd.Series([x for x in inside if x != sp]).value_counts()
                # records of this species that sit outside its largest clade are the likely culprits
                odd = []
                for t in terms:
                    sisters = [x for x in main.get_path(t)[-2].get_terminals() if x != t] if len(main.get_path(t)) > 1 else []
                    if sisters and all(meta.loc[x.name, "species"] != sp for x in sisters):
                        odd.append(t.name)
                nonmono.append({"species": sp, "records": len(terms), "mixed_with": "; ".join(intr.index[:5]),
                                "n_species_mixed": int(len(intr)), "records_outside_own_group": "; ".join(odd)})
    region = f"{lo + 1}-{hi}"
    name = f"{mode}_{region}_{model}_bs{bootstrap}"
    if species:
        import hashlib
        tag = hashlib.md5("|".join(sorted(species)).encode()).hexdigest()[:6]
        name += f"_{len(set(species))}sp-{tag}"
    out = paths.trees_dir(dataset, marker)
    Phylo.write(main, str(out / f"{name}.nwk"), "newick")
    info = {"name": name, "mode": mode, "region": region, "model": model, "bootstrap": bootstrap, "n_tips": len(ids),
            "pairs_without_overlap": n_nan, "excluded_records_left_out": len(excl),
            "short_records_left_out": len(short), "min_coverage": min_coverage,
            "built": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1),
            "not_monophyletic": nonmono, "species_selected": sorted(set(species)) if species else None}
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    log.info("Tree %s: %d tips, %d non-monophyletic species -> %s", name, len(ids), len(nonmono), out)
    return info


def list_trees(dataset: str, marker: str) -> list[str]:
    d = paths.trees_dir(dataset, marker)
    return sorted((p.stem for p in d.glob("*.nwk")), key=lambda n: (d / f"{n}.nwk").stat().st_mtime, reverse=True)


def load_tree(dataset: str, marker: str, name: str):
    d = paths.trees_dir(dataset, marker)
    info = json.loads((d / f"{name}.json").read_text()) if (d / f"{name}.json").exists() else {}
    return Phylo.read(str(d / f"{name}.nwk"), "newick"), info
