"""Taxonomic resolution of a PCR product: variation within vs between species."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo

from . import paths
from .align import run_mafft
from .config import settings
from .stats import alignment_matrix, barcode_gap, distance_matrix
from .utils import get_logger, read_fasta


def run_resolution(dataset: str, marker: str, primer: str, model: str | None = None) -> dict:
    log = get_logger("resolution")
    pdir, rdir = paths.pcr_dir(dataset, marker, primer), paths.resolution_dir(dataset, marker, primer)
    hits = pd.read_csv(pdir / "hits.csv")
    amp = read_fasta(pdir / "amplicons.fasta")
    amp = [(n, s) for n, s in amp if s]
    if len(amp) < 2:
        raise ValueError("Fewer than 2 amplicons: nothing to compare.")
    model = model or settings()["resolution"].get("distance_model", "k2p")
    log.info("Resolution for %s/%s/%s: aligning %d amplicons (primers removed)", dataset, marker, primer, len(amp))
    recs = run_mafft(pdir / "amplicons.fasta", rdir / "amplicons_aligned.fasta", extra=["--auto"])
    ids = [n for n, _ in recs]
    meta = hits.set_index("seq_id").loc[ids, ["species", "genus", "family"]].reset_index()
    X = alignment_matrix([s for _, s in recs])
    D = distance_matrix(X, model, min_overlap=min(50, max(10, X.shape[1] // 3)))
    gap, summary = barcode_gap(D, meta)
    gap.to_csv(rdir / "barcode_gap.csv", index=False)
    sp = meta["species"].values
    iu = np.triu_indices(len(ids), 1)
    same = sp[iu[0]] == sp[iu[1]]
    dvals = D[iu]
    intra = pd.DataFrame({"species": sp[iu[0]][same], "seq_a": np.array(ids)[iu[0]][same],
                          "seq_b": np.array(ids)[iu[1]][same], "distance": dvals[same]})
    intra.to_csv(rdir / "intra_distances.csv", index=False)
    bins = np.arange(0, 0.505, 0.005)
    hi, _ = np.histogram(dvals[same][np.isfinite(dvals[same])], bins=bins)
    he, _ = np.histogram(np.clip(dvals[~same][np.isfinite(dvals[~same])], 0, 0.4999), bins=bins)
    pd.DataFrame({"bin_start": bins[:-1], "intra": hi, "inter": he}).to_csv(rdir / "distance_histogram.csv", index=False)
    M = pd.DataFrame(D, index=ids, columns=ids)
    spm = M.groupby(sp).min().T.groupby(sp).min()
    spm.to_csv(rdir / "species_min_distance.csv")
    # neighbour-joining tree on one representative (medoid) per species
    try:
        reps = []
        for s in dict.fromkeys(sp):
            idx = np.flatnonzero(sp == s)
            sub = np.nan_to_num(D[np.ix_(idx, idx)], nan=1.0)
            reps.append(idx[int(np.argmin(sub.sum(1)))])
        Dr = np.nan_to_num(D[np.ix_(reps, reps)], nan=float(np.nanmax(D)))
        names = [sp[i].replace(" ", "_") for i in reps]
        tri = [[float(Dr[i, j]) for j in range(i + 1)] for i in range(len(reps))]
        tree = DistanceTreeConstructor().nj(DistanceMatrix(names, tri))
        Phylo.write(tree, str(rdir / "tree_nj.nwk"), "newick")
    except Exception as e:
        log.warning("Tree not built: %s", e)
    summary.update({"model": model, "n_amplicons": len(ids), "alignment_length": int(X.shape[1])})
    (rdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("  %d species: %d with a barcode gap, %d with unique amplicons, %d identical groups -> %s",
             summary["n_species"], summary["resolved_species"], summary["unique_amplicon"],
             len(summary["identical_groups"]), rdir)
    return summary
