"""Quality control of extracted sequences, and flagging of likely misidentified records."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths, taxonomy
from .config import marker_cfg, settings
from .stats import distance_matrix
from .utils import get_logger, read_fasta, write_fasta


def _limits(marker: str) -> tuple[int, int]:
    if marker.lower() == "mitogenome":
        dl = settings()["download"]
        return dl["mitogenome_min_len"], dl["mitogenome_max_len"]
    m = marker_cfg(marker)
    return m["min_len"], m["max_len"]


def run_qc(dataset: str, marker: str) -> pd.DataFrame:
    log = get_logger("qc")
    sdir, qdir = paths.sequences_dir(dataset, marker), paths.qc_dir(dataset, marker)
    recs = dict(read_fasta(sdir / "sequences.fasta"))
    meta = pd.read_csv(sdir / "metadata.csv", dtype={"taxid": str})
    qcfg = settings()["qc"]
    lo, hi = _limits(marker)
    meta["seq"] = meta["seq_id"].map(recs)
    meta["n_ambiguous"] = meta["seq"].map(lambda s: sum(b not in "ACGT" for b in s))
    meta["ambiguous_fraction"] = meta["n_ambiguous"] / meta["length"]
    reason = pd.Series("", index=meta.index)
    reason[meta["length"] < lo] = f"shorter than {lo} bp"
    reason[meta["length"] > hi] = f"longer than {hi} bp"
    reason[(reason == "") & (meta["ambiguous_fraction"] > qcfg["max_ambiguous_fraction"])] = \
        f"more than {qcfg['max_ambiguous_fraction']:.0%} ambiguous bases"
    meta["qc_reason"] = reason
    keep = meta[reason == ""].copy()
    removed = meta[reason != ""].drop(columns="seq")
    # exact duplicates within a species (kept, but marked; counts are useful for haplotypes)
    keep["duplicate_of"] = ""
    for sp, g in keep.groupby("species"):
        first = {}
        for i, s in zip(g.index, g["seq"]):
            if s in first:
                keep.at[i, "duplicate_of"] = first[s]
            else:
                first[s] = keep.at[i, "seq_id"]
    write_fasta(qdir / "clean.fasta", zip(keep["seq_id"], keep["seq"]))
    keep.drop(columns="seq").to_csv(qdir / "clean_metadata.csv", index=False)
    removed.to_csv(qdir / "removed.csv", index=False)
    tax = taxonomy.load(dataset)
    counts = pd.DataFrame({"species": tax["accepted_name"], "family": tax["family"], "order": tax["order"]})
    counts["n_raw"] = counts["species"].map(meta["species"].value_counts()).fillna(0).astype(int)
    counts["n_clean"] = counts["species"].map(keep["species"].value_counts()).fillna(0).astype(int)
    counts["n_unique"] = counts["species"].map(keep[keep["duplicate_of"] == ""]["species"].value_counts()).fillna(0).astype(int)
    for src in ("gene_record", "mitogenome"):
        counts[f"n_{src}"] = counts["species"].map(keep[keep["source"] == src]["species"].value_counts()).fillna(0).astype(int)
    counts.to_csv(qdir / "species_counts.csv", index=False)
    summary = {"raw": len(meta), "kept": len(keep), "removed": len(removed),
               "removed_by_reason": removed["qc_reason"].value_counts().to_dict(),
               "exact_duplicates": int((keep["duplicate_of"] != "").sum()),
               "species_total": len(counts), "species_with_data": int((counts["n_clean"] > 0).sum()),
               "species_without_data": counts.loc[counts["n_clean"] == 0, "species"].tolist(),
               "length_limits": [lo, hi]}
    (qdir / "qc_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("QC %s/%s: kept %d of %d (%d species with data, %d without)", dataset, marker, len(keep), len(meta),
             summary["species_with_data"], len(summary["species_without_data"]))
    return keep


def flag_misidentified(X: np.ndarray, meta: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """A record is flagged when its closest record from another genus is closer than
    any record of its own species (or, for single-record species, nearly identical)."""
    D = distance_matrix(X, "p", min_overlap)
    sp, ge = meta["species"].values, meta["genus"].values
    rows = []
    for i in range(len(meta)):
        d = D[i].copy()
        d[i] = np.nan
        own = d[(sp == sp[i])]
        other = np.where(ge != ge[i], d, np.nan)
        if not np.isfinite(other).any():
            continue
        j = int(np.nanargmin(other))
        d_other = float(other[j])
        d_own = float(np.nanmin(own)) if np.isfinite(own).any() else None
        flag = (d_own is not None and d_other < d_own) or (d_own is None and d_other <= 0.005)
        if flag:
            reason = ("far from all records (possible poor quality or non-homologous)" if d_other > 0.1
                      else "closer to another genus than to its own species")
            rows.append({"seq_id": meta["seq_id"].iat[i], "species": sp[i], "reason": reason, "closest_other": meta["seq_id"].iat[j],
                         "closest_other_species": sp[j], "distance_other": round(d_other, 4),
                         "closest_own_distance": None if d_own is None else round(d_own, 4)})
    return pd.DataFrame(rows, columns=["seq_id", "species", "reason", "closest_other", "closest_other_species",
                                       "distance_other", "closest_own_distance"])


def excluded_ids(dataset: str, marker: str) -> set[str]:
    """Records left out of PCR, statistics, trees and design: your manual exclusions,
    plus flagged records when qc.remove_misidentified is true."""
    from . import exclude
    out = exclude.ids(dataset, marker)
    if settings()["qc"].get("remove_misidentified", False):
        p = paths.alignment_dir(dataset, marker) / "misid_flags.csv"
        if p.exists():
            out |= set(pd.read_csv(p)["seq_id"])
    return out
