"""Check species names against WoRMS and NCBI Taxonomy.

Output: data/taxonomy/<dataset>/species_resolved.csv with one row per input name.
status: ok | synonym (name updated) | check (fuzzy match, confirm it) | not_found
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import ncbi, paths
from .config import settings
from .utils import get_logger, read_species_list

WORMS = "https://www.marinespecies.org/rest/AphiaRecordsByMatchNames"
RANKS = ["class", "order", "family", "genus"]


def worms_match(names: list[str], marine_only: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(names), 50):
        chunk = names[i:i + 50]
        params = [("scientificnames[]", n) for n in chunk] + [("marine_only", str(marine_only).lower())]
        r = requests.get(WORMS, params=params, timeout=60)
        if r.status_code == 204:
            continue
        r.raise_for_status()
        for name, matches in zip(chunk, r.json()):
            if not matches:
                continue
            best = next((m for m in matches if m.get("status") == "accepted"), matches[0])
            out[name] = best
    return out


def ncbi_taxid(name: str) -> str | None:
    for field in ("Scientific Name", "All Names"):
        ids = ncbi.esearch("taxonomy", f'"{name}"[{field}]', retmax=5)
        if ids:
            return ids[0]
    return None


def ncbi_lineages(taxids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(taxids), 100):
        recs = ncbi.efetch_xml("taxonomy", taxids[i:i + 100])
        for r in recs:
            d = {"ncbi_name": str(r["ScientificName"])}
            for lin in r.get("LineageEx", []):
                if lin["Rank"] in RANKS:
                    d[f"ncbi_{lin['Rank']}"] = str(lin["ScientificName"])
            out[str(r["TaxId"])] = d
    return out


def resolve(names: list[str], dataset: str, force: bool = False) -> pd.DataFrame:
    """Reuses data/taxonomy/<dataset>/species_resolved.csv when it covers the same names (no internet needed)."""
    log = get_logger("resolve")
    prev = paths.taxonomy_dir(dataset) / "species_resolved.csv"
    if not force and prev.exists():
        old = pd.read_csv(prev, dtype={"ncbi_taxid": str})
        if set(old["input_name"]) == set(names):
            log.info("Names for '%s' already checked; reusing %s (use force to re-check online)", dataset, prev)
            return old
    cfg = settings().get("taxonomy", {})
    log.info("Resolving %d names for dataset '%s'", len(names), dataset)
    wm = {}
    if cfg.get("use_worms", True):
        try:
            wm = worms_match(names, cfg.get("marine_only", False))
        except requests.RequestException as e:
            log.warning("WoRMS not reachable (%s); using NCBI only", e)
    rows = []
    for n in names:
        m = wm.get(n)
        row = {"input_name": n, "accepted_name": n, "aphia_id": None, "worms_status": None,
               "match_type": None, "class": None, "order": None, "family": None, "genus": n.split()[0]}
        status, note = "ok", ""
        if m:
            row.update(accepted_name=m.get("valid_name") or m["scientificname"], aphia_id=m.get("valid_AphiaID"),
                       worms_status=m.get("status"), match_type=m.get("match_type"),
                       **{k: m.get(k) for k in ("class", "order", "family")})
            row["genus"] = row["accepted_name"].split()[0]
            if m.get("match_type") != "exact":
                status, note = "check", f"fuzzy match ({m.get('match_type')}) to {m['scientificname']}"
            elif row["accepted_name"] != n:
                status, note = "synonym", f"{m.get('status')} name; accepted is {row['accepted_name']}"
        elif cfg.get("use_worms", True):
            status, note = "check", "no WoRMS match"
        taxid = ncbi_taxid(row["accepted_name"]) or (ncbi_taxid(n) if n != row["accepted_name"] else None)
        row["ncbi_taxid"] = taxid
        if not taxid:
            status, note = "not_found", (note + "; " if note else "") + "no NCBI taxid"
        row["status"], row["note"] = status, note
        rows.append(row)
        log.info("  %-32s -> %-32s %s", n, row["accepted_name"], status)
    df = pd.DataFrame(rows)
    ids = [t for t in df["ncbi_taxid"].dropna().unique()]
    lin = ncbi_lineages(ids) if ids else {}
    for col in ["ncbi_name", "ncbi_class", "ncbi_order", "ncbi_family"]:
        df[col] = df["ncbi_taxid"].map(lambda t: lin.get(t, {}).get(col) if t else None)
    for r in ("class", "order", "family"):
        df[r] = df[r].fillna(df[f"ncbi_{r}"])
    diff = df["ncbi_name"].notna() & (df["ncbi_name"] != df["accepted_name"])
    df.loc[diff, "note"] = df.loc[diff, "note"].fillna("") + "; NCBI uses " + df.loc[diff, "ncbi_name"]
    df["note"] = df["note"].str.strip("; ")
    out = paths.taxonomy_dir(dataset) / "species_resolved.csv"
    df.to_csv(out, index=False)
    log.info("Saved %s  (%s)", out, df["status"].value_counts().to_dict())
    return df


def resolve_file(list_path: Path, dataset: str | None = None, force: bool = False) -> pd.DataFrame:
    dataset = dataset or Path(list_path).stem
    return resolve(read_species_list(Path(list_path)), dataset, force)


def load(dataset: str) -> pd.DataFrame:
    p = paths.taxonomy_dir(dataset) / "species_resolved.csv"
    if not p.exists():
        raise FileNotFoundError(f"No resolved names for '{dataset}'. Run: ispcr resolve <list>")
    return pd.read_csv(p, dtype={"ncbi_taxid": str})
