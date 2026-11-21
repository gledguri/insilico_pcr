"""Command line: ispcr <command> ...   (run `ispcr -h` for help)

Typical run for a species list data/species_lists/norway_marine_fish_100.txt:
  ispcr resolve  norway_marine_fish_100
  ispcr fetch    norway_marine_fish_100 --target mitogenome
  ispcr fetch    norway_marine_fish_100 --target 12S
  ispcr extract  norway_marine_fish_100 --marker 12S
  ispcr qc       norway_marine_fish_100 --marker 12S
  ispcr align    norway_marine_fish_100 --marker 12S
  ispcr pcr      norway_marine_fish_100 --marker 12S --primers MiFish-U Teleo
  ispcr resolution norway_marine_fish_100 --marker 12S --primers MiFish-U Teleo
  ispcr design   norway_marine_fish_100 --marker 12S
  ispcr report   norway_marine_fish_100 --marker 12S
  ispcr pdf-report norway_marine_fish_100 --marker 12S          (uses the choices saved in the app)
Species-specific qPCR (one target species):
  ispcr qpcr-design norway_marine_fish_100 --marker COI --target "Gadus morhua"
  ispcr qpcr     norway_marine_fish_100 --marker COI --target "Gadus morhua" --fwd ... --rev ... --probe ...
or everything at once:
  ispcr pipeline norway_marine_fish_100 --marker 12S --primers MiFish-U Teleo Riaz-12S-V5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths


def _list_path(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    for cand in (paths.species_lists_dir() / name, paths.species_lists_dir() / f"{name}.txt",
                 paths.species_lists_dir() / f"{name}.csv"):
        if cand.exists():
            return cand
    raise SystemExit(f"Species list not found: {name} (looked in {paths.species_lists_dir()})")


def _dataset(name: str) -> str:
    return Path(name).stem


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ispcr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create the folder structure")

    s = sub.add_parser("resolve", help="check species names (WoRMS + NCBI Taxonomy)")
    s.add_argument("species_list")

    s = sub.add_parser("fetch", help="download GenBank records from NCBI")
    s.add_argument("dataset")
    s.add_argument("--target", required=True, help="mitogenome, 12S, 16S, COI, CYTB ...")
    s.add_argument("--max", type=int, default=None, help="max records per species")
    s.add_argument("--force", action="store_true", help="ignore the cache and download again")
    s.add_argument("--offline", action="store_true", help="use only files already downloaded")

    for name, hlp in [("extract", "cut the marker out of downloaded records"), ("qc", "quality control"),
                      ("align", "align with MAFFT")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("dataset")
        s.add_argument("--marker", required=True)

    s = sub.add_parser("pcr", help="run in silico PCR")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--primers", nargs="*", default=[], help="names from config/primers.yaml")
    s.add_argument("--fwd", help="custom forward primer (5'->3')")
    s.add_argument("--rev", help="custom reverse primer (5'->3')")
    s.add_argument("--name", default="custom", help="name for the custom pair")
    s.add_argument("--product-min", type=int, default=0)
    s.add_argument("--product-max", type=int, default=2000)

    s = sub.add_parser("resolution", help="within vs between species distances of the amplicons")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--primers", nargs="+", required=True)

    s = sub.add_parser("design", help="design new primer pairs from the alignment")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--product-min", type=int)
    s.add_argument("--product-max", type=int)
    s.add_argument("--max-degeneracy", type=int)
    s.add_argument("--evaluate-top", type=int, default=10)

    s = sub.add_parser("tree", help="neighbour-joining tree from the alignment")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--mode", choices=["species", "all"], default="species", help="one record per species, or all records")
    s.add_argument("--columns", help="alignment region, e.g. 200-420 (default: whole alignment)")
    s.add_argument("--bootstrap", type=int, default=0)
    s.add_argument("--model", choices=["k2p", "p"], default="k2p")

    s = sub.add_parser("exclude", help="leave records out of PCR, statistics, trees and design")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("seq_ids", nargs="*", help="record IDs (accessions) to exclude")
    s.add_argument("--reason", default="excluded by hand")
    s.add_argument("--undo", action="store_true", help="put the records back instead")
    s.add_argument("--flagged", action="store_true", help="also exclude every record in misid_flags.csv")

    s = sub.add_parser("export-downloads", help="zip the name check + downloads of a dataset for offline reuse")
    s.add_argument("dataset")
    s = sub.add_parser("import-downloads", help="unpack a zip made by export-downloads")
    s.add_argument("zipfile")

    s = sub.add_parser("qpcr", help="check a species-specific qPCR assay (primers + probe) against every species")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--target", help="target species (default: the assay's target)")
    s.add_argument("--assay", help="name in config/qpcr_assays.yaml")
    s.add_argument("--fwd")
    s.add_argument("--rev")
    s.add_argument("--probe", default="")
    s.add_argument("--probe-type", default="TaqMan", choices=["TaqMan", "MGB", "none"])
    s.add_argument("--name", default="custom_assay")

    s = sub.add_parser("qpcr-design", help="design qPCR assays specific to one species")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--product-min", type=int)
    s.add_argument("--product-max", type=int)
    s.add_argument("--probe-type", choices=["TaqMan", "MGB", "none"])
    s.add_argument("--top", type=int, default=20)

    s = sub.add_parser("pdf-report", help="PDF report of every step, using the choices saved in the app")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)
    s.add_argument("--mode", choices=["pcr", "qpcr"], help="default: the mode switched on in the app")

    s = sub.add_parser("report", help="write an HTML report of all PCR runs for a marker")
    s.add_argument("dataset")
    s.add_argument("--marker", required=True)

    s = sub.add_parser("pipeline", help="resolve -> fetch -> extract -> qc -> align -> pcr -> resolution -> report")
    s.add_argument("species_list")
    s.add_argument("--marker", required=True)
    s.add_argument("--primers", nargs="*", default=[])
    s.add_argument("--no-mitogenomes", action="store_true", help="do not download mitogenomes")

    a = ap.parse_args(argv)
    paths.init_layout()

    if a.cmd == "init":
        print(f"Folders ready under {paths.ROOT}")
    elif a.cmd == "resolve":
        from .taxonomy import resolve_file
        resolve_file(_list_path(a.species_list))
    elif a.cmd == "fetch":
        from .fetch import fetch_dataset
        fetch_dataset(_dataset(a.dataset), a.target, a.max, a.force, offline=True if a.offline else None)
    elif a.cmd == "extract":
        from .extract import extract
        extract(_dataset(a.dataset), a.marker)
    elif a.cmd == "qc":
        from .qc import run_qc
        run_qc(_dataset(a.dataset), a.marker)
    elif a.cmd == "align":
        from .align import align
        align(_dataset(a.dataset), a.marker)
    elif a.cmd == "pcr":
        from .pcr import PrimerPair, get_pair, run_pcr
        pairs = [get_pair(n) for n in a.primers]
        if a.fwd and a.rev:
            pairs.append(PrimerPair(a.name, a.fwd, a.rev, a.product_min, a.product_max, a.marker))
        if not pairs:
            raise SystemExit("Give --primers NAME ... or --fwd/--rev")
        for p in pairs:
            run_pcr(_dataset(a.dataset), a.marker, p)
    elif a.cmd == "resolution":
        from .resolution import run_resolution
        for p in a.primers:
            run_resolution(_dataset(a.dataset), a.marker, p)
    elif a.cmd == "design":
        from .design import design
        params = {k: v for k, v in {"product_min": a.product_min, "product_max": a.product_max,
                                    "max_degeneracy": a.max_degeneracy}.items() if v is not None}
        design(_dataset(a.dataset), a.marker, params, a.evaluate_top)
    elif a.cmd == "tree":
        from .phylo import build_tree
        cols = tuple(int(x) for x in a.columns.split("-")) if a.columns else None
        info = build_tree(_dataset(a.dataset), a.marker, a.mode, cols, a.model, a.bootstrap)
        for r in info["not_monophyletic"]:
            print(f"not monophyletic: {r['species']} (mixed with {r['mixed_with']})")
    elif a.cmd == "exclude":
        import pandas as pd
        from . import exclude
        ds = _dataset(a.dataset)
        if a.undo:
            exclude.remove(ds, a.marker, a.seq_ids)
        else:
            meta = pd.read_csv(paths.alignment_dir(ds, a.marker) / "aligned_metadata.csv").set_index("seq_id")
            rows = [{"seq_id": i, "species": meta.loc[i, "species"] if i in meta.index else "", "reason": a.reason} for i in a.seq_ids]
            if a.flagged:
                f = pd.read_csv(paths.alignment_dir(ds, a.marker) / "misid_flags.csv")
                rows += [{"seq_id": r.seq_id, "species": r.species, "reason": f"flagged: {r.reason}"} for r in f.itertuples()]
            exclude.add(ds, a.marker, rows)
        print(f"{len(exclude.ids(ds, a.marker))} records excluded for {ds}/{a.marker}. Re-run pcr/resolution/design to update results.")
    elif a.cmd == "export-downloads":
        from .bundle import export_downloads
        print(export_downloads(_dataset(a.dataset)))
    elif a.cmd == "import-downloads":
        from .bundle import import_downloads
        print(import_downloads(Path(a.zipfile)))
    elif a.cmd == "qpcr":
        from . import qpcr
        if a.assay:
            assay = qpcr.get_assay(a.assay)
        elif a.fwd and a.rev:
            assay = qpcr.Assay(a.name, a.fwd, a.rev, a.probe if a.probe_type != "none" else "", a.target or "", a.marker, a.probe_type)
        else:
            raise SystemExit("Give --assay NAME or --fwd/--rev (and --probe)")
        s_ = qpcr.evaluate(_dataset(a.dataset), a.marker, assay, a.target)["summary"]
        print(f"{s_['verdict']}: target {s_['target_detected']}/{s_['target_covered']} records detected; "
              f"other species detected: {', '.join(s_['nontarget_detected']) or 'none'}; at risk: {', '.join(s_['nontarget_at_risk']) or 'none'}")
    elif a.cmd == "qpcr-design":
        from .qpcr import design_assays
        params = {k: v for k, v in {"product_min": a.product_min, "product_max": a.product_max, "probe_type": a.probe_type}.items() if v is not None}
        d = design_assays(_dataset(a.dataset), a.marker, a.target, params, a.top)
        print(d[["assay_id", "forward", "reverse", "probe", "product_len", "verdict", "target_detected", "nontarget_detected",
                 "nontarget_at_risk", "closest_species", "closest_total_mm"]].head(10).to_string(index=False))
    elif a.cmd == "pdf-report":
        from .pdfreport import build_pdf
        print(build_pdf(_dataset(a.dataset), a.marker, {"mode": a.mode} if a.mode else None))
    elif a.cmd == "report":
        from .report import build_report
        print(build_report(_dataset(a.dataset), a.marker))
    elif a.cmd == "pipeline":
        from .align import align
        from .extract import extract
        from .fetch import fetch_dataset
        from .pcr import get_pair, run_pcr
        from .qc import run_qc
        from .report import build_report
        from .resolution import run_resolution
        from .taxonomy import resolve_file
        lp = _list_path(a.species_list)
        ds = _dataset(lp.name)
        resolve_file(lp, ds)
        if a.marker.lower() == "mitogenome" or not a.no_mitogenomes:
            fetch_dataset(ds, "mitogenome")
        if a.marker.lower() != "mitogenome":
            fetch_dataset(ds, a.marker)
        extract(ds, a.marker)
        run_qc(ds, a.marker)
        align(ds, a.marker)
        for n in a.primers:
            run_pcr(ds, a.marker, get_pair(n))
            try:
                run_resolution(ds, a.marker, n)
            except ValueError as e:
                print(f"Resolution skipped for {n}: {e}", file=sys.stderr)
        if a.primers:
            print(build_report(ds, a.marker))


if __name__ == "__main__":
    main()
