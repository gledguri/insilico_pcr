"""HTML report and primer-pair comparison table."""
from __future__ import annotations

import html
import json
import time

import pandas as pd
import plotly.io as pio

from . import paths, plots
from .utils import get_logger


def load_run(dataset: str, marker: str, primer: str) -> dict:
    d = paths.pcr_dir(dataset, marker, primer)
    run = {"name": primer, "summary": json.loads((d / "summary.json").read_text()),
           "hits": pd.read_csv(d / "hits.csv"), "species": pd.read_csv(d / "species_summary.csv"),
           "mmF": pd.read_csv(d / "mismatch_matrix_forward.csv", index_col=0),
           "mmR": pd.read_csv(d / "mismatch_matrix_reverse.csv", index_col=0)}
    r = paths.resolution_dir(dataset, marker, primer)
    if (r / "summary.json").exists():
        run["res_summary"] = json.loads((r / "summary.json").read_text())
        run["gap"] = pd.read_csv(r / "barcode_gap.csv")
        run["hist"] = pd.read_csv(r / "distance_histogram.csv")
    return run


def compare_runs(dataset: str, marker: str, primers: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for p in primers or paths.list_pcr_runs(dataset, marker):
        run = load_run(dataset, marker, p)
        s = run["summary"]
        t = max(1, s["species_testable"])
        row = {"primer": p, "forward": s["primer_pair"]["forward"], "reverse": s["primer_pair"]["reverse"],
               "median_insert_len": s["median_insert_len"], "species_testable": s["species_testable"],
               "species_amplified": s["species_amplified"], "species_amplified_pct": round(100 * s["species_amplified"] / t, 1),
               "seqs_amplified": s["seqs_amplified"], "seqs_testable": s["seqs_testable"],
               "tm_f": f'{s["forward"]["tm_min"]}-{s["forward"]["tm_max"]}', "tm_r": f'{s["reverse"]["tm_min"]}-{s["reverse"]["tm_max"]}',
               "degeneracy": s["forward"]["degeneracy"] * s["reverse"]["degeneracy"]}
        if "res_summary" in run:
            rs = run["res_summary"]
            n = max(1, rs["n_species"])
            row.update(species_resolved=rs["resolved_species"], species_resolved_pct=round(100 * rs["resolved_species"] / t, 1),
                       resolved_genus=rs["resolved_genus"], unique_amplicon=rs["unique_amplicon"],
                       identical_groups=" | ".join(", ".join(g) for g in rs["identical_groups"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _fig(fig, first: list[bool]) -> str:
    out = pio.to_html(fig, full_html=False, include_plotlyjs=first[0], config={"displaylogo": False})
    first[0] = False
    return out


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    return df.head(max_rows).to_html(index=False, classes="t", border=0, na_rep="", float_format=lambda v: f"{v:.3g}")


def build_report(dataset: str, marker: str, primers: list[str] | None = None) -> str:
    log = get_logger("report")
    primers = primers or paths.list_pcr_runs(dataset, marker)
    first = [True]
    parts = []
    qdir = paths.qc_dir(dataset, marker)
    if (qdir / "qc_summary.json").exists():
        q = json.loads((qdir / "qc_summary.json").read_text())
        counts = pd.read_csv(qdir / "species_counts.csv")
        parts.append(f"<h2>Sequences</h2><p>{q['kept']} of {q['raw']} records kept after QC; "
                     f"{q['species_with_data']} of {q['species_total']} species have sequences. "
                     f"Removed: {html.escape(json.dumps(q['removed_by_reason']))}. "
                     f"No data: <i>{html.escape(', '.join(q['species_without_data']) or 'none')}</i>.</p>")
        parts.append(_fig(plots.species_counts(counts), first))
    if primers:
        comp = compare_runs(dataset, marker, primers)
        comp.to_csv(paths.report_dir(dataset, marker) / "primer_comparison.csv", index=False)
        parts.append("<h2>Primer pairs compared</h2>" + _fig(plots.compare_bars(comp), first) + _table(comp))
    for p in primers:
        run = load_run(dataset, marker, p)
        s = run["summary"]
        fam = run["species"].set_index("species")["family"]
        parts.append(f"<h2>{html.escape(p)}</h2><p class='mono'>F {s['primer_pair']['forward']}<br>R {s['primer_pair']['reverse']}</p>"
                     f"<p>{s['species_amplified']} of {s['species_testable']} testable species amplified; "
                     f"{s['seqs_amplified']} of {s['seqs_testable']} testable sequences. Median insert {s['median_insert_len']} bp. "
                     f"Not testable (records end before a primer site): <i>{html.escape(', '.join(s['species_not_testable']) or 'none')}</i>. "
                     f"Fail: <i>{html.escape(', '.join(s['species_fail']) or 'none')}</i>.</p>")
        parts.append(_fig(plots.status_by_species(run["species"]), first))
        if len(run["mmF"]):
            parts.append(_fig(plots.mismatch_heatmap(run["mmF"], s["primer_pair"]["forward"], "Forward primer mismatches", family=fam), first))
            parts.append(_fig(plots.mismatch_heatmap(run["mmR"], s["primer_pair"]["reverse"], "Reverse primer mismatches", family=fam), first))
        if "gap" in run:
            rs = run["res_summary"]
            parts.append(f"<h3>Resolution ({rs['model']})</h3><p>{rs['resolved_species']} of {rs['n_species']} amplified species have a barcode gap; "
                         f"{rs['unique_amplicon']} have amplicons not shared with any other species. Identical groups: "
                         f"<i>{html.escape(' | '.join(', '.join(g) for g in rs['identical_groups']) or 'none')}</i>.</p>")
            parts.append(_fig(plots.barcode_gap_scatter(run["gap"]), first))
            parts.append(_fig(plots.distance_hist(run["hist"]), first))
            parts.append(_table(run["gap"].drop(columns=["shares_identical_with"])))
    css = """body{font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#0e1e25}
    h1{font-weight:600}h2{margin-top:40px;border-top:1px solid #d2dee2;padding-top:18px}p{line-height:1.5;max-width:90ch}
    .mono{font-family:'IBM Plex Mono',Menlo,monospace}table.t{border-collapse:collapse;font-size:12px;margin:12px 0;display:block;overflow-x:auto}
    table.t th,table.t td{padding:4px 8px;border-bottom:1px solid #e3eaed;text-align:left;white-space:nowrap}"""
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ispcr report {html.escape(dataset)} {html.escape(marker)}</title>"
           f"<style>{css}</style></head><body><h1>In silico PCR report</h1><p>Dataset <b>{html.escape(dataset)}</b>, "
           f"marker <b>{html.escape(marker)}</b>, generated {time.strftime('%Y-%m-%d %H:%M')}. Tables behind every figure are in "
           f"<span class='mono'>results/pcr/</span> and <span class='mono'>results/resolution/</span>.</p>{''.join(parts)}</body></html>")
    out = paths.report_dir(dataset, marker) / f"report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    out.write_text(doc)
    log.info("Report written: %s", out)
    return str(out)
