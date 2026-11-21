"""PDF report of the whole workflow, built from the choices saved in the app (selections.json).

Sections follow the app: name check, sequences & QC, alignment (one record per species with the primer
sites marked, and the chosen tree), in silico PCR or in silico qPCR (whichever is switched on), resolution,
primer / assay design (top N), comparison, and the mitogenome map. Figures are drawn with matplotlib.
Output: results/reports/<dataset>/<marker>/report_<time>.pdf
"""
from __future__ import annotations

import html
import io
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import paths, qc, selections  # noqa: E402
from .utils import IUPAC, get_logger, read_fasta, revcomp  # noqa: E402

ACCENT, GOLD, BAD, WARN, GOOD, GREY = "#0b6c77", "#d2961a", "#c43d33", "#b5761a", "#2c7a51", "#9aa9ae"
INK, MUTED, LINE, SOFT = "#0e1e25", "#587078", "#d6e0e3", "#e1f0f1"
NUC = {"A": "#3c9a4d", "C": "#2f6dcf", "G": "#d2961a", "T": "#cf4237"}
STATUS = {"amplified": GOOD, "fail_mismatch": BAD, "fail_length": WARN, "no_site": "#7a3d8f", "site_not_covered": GREY,
          "not_testable": GREY, "fails": BAD, "no_sequences": "#b9c2c5", "ok": GOOD, "synonym": ACCENT, "check": WARN,
          "not_found": BAD}
QSTATUS = {"target: detected": GOOD, "target: at_risk": WARN, "target: blocked": BAD, "target: not_testable": GREY, "detected": BAD, "at_risk": WARN, "blocked": GOOD, "not_testable": GREY, "no_sequences": "#b9c2c5", "not_covered": GREY}
MITO_COLORS = {"Complex I (NADH dehydrogenase)": "#0b6c77", "Complex III (cytochrome b)": "#7a3d8f",
               "Complex IV (cytochrome c oxidase)": "#c43d33", "Complex V (ATP synthase)": "#d2961a",
               "rRNA": "#2f6dcf", "tRNA": "#9aa9ae", "Control region": "#3c9a4d", "Other": "#5a6a70"}
PAIR_PALETTE = ["#0b6c77", "#d2961a", "#7a3d8f", "#2f6dcf", "#c43d33", "#3c9a4d", "#a64d79", "#1f4e79", "#b5761a"]

SECTIONS = ["names", "sequences", "alignment", "assay", "resolution", "design", "compare", "mitogenome"]
SECTION_TITLES = {"names": "Name check", "sequences": "Sequences & QC", "alignment": "Alignment",
                  "assay": "In silico PCR", "resolution": "Resolution", "design": "Primer design",
                  "compare": "Compare primer pairs", "mitogenome": "Mitogenome map"}
QSECTION_TITLES = {**SECTION_TITLES, "assay": "In silico qPCR", "resolution": "Resolution (target vs other species)",
                   "design": "Assay design (species-specific)", "compare": "Compare qPCR assays"}

MARGIN = 15 * mm
FRAME_W = A4[0] - 2 * MARGIN
FRAME_H = A4[1] - 2 * MARGIN - 8 * mm

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa9ae", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})


# ---------------------------------------------------------------- choices
def default_options(dataset: str, marker: str) -> dict:
    """The report's choices, taken from what was selected in the app (with sensible fallbacks)."""
    sel = selections.load(dataset)
    m = sel.get("markers", {}).get(marker, {})
    pr, qr = paths.list_pcr_runs(dataset, marker), paths.list_qpcr_runs(dataset, marker)

    def ok(v, opts):
        return v if v in opts else (opts[0] if opts else None)

    def best_pcr():
        best = None
        for r in pr:
            try:
                s_ = json.loads((paths.pcr_dir(dataset, marker, r) / "summary.json").read_text())
                k = (s_["species_amplified"], s_["seqs_amplified"])
                if best is None or k > best[0]:
                    best = (k, r)
            except (OSError, ValueError, KeyError):
                pass
        return best[1] if best else None

    pcr_run = m.get("pcr_run") if m.get("pcr_run") in pr else best_pcr()
    qpcr_run = ok(m.get("qpcr_run"), qr)
    res_ok = [r for r in pr if (paths.resolution_dir(dataset, marker, r) / "summary.json").exists()]
    res = m.get("resolution_run") if m.get("resolution_run") in res_ok else (pcr_run if pcr_run in res_ok else (res_ok[0] if res_ok else None))
    mode = sel.get("mode", "pcr")
    sites = [r for r in (m.get("align_runs") or []) if r in pr or (r.startswith("qPCR ") and r[5:] in qr)]
    if not sites:
        sites = [f"qPCR {qpcr_run}"] if mode == "qpcr" and qpcr_run else ([pcr_run] if pcr_run else [])
    from .phylo import list_trees
    trees = list_trees(dataset, marker) if (paths.alignment_dir(dataset, marker) / "aligned.fasta").exists() else []
    from . import mitomap
    mito_sp = mitomap.cached_species(dataset)
    return {
        "mode": mode, "sections": list(SECTIONS),
        "align_species": m.get("align_species") or [], "align_runs": sites, "tree": m.get("tree") if m.get("tree") in trees else None,
        "pcr_runs": [pcr_run] if pcr_run else [], "qpcr_run": qpcr_run,
        "resolution_runs": [res] if res else [],
        "design_run": ok(m.get("design_run"), paths.list_design_runs(dataset, marker)),
        "qpcr_design_run": ok(m.get("qpcr_design_run"), paths.list_qpcr_design_runs(dataset, marker)), "design_top": 20,
        "compare_runs": [r for r in (m.get("compare_runs") or pr) if r in pr],
        "compare_qpcr_runs": [r for r in (m.get("compare_qpcr_runs") or qr) if r in qr],
        "mito_species": next((x for x in (sel.get("mito_species"), m.get("qpcr_target") if mode == "qpcr" else None, "Gadus morhua")
                              if x in mito_sp), mito_sp[0] if mito_sp else None),
        "mito_record": sel.get("mito_record"),
        "mito_pairs": sel.get("mito_pairs") or [], "mito_assays": sel.get("mito_assays") or [],
    }


# ---------------------------------------------------------------- reportlab helpers
def _fonts():
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    have = pdfmetrics.getRegisteredFontNames()
    for name, f in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"), ("Sans-Italic", "DejaVuSans-Oblique.ttf"),
                    ("Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf"), ("Mono", "DejaVuSansMono.ttf")):
        if name not in have:
            pdfmetrics.registerFont(TTFont(name, str(d / f)))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-BoldItalic")


def _styles() -> dict:
    P = lambda n, **k: ParagraphStyle(n, **{"fontName": "Sans", "textColor": colors.HexColor(INK), **k})
    return {"title": P("title", fontName="Sans-Bold", fontSize=24, leading=29, spaceAfter=4),
            "subtitle": P("subtitle", fontSize=11, leading=15, textColor=colors.HexColor(MUTED), spaceAfter=14),
            "eyebrow": P("eyebrow", fontName="Sans-Bold", fontSize=7.5, leading=9, textColor=colors.HexColor(ACCENT), spaceAfter=2),
            "h1": P("h1", fontName="Sans-Bold", fontSize=16, leading=20, spaceAfter=6),
            "h2": P("h2", fontName="Sans-Bold", fontSize=10.5, leading=13.5, textColor=colors.HexColor(ACCENT), spaceBefore=8, spaceAfter=4),
            "body": P("body", fontSize=8.6, leading=12, spaceAfter=4),
            "small": P("small", fontSize=7, leading=9.2, textColor=colors.HexColor(MUTED), spaceAfter=3),
            "cell": P("cell", fontSize=6.6, leading=8.1),
            "cellh": P("cellh", fontName="Sans-Bold", fontSize=6.6, leading=8.1),
            "cellmono": P("cellmono", fontName="Mono", fontSize=6.4, leading=8.1),
            "mono": P("mono", fontName="Mono", fontSize=8.2, leading=11)}


def _esc(v) -> str:
    return html.escape(str(v))


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{v:.3g}"
    return str(v)


def _ital(names) -> str:
    return ", ".join(f"<i>{_esc(x)}</i>" for x in names)


class R:
    """Collects the flowables of the report."""

    def __init__(self):
        _fonts()
        self.S = _styles()
        self.story: list = []

    def p(self, text: str, style: str = "body"):
        self.story.append(Paragraph(text, self.S[style]))

    def h1(self, n: int | str, title: str, sub: str = ""):
        self.story.append(PageBreak())
        self.story.append(Paragraph(f"STEP {n}" if isinstance(n, int) else str(n), self.S["eyebrow"]))
        self.story.append(Paragraph(_esc(title), self.S["h1"]))
        if sub:
            self.story.append(Paragraph(sub, self.S["small"]))
        self.story.append(Spacer(1, 4))

    def h2(self, text: str):
        self.story.append(CondPageBreak(40 * mm))
        self.story.append(Paragraph(text, self.S["h2"]))

    def img(self, fig, width: float = 1.0, max_h: float = 0.88):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        iw, ih = ImageReader(buf).getSize()
        w = FRAME_W * width
        h = w * ih / iw
        if h > FRAME_H * max_h:
            h = FRAME_H * max_h
            w = h * iw / ih
        buf.seek(0)
        self.story.append(Image(buf, width=w, height=h))
        self.story.append(Spacer(1, 4))

    def kv(self, rows: list[tuple[str, str]], widths=(0.34, 0.66)):
        data = [[Paragraph(_esc(k), self.S["cellh"]), Paragraph(v, self.S["cell"])] for k, v in rows]
        t = Table(data, colWidths=[w * FRAME_W for w in widths])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(LINE)),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        self.story.append(t)
        self.story.append(Spacer(1, 6))

    def table(self, df: pd.DataFrame, max_rows: int | None = None, status_col: str | None = None, status_colors: dict | None = None,
              mono: tuple = (), italic: tuple = ("species", "accepted_name", "input_name", "nearest_species", "closest_species"),
              widths: list[float] | None = None, note: str | None = None):
        if df is None or not len(df):
            self.p("None.", "small")
            return
        total = len(df)
        if max_rows:
            df = df.head(max_rows)
        S = self.S
        rows = [[Paragraph(_esc(c).replace("_", " "), S["cellh"]) for c in df.columns]]
        for rec in df.itertuples(index=False):
            cells = []
            for c, v in zip(df.columns, rec):
                t = _esc(_fmt(v))
                if c in italic and t:
                    t = f"<i>{t}</i>"
                if status_col and c == status_col:
                    t = f"<font color='{(status_colors or STATUS).get(v, INK)}'><b>{t.replace('_', ' ')}</b></font>"
                cells.append(Paragraph(t, S["cellmono"] if c in mono else S["cell"]))
            rows.append(cells)
        if widths is None:
            est = []
            for c in df.columns:
                vals = df[c].map(_fmt).str.len()
                est.append(float(np.clip(max(len(str(c)) * 0.7, vals.quantile(0.9) if len(vals) else 4), 3, 42)))
            widths = list(np.array(est) / sum(est))
        t = Table(rows, colWidths=[w * FRAME_W for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SOFT)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(ACCENT)),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fa")]),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]))
        self.story.append(t)
        if max_rows and total > max_rows:
            self.p(f"First {max_rows} of {total} rows; the full table is in the results folder.", "small")
        if note:
            self.p(note, "small")
        self.story.append(Spacer(1, 6))


# ---------------------------------------------------------------- figures
def _bar_panels(labels: list[str], stacks: dict[str, np.ndarray], cols: dict[str, str], xlabel: str, title: str, per_panel: int = 52):
    n = len(labels)
    k = max(1, int(np.ceil(n / per_panel)))
    rows = int(np.ceil(n / k))
    fig, axes = plt.subplots(1, k, figsize=(7.4, max(2.0, 0.105 * rows + 0.9)), squeeze=False)
    xmax = max(1.0, float(np.sum(list(stacks.values()), axis=0).max()) if n else 1.0)
    for j in range(k):
        ax = axes[0, j]
        sl = slice(j * rows, min(n, (j + 1) * rows))
        lab = labels[sl]
        y = np.arange(len(lab))
        left = np.zeros(len(lab))
        for name, vals in stacks.items():
            v = np.asarray(vals)[sl]
            ax.barh(y, v, left=left, color=cols[name], height=0.78, label=name.replace("_", " "))
            left += v
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=5.6, style="italic")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.05)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.margins(y=0.005)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.97 if len(stacks) > 1 else 0.985))
    fig.suptitle(title, x=0.01, ha="left", fontsize=8.5, fontweight="bold", y=1.0)
    if len(stacks) > 1:
        fig.legend(h, l_, loc="upper right", ncol=len(stacks), fontsize=6.3, bbox_to_anchor=(1.0, 1.005))
    return fig


def _hist(values, xlabel: str, title: str, color=ACCENT, bins=40):
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.hist(values, bins=bins, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("records")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _alignment_rows(recs, meta, excl, species_sel, spans, L):
    rows = [(n, s) for n, s in recs if n in meta.index and n not in excl]
    if species_sel:
        rows = [r for r in rows if meta.loc[r[0], "species"] in set(species_sel)]
    cols = sorted({c for _, a, b in spans for c in range(a - 1, b)}) or list(range(L))
    best = {}
    for n, s in rows:
        sp = meta.loc[n, "species"]
        cov = sum(s[c] != "-" for c in cols if c < len(s))
        tot = len(s) - s.count("-")
        if sp not in best or (cov, tot) > best[sp][0]:
            best[sp] = ((cov, tot), (n, s))
    out = [v[1] for v in best.values()]
    out.sort(key=lambda r: (str(meta.loc[r[0], "order"]), str(meta.loc[r[0], "family"]), meta.loc[r[0], "species"]))
    return out


def _fig_alignment_overview(rows, meta, spans, L, title):
    lut = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 5}
    M = np.array([[lut.get(c, 4) for c in s.upper().ljust(L, "-")[:L]] for _, s in rows])
    # crop to the primer sites: forward primer start at the left edge, reverse primer end at the right edge
    lo, hi = (min(a for _, a, _ in spans), max(b for _, _, b in spans)) if spans else (1, L)
    M = M[:, lo - 1:hi]
    cmap = matplotlib.colors.ListedColormap([NUC["A"], NUC["C"], NUC["G"], NUC["T"], "#b9c2c5", "#f2f4f5"])
    h = max(2.2, 0.075 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(7.4, min(h, 9.6)))
    # extent keeps alignment column numbers on the x axis (column c is drawn at x = c - 1)
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=5, interpolation="nearest",
              extent=(lo - 1.5, hi - 0.5, len(rows) - 0.5, -0.5))
    ax.set_xlim(lo - 1.5, hi - 0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([meta.loc[n, "species"] for n, _ in rows], fontsize=max(3.2, min(6, 420 / max(1, len(rows)))), style="italic")
    for lab, a, b in spans:
        ax.axvspan(a - 1.5, b - 0.5, color="black", alpha=0.10, lw=0)
        ax.axvline(a - 1.5, color=INK, lw=0.6)
        ax.axvline(b - 0.5, color=INK, lw=0.6)
    if spans:
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xticks([(a + b) / 2 - 1 for _, a, b in spans])
        top.set_xticklabels([lab for lab, _, _ in spans], fontsize=5.8, rotation=35, ha="left", fontweight="bold", color=INK)
        top.tick_params(length=2, pad=1)
        for s_ in top.spines.values():
            s_.set_visible(False)
    ax.set_xlabel("alignment column")
    ax.set_title(title, pad=30 if spans else 6)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    handles = [matplotlib.patches.Patch(color=NUC[b], label=b) for b in "ACGT"] + [matplotlib.patches.Patch(color="#f2f4f5", label="gap")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
    fig.tight_layout()
    return fig


def _site_matrix(rows, a: int, b: int, oligo: str | None):
    """Characters of each row at alignment columns a..b (1-based), dropping columns that are gaps in most rows.
    Returns (reference string, list of row strings). Reference = oligo in alignment orientation, if it fits."""
    L = max(len(s) for _, s in rows)
    pad = 10 if oligo else 0
    cols = [c for c in range(max(0, a - 1 - pad), min(L, b + pad))
            if sum(s[c] != "-" for _, s in rows if c < len(s)) >= 0.5 * len(rows)]
    ref = None
    if oligo:
        cons = "".join(pd.Series([s[c] if c < len(s) else "-" for _, s in rows]).mode().iloc[0].upper() for c in cols)
        best = None
        for o in (oligo, revcomp(oligo)):
            for off in range(0, max(1, len(cols) - len(o) + 1)):
                mm = sum(c not in IUPAC.get(x, "ACGT") for x, c in zip(o, cons[off:off + len(o)]))
                key = (mm, abs(cols[min(off, len(cols) - 1)] - (a - 1)))
                if best is None or key < best[0]:
                    best = (key, o, off)
        if best and len(cols) >= len(oligo):
            ref, off = best[1], best[2]
            cols = cols[off:off + len(oligo)]
        else:
            cols = [c for c in cols if a - 1 <= c < b]
    seqs = ["".join(s[c].upper() if c < len(s) else "-" for c in cols) for _, s in rows]
    return ref, seqs


def _fig_site_letters(rows, labels, blocks, title):
    """blocks: list of (label, a, b, oligo). Letters per species at each oligo site; '.' = same as the oligo."""
    mats = [(lab, *_site_matrix(rows, a, b, olig)) for lab, a, b, olig in blocks]
    widths = [max(4, len(seqs[0]) if seqs else 4) for _, _, seqs in mats]
    n = len(rows)
    fig, axes = plt.subplots(1, len(mats), figsize=(min(7.4, 1.2 + 0.2 * sum(widths)), max(2.0, 0.112 * (n + 2) + 0.8)),
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    fs = max(3.6, min(6.2, 520 / max(1, n)))
    for j, (lab, ref, seqs) in enumerate(mats):
        ax = axes[0, j]
        k = len(seqs[0]) if seqs else 0
        Z = np.zeros((n + 1, k))
        for i, q in enumerate(seqs):
            for x, c in enumerate(q):
                if ref is None:
                    Z[i + 1, x] = 0
                elif c == "-":
                    Z[i + 1, x] = 0.5
                elif c not in IUPAC.get(ref[x], "ACGT") and c in "ACGT":
                    Z[i + 1, x] = 1
        Z[0, :] = -1
        ax.imshow(Z, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#dbe9eb", "#f7f9fa", "#e6e9ea", "#f3b6ae"]),
                  norm=matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.25, 0.75, 1.5], 4), interpolation="nearest")
        if ref:
            for x, c in enumerate(ref):
                ax.text(x, 0, c, ha="center", va="center", fontsize=fs + 0.6, family="DejaVu Sans Mono", fontweight="bold", color=ACCENT)
        for i, q in enumerate(seqs):
            for x, c in enumerate(q):
                same = ref is not None and c in IUPAC.get(ref[x], "ACGT")
                ax.text(x, i + 1, "·" if same else c, ha="center", va="center", fontsize=fs, family="DejaVu Sans Mono",
                        color=MUTED if same else (BAD if ref is not None and c in "ACGT" else INK))
        ax.set_xticks([])
        ax.set_title(lab, fontsize=7.5, loc="center")
        for s in ax.spines.values():
            s.set_visible(False)
        if j == 0:
            ax.set_yticks(range(n + 1))
            ax.set_yticklabels(["oligo"] + labels, fontsize=fs, style="italic")
        else:
            ax.set_yticks([])
    fig.suptitle(title, x=0.01, ha="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_tree(tree, meta):
    from Bio import Phylo
    tips = tree.get_terminals()
    for t in tips:
        if t.name in meta.index:
            t.name = f"{meta.loc[t.name, 'species']}  {t.name}"
    fig, ax = plt.subplots(figsize=(7.4, min(9.8, max(2.5, 0.11 * len(tips) + 0.8))))
    plt.rcParams["font.size"] = max(3.5, min(6.5, 480 / max(1, len(tips))))
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=True,
               branch_labels=lambda c: (f"{int(c.confidence)}" if c.confidence is not None and c.confidence >= 50 else ""))
    plt.rcParams["font.size"] = 7
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("distance")
    fig.tight_layout()
    return fig


def _fig_heatmaps(mmF, mmR, f, r, family):
    order = sorted(mmF.index, key=lambda s: (str(family.get(s, "")), s))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, max(2.2, 0.085 * len(order) + 1.0)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mm", ["#eef3f4", "#f6d5d1", BAD])
    for ax, mm_, primer, lab in ((axes[0], mmF, f, "Forward"), (axes[1], mmR, r, "Reverse")):
        mm_ = mm_.reindex(order)
        im = ax.imshow(mm_.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(primer)))
        ax.set_xticklabels([f"{b}" for b in primer], fontsize=5.5, family="DejaVu Sans Mono")
        ax.xaxis.tick_top()
        ax.axvspan(len(primer) - 5.5, len(primer) - 0.5, fill=False, edgecolor=ACCENT, lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=max(3.2, min(5.6, 420 / max(1, len(order)))), style="italic")
        ax.set_title(f"{lab} primer (5′→3′, box = 3′ end)", pad=14)
    cb = fig.colorbar(im, ax=axes, shrink=0.4, pad=0.02)
    cb.set_label("share of records with a mismatch", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    return fig


def _fig_gap(gap):
    g = gap.copy()
    x, y = 100 * g["max_intra"].fillna(0), 100 * g["min_inter"]
    ok = g["resolved_species"].astype(bool)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.scatter(x[ok], y[ok], s=14, color=ACCENT, label="resolved", edgecolor="white", lw=0.4)
    ax.scatter(x[~ok], y[~ok], s=14, color=BAD, label="not resolved", edgecolor="white", lw=0.4)
    m = max(1.0, float(np.nanmax(np.r_[x.values, y.values])) if len(g) else 1.0)
    ax.plot([0, m], [0, m], ls="--", color=GREY, lw=0.8)
    ax.set_xlabel("largest distance within species (%)")
    ax.set_ylabel("distance to nearest other species (%)")
    ax.set_title("Barcode gap")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def _fig_dist_hist(h):
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    x = h["bin_start"] * 100
    ax.bar(x, h["inter"] / max(1, h["inter"].sum()), width=0.5, color=GOLD, alpha=0.8, label="between species", align="edge")
    ax.bar(x, h["intra"] / max(1, h["intra"].sum()), width=0.5, color=ACCENT, alpha=0.8, label="within species", align="edge")
    ax.set_xlabel("pairwise distance (%)")
    ax.set_ylabel("share of pairs")
    ax.set_title("Distances")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def _side_by_side(rep: R, figs):
    imgs = []
    for f in figs:
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(f)
        buf.seek(0)
        iw, ih = ImageReader(buf).getSize()
        w = FRAME_W / len(figs) - 4
        buf.seek(0)
        imgs.append(Image(buf, width=w, height=w * ih / iw))
    t = Table([imgs], colWidths=[FRAME_W / len(figs)] * len(figs))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    rep.story.append(t)
    rep.story.append(Spacer(1, 6))


def _fig_bars(labels, values, title, ylabel, color=ACCENT, notes=None, ymax=100):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, width=0.62)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.015, (f"{v:.0f}%" if ymax == 100 else f"{v:g}") + (f"\n{notes[i]}" if notes else ""),
                ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center", fontsize=6.8)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _fig_qpcr_margin(sp, target, n=40):
    d = sp[(~sp["is_target"].astype(bool)) & sp["result"].isin(["detected", "at_risk", "blocked"])]
    d = d.sort_values(["min_total_mm", "min_distance_pct"]).head(n)
    fig, ax = plt.subplots(figsize=(7.2, max(2.0, 0.13 * len(d) + 0.9)))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for col, lab, c in (("min_f_mm", "forward", ACCENT), ("min_r_mm", "reverse", GOLD), ("min_p_mm", "probe", "#7a3d8f")):
        if col in d and d[col].notna().any():
            v = d[col].fillna(0).values
            ax.barh(y, v, left=left, color=c, height=0.72, label=lab)
            left += v
    ax.scatter(left + 0.35, y, c=[QSTATUS.get(r, GREY) for r in d["result"]], s=18, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["species"], fontsize=5.8, style="italic")
    ax.invert_yaxis()
    ax.set_xlabel("mismatches in the closest record of each species (dot: red detected, amber at risk, green blocked)")
    ax.set_title(f"Species closest to {target} at the primer and probe sites")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    return fig


def _fig_qpcr_letters(hits, assay, target, n=40, w=5):
    from .plots import qpcr_rows
    rows = qpcr_rows(hits, target, n)
    blocks = [("Forward", assay["forward"], "f_site", True)]
    if assay.get("probe"):
        blocks.append(("Probe", assay["probe"], "p_site", False))
    blocks.append(("Reverse", assay["reverse"], "r_site", True))
    widths = [len(o) for _, o, _, _ in blocks]
    k = len(rows)
    fs = max(3.6, min(6.0, 480 / max(1, k)))
    fig, axes = plt.subplots(1, len(blocks), figsize=(7.4, max(2.0, 0.12 * (k + 1) + 0.9)), gridspec_kw={"width_ratios": widths})
    for j, (lab, olig, col, primer) in enumerate(blocks):
        ax = axes[j]
        Z = np.zeros((k + 1, len(olig)))
        Z[0] = -1
        for i, r in enumerate(rows.itertuples()):
            site = str(getattr(r, col) or "")
            for x, b in enumerate(olig):
                t = site[x] if x < len(site) else "?"
                if t not in IUPAC.get(b, "ACGT") and t not in "N?":
                    Z[i + 1, x] = 2 if primer and x >= len(olig) - w else 1
                ax.text(x, i + 1, "·" if Z[i + 1, x] == 0 else t, ha="center", va="center", fontsize=fs, family="DejaVu Sans Mono",
                        color=MUTED if Z[i + 1, x] == 0 else "white" if Z[i + 1, x] == 2 else INK)
        for x, b in enumerate(olig):
            ax.text(x, 0, b, ha="center", va="center", fontsize=fs + 0.5, family="DejaVu Sans Mono", fontweight="bold", color=ACCENT)
        ax.imshow(Z, aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(["#dbe9eb", "#f7f9fa", "#f3b6ae", BAD]),
                  norm=matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], 4))
        if primer:
            ax.axvspan(len(olig) - w - 0.5, len(olig) - 0.5, fill=False, edgecolor=ACCENT, lw=0.8)
        ax.set_xticks([])
        ax.set_title(f"{lab} 5′→3′", fontsize=7, loc="center")
        for s in ax.spines.values():
            s.set_visible(False)
        if j == 0:
            ax.set_yticks(range(k + 1))
            ax.set_yticklabels(["oligo"] + [f"{'▶ ' if r.species == target else ''}{r.species} ({r.outcome.replace('_', ' ')})"
                                            for r in rows.itertuples()], fontsize=fs)
        else:
            ax.set_yticks([])
    fig.suptitle("Primer and probe sites in the closest record of each species (· same base; red = mismatch; dark = in the 3′ window)",
                 x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_qpcr_distance(sp, target, intra, n=40):
    d = sp[(~sp["is_target"].astype(bool)) & sp["min_distance_pct"].notna()].sort_values("min_distance_pct").head(n)
    fig, ax = plt.subplots(figsize=(7.2, max(2.0, 0.13 * len(d) + 0.9)))
    y = np.arange(len(d))
    ax.barh(y, d["min_distance_pct"], color=[QSTATUS.get(r, GREY) for r in d["result"]], height=0.72)
    if intra is not None:
        ax.axvline(intra, color=ACCENT, ls="--", lw=0.8)
        ax.text(intra, -0.9, " largest within target", color=ACCENT, fontsize=6, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(d["species"], fontsize=5.8, style="italic")
    ax.invert_yaxis()
    ax.set_xlabel(f"distance of the amplicon (between the primers) to the nearest {target} record (%); colour = qPCR outcome")
    ax.set_title("Amplicon distance to the target")
    fig.tight_layout()
    return fig


def _fig_mito(feats, L, organism, subtitle, gcw, sites):
    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0.06, 0.1, 0.88, 0.86], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.22)
    ax.axis("off")
    th = lambda p: 2 * np.pi * (p - 1) / L
    RP, RM, TH = 0.80, 0.665, 0.115
    t = np.linspace(0, 2 * np.pi, 500)
    ax.plot(t, np.full_like(t, RP - 0.0075), color="#c9d5d9", lw=0.6)
    ax.plot(t, np.full_like(t, RM + TH + 0.0075), color="#c9d5d9", lw=0.6)
    for r in feats.itertuples():
        a, b = th(r.start), th(r.end + 1)
        base = RP if r.strand == "+" else RM
        ax.bar((a + b) / 2, TH, width=b - a, bottom=base, color=MITO_COLORS.get(r.category, "#5a6a70"), edgecolor="white", lw=0.25)
        if r.part != 0:
            continue
        mid = (a + b) / 2
        if r.category == "tRNA":
            ax.text(mid, base + TH / 2, str(r.label), ha="center", va="center", fontsize=4.2, color="white", fontweight="bold")
        else:
            rr = base + TH + 0.045 if r.strand == "+" else base - 0.05
            ax.text(mid, rr, str(r.label), ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK)
    if gcw is not None and len(gcw):
        mgc = gcw["gc"].mean()
        mids = np.array([th((s + e) / 2) for s, e in zip(gcw["start"], gcw["end"])])
        wid = 2 * np.pi * (gcw["end"] - gcw["start"] + 1).values / L
        v = (gcw["gc"].values - mgc) * 0.9
        ax.bar(mids, np.abs(v), width=wid, bottom=np.where(v >= 0, 0.5, 0.5 + v), color=np.where(v >= 0, ACCENT, GOLD), lw=0)
        ax.plot(t, np.full_like(t, 0.5), color="#c9d5d9", lw=0.5)
    for k in range(0, L, 1000):
        ax.plot([th(k + 1)] * 2, [0.6, 0.625], color=MUTED, lw=0.6)
        if k % 2000 == 0:
            ax.text(th(k + 1), 0.575, f"{k // 1000} kb", ha="center", va="center", fontsize=5.2, color=MUTED)
    handles = []
    if sites is not None and len(sites):
        f = sites[sites["found"]] if "found" in sites else sites.iloc[0:0]
        for i, r in enumerate(f.itertuples()):
            c = PAIR_PALETTE[i % len(PAIR_PALETTE)]
            rr = 0.955 + 0.04 * (i % 6)
            tt = np.linspace(th(r.start), th(r.end + 1), 60)
            ok = r.status == "amplified"
            ax.plot(tt, np.full_like(tt, rr), color=c, lw=3 if ok else 2, ls="-" if ok else (0, (2, 1.2)), solid_capstyle="butt")
            handles.append(matplotlib.lines.Line2D([], [], color=c, lw=3, ls="-" if ok else "--",
                                                   label=f"{r.pair}: {int(r.product_len)} bp {'✓' if ok else '✗ ' + str(r.status).replace('_', ' ')}"))
    ax.text(0, 0.07, organism, ha="center", va="center", fontsize=9, color=INK, style="italic", fontweight="bold")
    ax.text(np.pi, 0.07, subtitle, ha="center", va="center", fontsize=6.5, color=MUTED)
    cats = [c for c in MITO_COLORS if c in set(feats["category"])]
    leg1 = [matplotlib.patches.Patch(color=MITO_COLORS[c], label=c) for c in cats]
    fig.legend(handles=leg1, loc="lower left", ncol=2, fontsize=6, bbox_to_anchor=(0.02, 0.0))
    if handles:
        fig.legend(handles=handles, loc="lower right", fontsize=6, bbox_to_anchor=(0.99, 0.0), title="PCR products", title_fontsize=6.5)
    return fig


# ---------------------------------------------------------------- sections
def _sec_names(rep: R, ds: str, n: int):
    p = paths.taxonomy_dir(ds) / "species_resolved.csv"
    rep.h1(n, "Name check", "Names checked against WoRMS (accepted names, synonyms) and NCBI Taxonomy (taxid, lineage).")
    if not p.exists():
        rep.p("No name check yet.")
        return
    tax = pd.read_csv(p, dtype={"ncbi_taxid": str})
    vc = tax["status"].value_counts()
    rep.p(f"{len(tax)} names: <b>{vc.get('ok', 0)}</b> accepted as given, <b>{vc.get('synonym', 0)}</b> synonyms replaced by the "
          f"accepted name, <b>{vc.get('check', 0)}</b> to check, <b>{vc.get('not_found', 0)}</b> not found.")
    syn = tax[tax["input_name"] != tax["accepted_name"]]
    if len(syn):
        rep.p("Names changed: " + "; ".join(f"<i>{_esc(a)}</i> → <i>{_esc(b)}</i>" for a, b in zip(syn["input_name"], syn["accepted_name"])))
    rep.table(tax[["input_name", "accepted_name", "family", "order", "ncbi_taxid", "status", "note"]], status_col="status",
              widths=[0.2, 0.2, 0.14, 0.14, 0.08, 0.08, 0.16])


def _sec_sequences(rep: R, ds: str, marker: str, n: int):
    q = paths.qc_dir(ds, marker)
    rep.h1(n, "Sequences & QC", f"{marker} sequences cut out of GenBank gene records and mitogenomes, after quality control.")
    if not (q / "qc_summary.json").exists():
        rep.p("No QC results yet.")
        return
    s = json.loads((q / "qc_summary.json").read_text())
    rm = ", ".join(f"{v} {k}" for k, v in s["removed_by_reason"].items()) or "none"
    rep.p(f"<b>{s['kept']:,}</b> of {s['raw']:,} records kept ({s['length_limits'][0]}–{s['length_limits'][1]} bp); removed: {rm}. "
          f"<b>{s['species_with_data']}</b> of {s['species_total']} species have sequences; {s['exact_duplicates']} records are exact "
          f"duplicates of another record (kept).")
    if s["species_without_data"]:
        rep.p("No sequences: " + _ital(s["species_without_data"]))
    counts = pd.read_csv(q / "species_counts.csv")
    c = counts[counts["n_clean"] > 0].sort_values("n_clean", ascending=False)
    rep.img(_bar_panels(c["species"].tolist(), {"gene records": c["n_gene_record"].values, "from mitogenomes": c["n_mitogenome"].values},
                        {"gene records": ACCENT, "from mitogenomes": GOLD}, "records after QC", "Records per species"), max_h=0.8)
    meta = pd.read_csv(q / "clean_metadata.csv")
    rep.img(_hist(meta["length"], "length (bp)", "Length of the records"), width=0.9)
    flags = paths.alignment_dir(ds, marker) / "misid_flags.csv"
    if flags.exists():
        f = pd.read_csv(flags)
        rep.h2(f"Possibly misidentified records ({len(f)})")
        rep.table(f, max_rows=40)
    from . import exclude
    ex = exclude.load(ds, marker)
    rep.h2(f"Records excluded from step 3 onward ({len(ex)})")
    rep.table(ex, max_rows=60)


def _spans(ds: str, marker: str, runs: list[str]) -> list[tuple[str, int, int, str]]:
    """(label, first col, last col, oligo) for PCR runs and qPCR checks."""
    out = []
    for r in runs:
        if r.startswith("qPCR "):
            p = paths.qpcr_dir(ds, marker, r[5:]) / "summary.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            a = s["assay"]
            olig = {"F": a["forward"], "R": a["reverse"], "P": a.get("probe", "")}
            for part in s["site_columns"].split(";"):
                d, rng = part.strip().split(" ")
                x, y = (int(v) for v in rng.split("-"))
                out.append((f"{r[5:]} {d}", x, y, olig.get(d, "")))
            continue
        d = paths.pcr_dir(ds, marker, r)
        if not (d / "hits.csv").exists():
            continue
        h = pd.read_csv(d / "hits.csv", usecols=lambda c: c == "site_columns")
        pp = json.loads((d / "summary.json").read_text())["primer_pair"]
        if "site_columns" in h and h["site_columns"].notna().any():
            for part in str(h["site_columns"].dropna().iloc[0]).split(";"):
                dd, rng = part.strip().split(" ")
                x, y = (int(v) for v in rng.split("-"))
                out.append((f"{r} {dd}", x, y, pp["forward"] if dd == "F" else pp["reverse"]))
    return out


def _sec_alignment(rep: R, ds: str, marker: str, opts: dict, n: int):
    a = paths.alignment_dir(ds, marker)
    rep.h1(n, "Alignment", "One record per species (the one covering the primer sites best), with the primer sites marked.")
    if not (a / "aligned.fasta").exists():
        rep.p("No alignment yet.")
        return
    recs = read_fasta(a / "aligned.fasta")
    meta = pd.read_csv(a / "aligned_metadata.csv").set_index("seq_id")
    cols = pd.read_csv(a / "columns.csv")
    L = len(cols)
    excl = qc.excluded_ids(ds, marker)
    spans4 = _spans(ds, marker, opts.get("align_runs") or [])
    spans = [(lab, x, y) for lab, x, y, _ in spans4]
    sel = opts.get("align_species") or []
    rows = _alignment_rows(recs, meta, excl, sel, spans, L)
    rep.p(f"{len(recs):,} records, {L:,} columns ({int((cols['entropy'] > 0).sum()):,} variable), {len(excl)} excluded. "
          f"Shown: {len(rows)} species{' selected in step 3' if sel else ''}. Primer sites from: "
          f"{_esc(', '.join(opts.get('align_runs') or []) or 'none')}.")
    title = f"{marker} alignment, one record per species"
    if spans:
        title += f", columns {min(x for _, x, _ in spans)}–{max(y for _, _, y in spans)} (forward primer to reverse primer)"
    rep.img(_fig_alignment_overview(rows, meta, spans, L, title), max_h=0.9)
    if spans4:
        # one figure per run: its F (probe) R sites side by side
        groups: dict[str, list] = {}
        for lab, x, y, olig in spans4:
            groups.setdefault(lab.rsplit(" ", 1)[0], []).append((lab.rsplit(" ", 1)[1], x, y, olig))
        labels = [meta.loc[nm, "species"] for nm, _ in rows]
        for g, blocks in groups.items():
            order = {"F": 0, "P": 1, "R": 2}
            blocks = sorted(blocks, key=lambda b: order.get(b[0], 3))
            named = [({"F": "Forward", "R": "Reverse (as on the alignment)", "P": "Probe"}.get(b, b) + f"  cols {x}–{y}", x, y, o)
                     for b, x, y, o in blocks]
            rep.h2(f"Primer sites: {_esc(g)}")
            rep.img(_fig_site_letters(rows, labels, named, f"{g}: bases of each species at the oligo sites (· = same as the oligo, red = mismatch)"),
                    max_h=0.92)
    t = opts.get("tree")
    if t:
        from .phylo import load_tree
        try:
            tree, info = load_tree(ds, marker, t)
            rep.h2(f"Tree: {_esc(t)}")
            rep.p(f"Neighbour joining on {info.get('model', 'k2p').upper()} distances, region {info.get('region')}, "
                  f"{info.get('n_tips')} tips, bootstrap {info.get('bootstrap', 0)} (support ≥ 50 % on branches)"
                  + (f"; species: {_ital(info['species_selected'])}" if info.get("species_selected") else "") + ".", "small")
            rep.img(_fig_tree(tree, meta), max_h=0.92)
        except Exception as e:  # noqa: BLE001
            rep.p(f"Tree not drawn: {_esc(e)}", "small")


def _closest_rows(hits: pd.DataFrame) -> pd.DataFrame:
    t = hits[hits["status"].isin(["amplified", "fail_mismatch", "fail_length"])].copy()
    t["_mm"] = t["f_mm"].fillna(9) + t["r_mm"].fillna(9)
    return t.sort_values("_mm").drop_duplicates("species").set_index("species")


def _sec_pcr(rep: R, ds: str, marker: str, opts: dict, n: int):
    from .report import load_run
    rep.h1(n, "In silico PCR", "Binding sites with mismatches on both strands, the 3′-end rule and the product length, for every record.")
    runs = opts.get("pcr_runs") or []
    if not runs:
        rep.p("No PCR run selected.")
        return
    for k, name in enumerate(runs):
        run = load_run(ds, marker, name)
        s, sp, hits = run["summary"], run["species"], run["hits"]
        if k:
            rep.story.append(PageBreak())
        rep.h2(_esc(name))
        pp = s["primer_pair"]
        rules = s.get("rules", {})
        rep.p(f"F&nbsp;{pp['forward']}<br/>R&nbsp;{pp['reverse']}", "mono")
        rep.kv([("species amplified", f"<b>{s['species_amplified']}</b> of {s['species_testable']} testable species"),
                ("sequences amplified", f"{s['seqs_amplified']:,} of {s['seqs_testable']:,}"),
                ("median insert / product", f"{s['median_insert_len'] or 0:.0f} / {s['median_product_len'] or 0:.0f} bp"),
                ("Tm F / R", f"{s['forward']['tm_min']}–{s['forward']['tm_max']} / {s['reverse']['tm_min']}–{s['reverse']['tm_max']} °C"),
                ("rules", f"≤ {rules.get('max_mismatches')} mismatches per primer, ≤ {rules.get('max_3prime_mismatches')} in the last "
                          f"{rules.get('three_prime_window')} bases (3′ end), product {pp['product_min']}–{pp['product_max']} bp"),
                ("fail", _ital(s["species_fail"]) or "none"), ("not testable", _ital(s["species_not_testable"]) or "none"),
                ("no sequences", _ital(s["species_no_sequences"]) or "none"),
                ("reference", _esc(pp.get("reference", "")))])
        d = sp[sp["n_seqs"] > 0].sort_values(["family", "species"])
        cols = [c for c in ["amplified", "fail_mismatch", "fail_length", "no_site", "site_not_covered"] if c in d]
        rep.img(_bar_panels(d["species"].tolist(), {c: d[c].fillna(0).values for c in cols}, {c: STATUS[c] for c in cols},
                            "records", "Result per species (records)"), max_h=0.85)
        fam = sp.set_index("species")["family"]
        if len(run["mmF"]):
            rep.img(_fig_heatmaps(run["mmF"], run["mmR"], pp["forward"], pp["reverse"], fam), max_h=0.92)
        cl = _closest_rows(hits)
        t = sp.copy()
        t["F mismatches (closest record)"] = t["species"].map(cl["f_mismatches"]).fillna("")
        t["R mismatches (closest record)"] = t["species"].map(cl["r_mismatches"]).fillna("")
        t = t[["species", "family", "result", "amplified", "n_testable", "min_f_mm", "min_r_mm", "F mismatches (closest record)",
               "R mismatches (closest record)", "median_insert_len"]].rename(columns={"n_testable": "testable", "median_insert_len": "insert bp"})
        rep.h2("Per species: amplified or not, and the primer mismatches")
        rep.table(t, status_col="result", widths=[0.18, 0.11, 0.08, 0.06, 0.06, 0.05, 0.05, 0.18, 0.18, 0.05],
                  note="Mismatches: position from the 5′ end, primer base > template base, in the record with the fewest mismatches.")


def _sec_qpcr(rep: R, ds: str, marker: str, opts: dict, n: int):
    from . import qpcr
    rep.h1(n, "In silico qPCR", "Is the assay specific to one species? Every record is read at the primer and probe sites.")
    name = opts.get("qpcr_run")
    if not name:
        rep.p("No qPCR check selected.")
        return
    run = qpcr.load_run(ds, marker, name)
    s, sp, hits, chk = run["summary"], run["species"], run["hits"], run["checks"]
    a, R_ = s["assay"], s["rules"]
    col = GOOD if s["verdict"] == "specific" else WARN if s["verdict"].startswith("specific") else BAD
    rep.h2(f"{_esc(name)} · target <i>{_esc(s['target'])}</i>")
    rep.p(f"<font color='{col}'><b>{_esc(s['verdict'].capitalize())}</b></font> (in silico, for the species in this dataset)")
    rep.p(f"F&nbsp;{a['forward']}<br/>R&nbsp;{a['reverse']}" + (f"<br/>P&nbsp;{a['probe']} ({a.get('probe_type')})" if a.get("probe") else ""), "mono")
    rep.kv([("target records detected", f"<b>{s['target_detected']}</b> of {s['target_covered']} covering the sites "
                                        f"({s['target_perfect']} match every oligo perfectly)"),
            ("other species detected", _ital(s["nontarget_detected"]) or "none"),
            ("other species at risk", _ital(s["nontarget_at_risk"]) or "none"),
            ("other species blocked", f"{s['nontarget_blocked']} of {s['nontarget_species_tested']} testable"),
            ("not testable", _ital(s["nontarget_not_testable"]) or "none"),
            ("closest species", f"<i>{_esc(s['closest_species'])}</i>: {s['closest_total_mm']} mismatches (F + R + probe)"),
            ("amplicon", f"{s['amplicon_len'] or 0:.0f} bp; alignment columns {s['site_columns']}"),
            ("rules", f"detected: ≤ {R_['detect_primer_mm']} mismatches per primer, ≤ {R_['detect_primer_3prime']} in the last "
                      f"{R_['three_prime_window']} bases, ≤ {R_['detect_probe_mm']} in the probe · blocked: ≥ {R_['block_primer_3prime']} at "
                      f"one 3′ end, ≥ {R_['block_primer_mm']} in one primer, ≥ {R_['block_pair_mm']} in both, ≥ {R_['block_probe_mm']} in the probe")])
    rep.img(_fig_qpcr_letters(hits, a, s["target"]), max_h=0.9)
    rep.img(_fig_qpcr_margin(sp, s["target"]), max_h=0.75)
    rep.h2(f"Oligo checks: {s['checks_passed']} of {s['checks_total']} pass")
    c = chk.copy()
    c["ok"] = c["ok"].map({True: "pass", False: "check"})
    rep.table(c, status_col="ok", status_colors={"pass": GOOD, "check": BAD}, widths=[0.1, 0.3, 0.2, 0.3, 0.1])
    nclose = 40
    rep.h2(f"Per species: the target and the {nclose} closest other species")
    sp = sp.copy()
    tg = sp["is_target"].astype(bool)
    sp.loc[tg, "result"] = "target: " + sp.loc[tg, "result"].astype(str)
    sp = pd.concat([sp[tg], sp[~tg].head(nclose)])
    t = sp[["species", "family", "is_target", "result", "n_covered", "detected", "at_risk", "blocked", "min_f_mm", "min_r_mm", "min_p_mm",
            "min_total_mm", "min_distance_pct", "f_mismatches", "r_mismatches", "p_mismatches"]].rename(columns={"is_target": "target"})
    rep.table(t, status_col="result", status_colors=QSTATUS,
              widths=[0.14, 0.09, 0.04, 0.07, 0.045, 0.045, 0.04, 0.045, 0.035, 0.035, 0.035, 0.04, 0.045, 0.095, 0.095, 0.09],
              note=f"All species are in results/qpcr/{_esc(paths.safe(ds))}/{_esc(marker)}/{_esc(name)}/species_summary.csv.")
    rep.p("References for the rules: " + "; ".join(_esc(x) for x in qpcr.REFERENCES) + ". Species that are not in this list "
          "(or not in GenBank) are not checked: also run Primer-BLAST and test tissue DNA of close relatives.", "small")


def _sec_resolution(rep: R, ds: str, marker: str, opts: dict, n: int, mode: str):
    if mode == "qpcr":
        from . import qpcr
        rep.h1(n, "Resolution", "How different is the target's amplicon (between the primers) from every other species?")
        name = opts.get("qpcr_run")
        if not name:
            rep.p("No qPCR check selected.")
            return
        run = qpcr.load_run(ds, marker, name)
        s, sp = run["summary"], run["species"]
        nt = sp[(~sp["is_target"].astype(bool)) & sp["min_distance_pct"].notna()]
        intra = s.get("target_max_intra_pct") or 0
        same = nt[nt["min_distance_pct"] <= 0]
        rep.p(f"Largest distance within <i>{_esc(s['target'])}</i>: <b>{intra:.2f} %</b>. Nearest other species: "
              + (f"<i>{_esc(nt.nsmallest(1, 'min_distance_pct')['species'].iloc[0])}</i> at {nt['min_distance_pct'].min():.2f} %. " if len(nt) else "— . ")
              + f"{int((nt['min_distance_pct'] > intra).sum())} of {len(nt)} species are further away than the target's own variation."
              + (f" Identical amplicon to some target records: {_ital(same['species'])} (only primer/probe mismatches keep the assay specific)." if len(same) else ""))
        rep.img(_fig_qpcr_distance(sp, s["target"], intra), max_h=0.8)
        rep.table(nt.sort_values("min_distance_pct")[["species", "family", "result", "min_distance_pct", "min_total_mm", "closest_record"]],
                  status_col="result", status_colors=QSTATUS, max_rows=60)
        return
    from .report import load_run
    rep.h1(n, "Resolution", "Can the amplicon tell species apart? Largest distance within a species against the distance to the nearest other species.")
    runs = opts.get("resolution_runs") or []
    if not runs:
        rep.p("No resolution statistics yet (step 6).")
        return
    for name in runs:
        run = load_run(ds, marker, name)
        if "gap" not in run:
            rep.p(f"{_esc(name)}: no resolution statistics yet.")
            continue
        rs, gap = run["res_summary"], run["gap"]
        rep.h2(_esc(name))
        rep.kv([("species with a barcode gap", f"<b>{rs['resolved_species']}</b> of {rs['n_species']} amplified species"),
                ("resolved to genus / family", f"{rs['resolved_genus']} / {rs['resolved_family']}"),
                ("unique amplicons", f"{rs['unique_amplicon']} of {rs['n_species']}"),
                ("median gap", f"{100 * (rs['median_gap'] or 0):.2f} % ({rs['model'].upper()}, {rs['n_amplicons']} amplicons, primers removed)"),
                ("identical amplicons", " | ".join(_ital(g) for g in rs["identical_groups"]) or "none")])
        _side_by_side(rep, [_fig_gap(gap), _fig_dist_hist(run["hist"])])
        t = gap.sort_values("gap")[["species", "family", "n_seqs", "max_intra", "min_inter", "nearest_species", "gap", "resolved_species",
                                    "resolved_genus"]].copy()
        for c_ in ("max_intra", "min_inter", "gap"):
            t[c_] = (100 * t[c_]).round(2)
        rep.table(t.rename(columns={"max_intra": "max intra %", "min_inter": "min inter %", "gap": "gap %"}),
                  widths=[0.2, 0.12, 0.05, 0.07, 0.07, 0.22, 0.07, 0.1, 0.1])


def _sec_design(rep: R, ds: str, marker: str, opts: dict, n: int, mode: str):
    top = int(opts.get("design_top") or 20)
    if mode == "qpcr":
        rep.h1(n, "Assay design", f"Top {top} species-specific assays of the selected design run.")
        run = opts.get("qpcr_design_run")
        if not run:
            rep.p("No assay design run yet (step 7).")
            return
        d = paths.qpcr_design_dir(ds, marker, run)
        D = pd.read_csv(d / "assays.csv")
        prm = json.loads((d / "params.json").read_text())
        rep.p(f"Run {_esc(run)}: target <i>{_esc(prm['target'])}</i>, {prm['n_target_records']} target records of {prm['n_records']}; "
              f"product {prm['design'].get('product_min')}–{prm['design'].get('product_max')} bp, probe {prm['design'].get('probe_type')}, "
              f"oligos allowed {prm['design'].get('target_max_mismatches', 0)} mismatches to target records.")
        cols = ["assay_id", "forward", "reverse", "probe", "product_len", "tm_f", "tm_r", "tm_p", "verdict", "target_detected",
                "species_detected", "species_at_risk", "closest_species", "closest_total_mm", "checks"]
        rep.table(D[[c for c in cols if c in D]].head(top), mono=("forward", "reverse", "probe"),
                  widths=[0.04, 0.12, 0.12, 0.14, 0.045, 0.035, 0.035, 0.035, 0.1, 0.05, 0.045, 0.045, 0.11, 0.04, 0.045])
        return
    rep.h1(n, "Primer design", f"Top {top} candidate pairs of the selected design run.")
    run = opts.get("design_run")
    if not run or not (paths.design_dir(ds, marker, run) / "pairs.csv").exists():
        rep.p("No primer design run yet (step 7).")
        return
    d = paths.design_dir(ds, marker, run)
    P = pd.read_csv(d / "pairs.csv").head(top)
    prm = json.loads((d / "params.json").read_text()) if (d / "params.json").exists() else {}
    dp = prm.get("design", {})
    rep.p(f"Run {_esc(run)}: {prm.get('n_sequences', '?')} sequences, {prm.get('n_species', '?')} species; product "
          f"{dp.get('product_min')}–{dp.get('product_max')} bp, primers {dp.get('primer_min_len')}–{dp.get('primer_max_len')} nt, "
          f"Tm {dp.get('tm_min')}–{dp.get('tm_max')} °C, degeneracy ≤ {dp.get('max_degeneracy')}.")
    P["degeneracy"] = P["deg_f"] * P["deg_r"]
    cols = ["pair_id", "forward", "reverse", "f_cols", "r_cols", "product_len_est", "species_coverage", "seq_coverage",
            "resolution_unique", "tm_f", "tm_r", "degeneracy", "heterodimer_dg", "score"]
    rep.table(P[cols], mono=("forward", "reverse"), widths=[0.04, 0.15, 0.15, 0.065, 0.065, 0.055, 0.06, 0.06, 0.06, 0.045, 0.045, 0.05, 0.055, 0.05],
              note="species_coverage = species amplified / species with records covering both sites; resolution_unique = share of "
                   "amplified species whose amplicon is not shared with another species.")
    if (d / "evaluated_pairs.csv").exists():
        rep.h2("Top pairs checked with the full in silico PCR and barcode gap")
        rep.table(pd.read_csv(d / "evaluated_pairs.csv"), mono=("forward", "reverse"))


def _sec_compare(rep: R, ds: str, marker: str, opts: dict, n: int, mode: str):
    if mode == "qpcr":
        from . import qpcr
        rep.h1(n, "Compare qPCR assays", "Target records detected against other species detected or at risk.")
        runs = opts.get("compare_qpcr_runs") or []
        if not runs:
            rep.p("No assays selected.")
            return
        comp = qpcr.compare_runs(ds, marker, runs)
        rep.img(_fig_bars(comp["assay"].tolist(), comp["target_detected_pct"].tolist(), "Target records detected", "% of target records",
                          GOOD, comp["target_records"].tolist()), width=0.95)
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        x = np.arange(len(comp))
        ax.bar(x, comp["nontarget_detected"], color=BAD, width=0.6, label="detected")
        ax.bar(x, comp["nontarget_at_risk"], bottom=comp["nontarget_detected"], color=WARN, width=0.6, label="at risk")
        ax.set_xticks(x)
        ax.set_xticklabels(comp["assay"], fontsize=6.8)
        ax.set_ylabel("other species")
        ax.set_title("Other species detected or at risk (fewer is better)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        rep.img(fig, width=0.95)
        rep.table(comp)
        return
    from .report import compare_runs
    rep.h1(n, "Compare primer pairs", "First the species each pair amplifies, then the species each pair resolves (barcode gap).")
    runs = opts.get("compare_runs") or []
    if not runs:
        rep.p("No primer pairs selected.")
        return
    comp = compare_runs(ds, marker, runs)
    c1 = comp.sort_values("species_amplified_pct", ascending=False)
    rep.h2("Species amplified")
    rep.img(_fig_bars(c1["primer"].tolist(), c1["species_amplified_pct"].tolist(), "Species amplified (% of testable species)",
                      "% of testable species", ACCENT, [f"{a}/{t}" for a, t in zip(c1["species_amplified"], c1["species_testable"])]), width=0.95)
    rep.h2("Species resolved")
    if "species_resolved_pct" in comp and comp["species_resolved_pct"].notna().any():
        c2 = comp[comp["species_resolved_pct"].notna()].sort_values("species_resolved_pct", ascending=False)
        rep.img(_fig_bars(c2["primer"].tolist(), c2["species_resolved_pct"].tolist(), "Species resolved by a barcode gap (% of testable species)",
                          "% of testable species", GOLD, [f"{int(a)}/{t}" for a, t in zip(c2["species_resolved"], c2["species_testable"])]), width=0.95)
        miss = [r for r in runs if r not in set(c2["primer"])]
        if miss:
            rep.p("No resolution statistics for: " + _esc(", ".join(miss)) + " (compute them in step 6).", "small")
    else:
        rep.p("No resolution statistics for these pairs yet (step 6).", "small")
    t = comp.drop(columns=[c for c in ("identical_groups",) if c in comp])
    rep.table(t, mono=("forward", "reverse"))
    if "identical_groups" in comp:
        rep.h2("Species with identical amplicons")
        for r in comp.itertuples():
            if isinstance(getattr(r, "identical_groups", None), str) and r.identical_groups:
                rep.p(f"<b>{_esc(r.primer)}</b>: " + " | ".join(_ital(g.split(", ")) for g in r.identical_groups.split(" | ")), "small")


def _sec_mito(rep: R, ds: str, opts: dict, n: int, mode: str):
    from . import config, mitomap
    rep.h1(n, "Mitogenome map", "Gene layout of the selected mitochondrial genome and where the selected primer pairs amplify.")
    sp = opts.get("mito_species")
    if not sp:
        rep.p("No mitogenome downloaded.")
        return
    recs = sorted(mitomap.records(sp), key=lambda r: (not r.id.startswith("NC_"), r.id))
    rec = next((r for r in recs if r.id == opts.get("mito_record")), recs[0] if recs else None)
    if rec is None:
        rep.p(f"No mitogenome for <i>{_esc(sp)}</i>.")
        return
    seq = str(rec.seq)
    feats = mitomap.features(rec)
    if mode == "qpcr":
        from . import qpcr
        lib = {k: {**v, "product_min": 0, "product_max": 3000} for k, v in qpcr.library().items()}
        pick = [p for p in (opts.get("mito_assays") or []) if p in lib] or ([opts["qpcr_run"]] if opts.get("qpcr_run") in lib else list(lib)[:4])
    else:
        lib = config.primers()
        pick = [p for p in (opts.get("mito_pairs") or []) if p in lib] or list(lib)[:4]
    sites = mitomap.primer_sites(seq, {k: lib[k] for k in pick}) if pick else None
    L = len(seq)
    gcp = 100 * sum(b in "GC" for b in seq.upper()) / L
    rep.p(f"<i>{_esc(sp)}</i>, {rec.id}: {L:,} bp, GC {gcp:.1f} %. Outer ring: genes on the + strand; inner ring: − strand; "
          f"centre: GC content above (teal) or below (gold) the genome average in 200 bp windows; arcs: PCR products "
          f"(solid = amplifies under the PCR rules, dashed = fails).")
    rep.img(_fig_mito(feats, L, rec.annotations.get("organism", sp), f"{rec.id} · {L:,} bp · GC {gcp:.1f}%",
                      mitomap.gc_windows(seq), sites), width=0.95, max_h=0.78)
    if sites is not None and len(sites):
        rep.h2("Primer pairs on this genome")
        s = sites.copy()
        if "start" in s:
            s["genes covered"] = [", ".join(feats[(feats["end"] >= r.start) & (feats["start"] <= r.end)]["label"].unique()) if r.found else ""
                                  for r in s.itertuples()]
        rep.table(s[[c for c in ["pair", "status", "start", "end", "strand", "product_len", "f_mm", "r_mm", "f_mismatches",
                                 "r_mismatches", "genes covered"] if c in s]])
    rep.h2("Genes")
    rep.table(feats[feats["part"] == 0][["label", "name", "category", "start", "end", "strand", "length", "product"]],
              widths=[0.06, 0.1, 0.2, 0.07, 0.07, 0.05, 0.06, 0.39])


# ---------------------------------------------------------------- build
def build_pdf(dataset: str, marker: str, opts: dict | None = None, progress: Callable[[str, float], None] | None = None) -> Path:
    log = get_logger("report")
    o = {**default_options(dataset, marker), **(opts or {})}
    mode = o["mode"]
    titles = QSECTION_TITLES if mode == "qpcr" else SECTION_TITLES
    secs = [s for s in SECTIONS if s in o["sections"]]
    rep = R()
    prog = progress or (lambda m, f: None)
    # ---- cover
    rep.story.append(Spacer(1, 30 * mm))
    rep.p("ISPCR · IN SILICO PCR STUDIO", "eyebrow")
    if mode == "qpcr":
        tgt = None
        if o.get("qpcr_run"):
            tgt = json.loads((paths.qpcr_dir(dataset, marker, o["qpcr_run"]) / "summary.json").read_text())["target"]
        rep.p("In silico qPCR report", "title")
        rep.p(f"Species-specific detection of <i>{_esc(tgt or '—')}</i> · {marker} · dataset {_esc(dataset)}", "subtitle")
    else:
        rep.p("In silico PCR report", "title")
        rep.p(f"Metabarcoding primers · {marker} · dataset {_esc(dataset)}", "subtitle")
    rep.p(f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the choices made in the app (results/reports/{_esc(paths.safe(dataset))}/selections.json).", "small")
    rep.story.append(Spacer(1, 6 * mm))
    choice_rows = [("mode", "in silico qPCR (species-specific)" if mode == "qpcr" else "in silico PCR (metabarcoding)"),
                   ("marker", marker),
                   ("alignment: species", _ital(o["align_species"]) if o["align_species"] else "all (one record per species)"),
                   ("alignment: primer sites", _esc(", ".join(o["align_runs"]) or "none")),
                   ("tree", _esc(o.get("tree") or "none"))]
    if mode == "qpcr":
        choice_rows += [("qPCR assay check", _esc(o.get("qpcr_run") or "none")),
                        ("assay design run", _esc(o.get("qpcr_design_run") or "none") + f" (top {o.get('design_top')})"),
                        ("assays compared", _esc(", ".join(o.get("compare_qpcr_runs") or []) or "none")),
                        ("mitogenome", f"<i>{_esc(o.get('mito_species') or '—')}</i> {_esc(o.get('mito_record') or '')}; assays: "
                                       f"{_esc(', '.join(o.get('mito_assays') or []) or 'as in the library')}")]
    else:
        choice_rows += [("in silico PCR", _esc(", ".join(o.get("pcr_runs") or []) or "none")),
                        ("resolution", _esc(", ".join(o.get("resolution_runs") or []) or "none")),
                        ("primer design run", _esc(o.get("design_run") or "none") + f" (top {o.get('design_top')})"),
                        ("pairs compared", _esc(", ".join(o.get("compare_runs") or []) or "none")),
                        ("mitogenome", f"<i>{_esc(o.get('mito_species') or '—')}</i> {_esc(o.get('mito_record') or '')}; pairs: "
                                       f"{_esc(', '.join(o.get('mito_pairs') or []) or 'first in the library')}")]
    rep.p("<b>Choices used</b>", "body")
    rep.kv(choice_rows, widths=(0.28, 0.72))
    rep.p("<b>Contents</b>", "body")
    for i, s in enumerate(secs, 1):
        rep.p(f"{i}. {titles[s]}", "body")
    # ---- sections
    fns = {"names": lambda k: _sec_names(rep, dataset, k),
           "sequences": lambda k: _sec_sequences(rep, dataset, marker, k),
           "alignment": lambda k: _sec_alignment(rep, dataset, marker, o, k),
           "assay": lambda k: (_sec_qpcr if mode == "qpcr" else _sec_pcr)(rep, dataset, marker, o, k),
           "resolution": lambda k: _sec_resolution(rep, dataset, marker, o, k, mode),
           "design": lambda k: _sec_design(rep, dataset, marker, o, k, mode),
           "compare": lambda k: _sec_compare(rep, dataset, marker, o, k, mode),
           "mitogenome": lambda k: _sec_mito(rep, dataset, o, k, mode)}
    for i, s in enumerate(secs, 1):
        prog(f"Writing {titles[s]}", (i - 1) / max(1, len(secs)))
        try:
            fns[s](i)
        except Exception as e:  # noqa: BLE001 - one broken section should not stop the report
            log.exception("Report section %s failed", s)
            rep.p(f"This section could not be written: {_esc(e)}", "small")
    prog("Saving PDF", 0.97)
    out = paths.report_dir(dataset, marker) / f"report_{'qpcr' if mode == 'qpcr' else 'pcr'}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    footer = f"ispcr · {dataset} · {marker} · {'in silico qPCR' if mode == 'qpcr' else 'in silico PCR'}"

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 6.8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setStrokeColor(colors.HexColor(LINE))
        canvas.line(MARGIN, 11 * mm, A4[0] - MARGIN, 11 * mm)
        canvas.drawString(MARGIN, 7.5 * mm, footer)
        canvas.drawRightString(A4[0] - MARGIN, 7.5 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN + 4 * mm,
                            title=f"ispcr report {dataset} {marker}", author="ispcr")
    doc.build(rep.story, onFirstPage=on_page, onLaterPages=on_page)
    (out.with_suffix(".json")).write_text(json.dumps(o, indent=1, default=str))
    log.info("PDF report written: %s", out)
    prog("Done", 1.0)
    return out
