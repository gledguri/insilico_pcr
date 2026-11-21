"""Plotly figures shared by the Streamlit app and the HTML report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#0b6c77"
GOLD = "#d2961a"
BAD = "#c43d33"
WARN = "#b5761a"
GOOD = "#2c7a51"
GREY = "#9aa9ae"
NUC_COLORS = {"A": "#3c9a4d", "C": "#2f6dcf", "G": "#d2961a", "T": "#cf4237", "-": "#f2f4f5", "N": "#b9c2c5"}
STATUS_COLORS = {"amplified": GOOD, "fail_mismatch": BAD, "fail_length": WARN, "no_site": "#7a3d8f",
                 "site_not_covered": GREY, "not_testable": GREY, "fails": BAD, "no_sequences": "#d8dde0"}


def _layout(fig: go.Figure, h: int = 420, **kw) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", size=12), **kw)
    return fig


def species_counts(counts: pd.DataFrame) -> go.Figure:
    c = counts.sort_values(["n_clean"], ascending=True)
    fig = go.Figure()
    fig.add_bar(y=c["species"], x=c["n_gene_record"], name="gene records", orientation="h", marker_color=ACCENT)
    fig.add_bar(y=c["species"], x=c["n_mitogenome"], name="from mitogenomes", orientation="h", marker_color=GOLD)
    fig.update_layout(barmode="stack", xaxis_title="sequences after QC", yaxis=dict(tickfont=dict(size=10)))
    return _layout(fig, max(400, 16 * len(c)), legend=dict(orientation="h", y=1.02))


def length_hist(meta: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for src, col in (("gene_record", ACCENT), ("mitogenome", GOLD)):
        d = meta[meta["source"] == src]
        if len(d):
            fig.add_histogram(x=d["length"], name=src.replace("_", " "), marker_color=col, nbinsx=60, opacity=.85)
    fig.update_layout(barmode="stack", xaxis_title="length (bp)", yaxis_title="sequences")
    return _layout(fig, 300, legend=dict(orientation="h", y=1.05))


def entropy_track(cols: pd.DataFrame, primer_spans: list[tuple[str, int, int]] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=cols["column"], y=cols["entropy"], marker_color=ACCENT, name="entropy", marker_line_width=0)
    fig.add_scatter(x=cols["column"], y=cols["gap_fraction"] * 2, mode="lines", line=dict(color=GREY, width=1),
                    name="gap fraction (x2)")
    for name, a, b in primer_spans or []:
        fig.add_vrect(x0=a - .5, x1=b + .5, fillcolor=GOLD, opacity=.25, line_width=0, annotation_text=name,
                      annotation_position="top left")
    fig.update_layout(xaxis_title="alignment column", yaxis_title="Shannon entropy (bits)", bargap=0)
    return _layout(fig, 260, legend=dict(orientation="h", y=1.15))


FAMILY_PALETTE = ["#0b6c77", "#d2961a", "#7a3d8f", "#2f6dcf", "#c43d33", "#3c9a4d", "#b5761a", "#1f4e79", "#a64d79",
                  "#5b8c2a", "#e07a5f", "#3d405b", "#81b29a", "#9c6644", "#48cae4", "#6d597a", "#bc4749", "#386641"]


def family_colors(families) -> dict:
    fams = sorted({str(f) for f in families})
    return {f: FAMILY_PALETTE[i % len(FAMILY_PALETTE)] for i, f in enumerate(fams)}


LETTER_MAX_COLS, LETTER_MAX_CELLS = 60, 4000  # draw base letters only up to this window width / number of cells


def alignment_view(recs: list[tuple[str, str]], labels: list[str], start: int, end: int,
                   variable_only: bool = False, cols: pd.DataFrame | None = None,
                   primer_spans: list[tuple[str, int, int]] | None = None,
                   families: list[str] | None = None) -> go.Figure:
    """Entropy track + colour-coded alignment on one shared, numeric x axis.
    start/end are 1-based inclusive alignment columns."""
    from plotly.subplots import make_subplots
    idx = np.arange(start - 1, end)
    if variable_only and cols is not None:
        ent_all = cols["entropy"].values
        idx = idx[ent_all[idx] > 0]
    k = len(idx)
    pos = np.arange(k)
    letters = np.array([[s[i] if i < len(s) else "-" for i in idx] for _, s in recs]) if k else np.zeros((len(recs), 0), dtype=str)
    lut = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}
    z = np.vectorize(lambda b: lut.get(b, 5), otypes=[int])(letters) if letters.size else np.zeros((len(recs), 0))
    scale = [[0, NUC_COLORS["A"]], [.1667, NUC_COLORS["A"]], [.1667, NUC_COLORS["C"]], [.3333, NUC_COLORS["C"]],
             [.3333, NUC_COLORS["G"]], [.5, NUC_COLORS["G"]], [.5, NUC_COLORS["T"]], [.6667, NUC_COLORS["T"]],
             [.6667, NUC_COLORS["-"]], [.8333, NUC_COLORS["-"]], [.8333, NUC_COLORS["N"]], [1, NUC_COLORS["N"]]]
    n = len(recs)
    track_h, row_h = 70, 16
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.012,
                        row_heights=[track_h, max(row_h * n, 60)])
    if cols is not None and k:
        ent = cols["entropy"].values[idx]
        fig.add_bar(x=pos, y=ent, marker_color=ACCENT, marker_line_width=0, width=1.0, row=1, col=1,
                    customdata=idx + 1, hovertemplate="column %{customdata}<br>entropy %{y:.2f} bits<extra></extra>",
                    showlegend=False)
    # Drawing every base as a letter is slow in the browser (13,000 letters took 1.4 s per redraw), so letters
    # are drawn only when zoomed in. Hovering always shows the base.
    show_letters = k <= LETTER_MAX_COLS or k * n <= LETTER_MAX_CELLS
    rows_y = list(labels)  # row labels as categories: the hover reads the name from y, not from a per-cell copy
    if k:
        fig.add_trace(go.Heatmap(z=z, x=pos, y=rows_y, colorscale=scale, zmin=0, zmax=5, showscale=False,
                                 text=letters, texttemplate="%{text}" if show_letters else None,
                                 textfont=dict(size=9, family="IBM Plex Mono, Menlo, monospace", color="white"),
                                 customdata=np.tile(idx + 1, (n, 1)),
                                 hovertemplate="%{y}<br>column %{customdata}: <b>%{text}</b><extra></extra>",
                                 xgap=0.5 if k <= 160 else 0, ygap=1), row=2, col=1)
    colset = {int(c) + 1: p for p, c in enumerate(idx)}
    for name, a, b in primer_spans or []:
        inside = [colset[c] for c in range(a, b + 1) if c in colset]
        if inside:
            fig.add_vrect(x0=min(inside) - 0.5, x1=max(inside) + 0.5, row="all", col=1,
                          line=dict(color="#0e1e25", width=2), fillcolor="rgba(210,150,26,0.10)", layer="above")
            fig.add_annotation(x=(min(inside) + max(inside)) / 2, y=1.0, yref="paper", yanchor="bottom", text=name,
                               showarrow=False, font=dict(size=11, color="#0e1e25"), bgcolor="rgba(255,255,255,.85)")
    # x ticks show real alignment columns
    step = max(1, int(np.ceil(k / 24)))
    tick_pos = list(range(0, k, step))
    fig.update_xaxes(range=[-0.5, k - 0.5], tickvals=tick_pos, ticktext=[str(idx[p] + 1) for p in tick_pos],
                     showgrid=False, zeroline=False, row=2, col=1, tickfont=dict(size=10))
    fig.update_xaxes(range=[-0.5, k - 0.5], showgrid=False, row=1, col=1)
    tick_text = labels
    if families is not None:
        fc = family_colors(families)
        tick_text = [f"<span style='color:{fc[str(f)]}'>■</span> {lab}" for lab, f in zip(labels, families)]
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=rows_y, tickvals=rows_y, ticktext=tick_text,
                     autorange="reversed", tickfont=dict(size=10), showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="entropy", title_font=dict(size=10), tickfont=dict(size=9), showgrid=False, row=1, col=1)
    fig.update_layout(height=track_h + row_h * n + 120, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", size=12), bargap=0,
                      plot_bgcolor="white", hoverlabel=dict(font_family="IBM Plex Mono, Menlo, monospace"))
    return fig


def tree_figure(tree, meta: pd.DataFrame, highlight: set[str] | None = None, show_support: bool = True,
                label_mode: str = "species + record") -> go.Figure:
    """Rectangular phylogram of a Bio.Phylo tree. meta: indexed by seq_id with species/family."""
    highlight = highlight or set()
    depths = tree.depths() or {}
    if not max(depths.values(), default=0):
        depths = tree.depths(unit_branch_lengths=True)
    terms = tree.get_terminals()
    ypos = {t: i for i, t in enumerate(terms)}

    def y_of(cl):
        if cl in ypos:
            return ypos[cl]
        ypos[cl] = float(np.mean([y_of(c) for c in cl.clades]))
        return ypos[cl]
    y_of(tree.root)
    xs, ys = [], []
    for cl in tree.find_clades(order="level"):
        x0 = depths.get(cl, 0)
        for ch in cl.clades:
            x1 = depths.get(ch, 0)
            xs += [x0, x1, None]; ys += [ypos[ch], ypos[ch], None]
        if cl.clades:
            yy = [ypos[c] for c in cl.clades]
            xs += [x0, x0, None]; ys += [min(yy), max(yy), None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#4a5a60", width=1.2), hoverinfo="skip", showlegend=False))
    fams = [str(meta.loc[t.name, "family"]) if t.name in meta.index else "?" for t in terms]
    fc = family_colors(fams)
    maxx = max(depths.values(), default=1) or 1
    for fam in sorted(set(fams)):
        sel = [t for t, f in zip(terms, fams) if f == fam]
        sp = [meta.loc[t.name, "species"] if t.name in meta.index else t.name for t in sel]
        if label_mode == "species":
            text = sp
        elif label_mode == "record":
            text = [t.name for t in sel]
        else:
            text = [f"{s}  ·  {t.name}" for s, t in zip(sp, sel)]
        fig.add_scatter(x=[depths.get(t, 0) for t in sel], y=[ypos[t] for t in sel], mode="markers+text", name=fam,
                        text=[f"  {x}" for x in text], textposition="middle right",
                        textfont=dict(size=10, color=[BAD if t.name in highlight else "#0e1e25" for t in sel]),
                        marker=dict(color=fc[fam], size=7, line=dict(width=1.5, color=[BAD if t.name in highlight else "white" for t in sel])),
                        customdata=np.array([[s, t.name, fam] for s, t in zip(sp, sel)]),
                        hovertemplate="<i>%{customdata[0]}</i><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    if show_support:
        sx, sy, st_ = [], [], []
        for cl in tree.get_nonterminals():
            if cl.confidence is not None and cl.confidence >= 50 and cl != tree.root:
                sx.append(depths.get(cl, 0)); sy.append(ypos[cl]); st_.append(str(int(cl.confidence)))
        if sx:
            fig.add_scatter(x=sx, y=sy, mode="text", text=st_, textposition="top left", textfont=dict(size=9, color=GREY),
                            hoverinfo="skip", showlegend=False)
    n = len(terms)
    fig.update_yaxes(visible=False, autorange=False, range=[n - 0.4, -0.8])
    fig.update_xaxes(title_text="substitutions per site", range=[-0.02 * maxx, maxx * 1.55], showgrid=False, zeroline=False)
    return _layout(fig, max(450, 15 * n + 120), legend=dict(title="Family", font=dict(size=10)), plot_bgcolor="white")


def status_by_species(sp: pd.DataFrame) -> go.Figure:
    cols = ["amplified", "fail_mismatch", "fail_length", "no_site", "site_not_covered"]
    d = sp[sp["n_seqs"] > 0].sort_values(["family", "species"])
    fig = go.Figure()
    for c in cols:
        if c in d:
            fig.add_bar(x=d["species"], y=d[c], name=c.replace("_", " "), marker_color=STATUS_COLORS[c])
    fig.update_layout(barmode="stack", yaxis_title="sequences", xaxis=dict(tickangle=-60, tickfont=dict(size=9)))
    return _layout(fig, 420, legend=dict(orientation="h", y=1.08))


def mismatch_heatmap(mm: pd.DataFrame, primer: str, title: str, three_prime: int = 5, family: pd.Series | None = None) -> go.Figure:
    if family is not None:
        mm = mm.loc[sorted(mm.index, key=lambda s: (str(family.get(s, "")), s))]
    x = [f"{i + 1}{b}" for i, b in enumerate(primer)]
    fig = go.Figure(go.Heatmap(z=mm.values, x=x, y=list(mm.index), zmin=0, zmax=1,
                               colorscale=[[0, "#eef3f4"], [0.001, "#f6d5d1"], [1, BAD]],
                               colorbar=dict(title="share of<br>records", thickness=10),
                               hovertemplate="%{y}<br>primer position %{x}<br>mismatch in %{z:.0%} of records<extra></extra>",
                               xgap=1, ygap=1))
    fig.add_vrect(x0=len(primer) - three_prime - .5, x1=len(primer) - .5, line=dict(color=ACCENT, width=2),
                  fillcolor="rgba(0,0,0,0)", annotation_text="3′ end", annotation_position="top right")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    fig.update_xaxes(tickfont=dict(size=10, family="IBM Plex Mono, monospace"), side="top")
    return _layout(fig, max(300, 13 * len(mm) + 90), title=dict(text=title, x=0, font=dict(size=13)))


def insert_length_hist(hits: pd.DataFrame) -> go.Figure:
    a = hits[hits["status"] == "amplified"]
    fig = go.Figure(go.Histogram(x=a["insert_len"], marker_color=ACCENT, nbinsx=40))
    fig.update_layout(xaxis_title="amplicon length without primers (bp)", yaxis_title="sequences")
    return _layout(fig, 280)


def barcode_gap_scatter(gap: pd.DataFrame, level: str = "species") -> go.Figure:
    g = gap.copy()
    g["max_intra_pct"] = g["max_intra"].fillna(0) * 100
    g["min_inter_pct"] = g["min_inter"] * 100
    col = f"resolved_{level}"
    g["ok"] = g[col].map({True: "resolved", False: "not resolved"})
    fig = go.Figure()
    for k, c in (("resolved", ACCENT), ("not resolved", BAD)):
        d = g[g["ok"] == k]
        fig.add_scatter(x=d["max_intra_pct"], y=d["min_inter_pct"], mode="markers", name=k,
                        marker=dict(color=c, size=9, line=dict(width=1, color="white")),
                        customdata=np.stack([d["species"], d["nearest_species"].fillna(""), d["n_seqs"]], axis=1) if len(d) else None,
                        hovertemplate="<i>%{customdata[0]}</i> (n=%{customdata[2]})<br>max within species %{x:.2f}%<br>"
                                      "nearest: <i>%{customdata[1]}</i> at %{y:.2f}%<extra></extra>")
    m = max(1.0, float(np.nanmax(g[["max_intra_pct", "min_inter_pct"]].values)) if len(g) else 1)
    fig.add_scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=GREY, dash="dash"), name="no gap (y = x)")
    fig.update_layout(xaxis_title="largest distance within species (%)", yaxis_title="distance to nearest other species (%)")
    return _layout(fig, 460, legend=dict(orientation="h", y=1.08))


def distance_hist(h: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = h["bin_start"] * 100
    fig.add_bar(x=x, y=h["inter"] / max(1, h["inter"].sum()), name="between species", marker_color=GOLD, opacity=.8)
    fig.add_bar(x=x, y=h["intra"] / max(1, h["intra"].sum()), name="within species", marker_color=ACCENT, opacity=.8)
    fig.update_layout(barmode="overlay", xaxis_title="pairwise distance (%)", yaxis_title="share of pairs",
                      yaxis_tickformat=".0%", bargap=0)
    return _layout(fig, 420, legend=dict(orientation="h", y=1.08))


def conservation_plot(cons: pd.DataFrame, fwd: pd.DataFrame | None = None, rev: pd.DataFrame | None = None,
                      highlight: tuple[str, str] | None = None) -> go.Figure:
    fig = go.Figure()
    sm = cons["entropy"].rolling(5, center=True, min_periods=1).mean()
    fig.add_scatter(x=cons["column"], y=sm, fill="tozeroy", mode="lines", line=dict(color=GREY, width=1),
                    name="entropy (5-col mean)")
    if fwd is not None and len(fwd):
        fig.add_scatter(x=fwd["start_col"], y=[-0.12] * len(fwd), mode="markers", marker=dict(symbol="triangle-right", color=ACCENT, size=7),
                        name="forward candidates", customdata=fwd[["primer", "species_coverage"]].values,
                        hovertemplate="%{customdata[0]}<br>species coverage %{customdata[1]:.0%}<extra></extra>")
    if rev is not None and len(rev):
        fig.add_scatter(x=rev["end_col"], y=[-0.25] * len(rev), mode="markers", marker=dict(symbol="triangle-left", color=GOLD, size=7),
                        name="reverse candidates", customdata=rev[["primer", "species_coverage"]].values,
                        hovertemplate="%{customdata[0]}<br>species coverage %{customdata[1]:.0%}<extra></extra>")
    if highlight:
        for span, c in zip(highlight, (ACCENT, GOLD)):
            a, b = (int(v) for v in span.split("-"))
            fig.add_vrect(x0=a, x1=b, fillcolor=c, opacity=.35, line_width=0)
    fig.update_layout(xaxis_title="alignment column", yaxis_title="entropy (bits)")
    return _layout(fig, 300, legend=dict(orientation="h", y=1.15))


def compare_bars(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=rows["primer"], y=rows["species_amplified_pct"], name="species amplified", marker_color=ACCENT)
    if "species_resolved_pct" in rows:
        fig.add_bar(x=rows["primer"], y=rows["species_resolved_pct"], name="species resolved (barcode gap)", marker_color=GOLD)
    fig.update_layout(barmode="group", yaxis_title="% of testable species", yaxis_range=[0, 100])
    return _layout(fig, 360, legend=dict(orientation="h", y=1.1))


# ---------------------------------------------------------------- mitogenome map (step 8)
MITO_COLORS = {"Complex I (NADH dehydrogenase)": "#0b6c77", "Complex III (cytochrome b)": "#7a3d8f",
               "Complex IV (cytochrome c oxidase)": "#c43d33", "Complex V (ATP synthase)": "#d2961a",
               "rRNA": "#2f6dcf", "tRNA": "#9aa9ae", "Control region": "#3c9a4d", "Other": "#5a6a70"}
PCR_COLORS = {"amplified": "#2c7a51", "fail_mismatch": "#c43d33", "fail_length": "#b5761a", "no_site": "#7a3d8f"}


def mito_circle(feats: pd.DataFrame, L: int, title: str, subtitle: str = "", gc: pd.DataFrame | None = None,
                sites: pd.DataFrame | None = None, show_trna_labels: bool = True, highlight: str | None = None) -> go.Figure:
    """Circular map: + strand genes on the outer ring, − strand on the inner ring, GC content inside,
    PCR products as arcs outside. Hover any part for details."""
    deg = lambda p: (p - 1) / L * 360.0
    fig = go.Figure()
    R_PLUS, R_MINUS, TH = 0.80, 0.665, 0.115
    # backbone circles
    th = np.linspace(0, 360, 361)
    for r in (R_PLUS - 0.0075, R_MINUS + TH + 0.0075):
        fig.add_trace(go.Scatterpolar(r=[r] * len(th), theta=th, mode="lines", line=dict(color="#c9d5d9", width=1),
                                      hoverinfo="skip", showlegend=False))
    for cat in MITO_COLORS:
        d = feats[feats["category"] == cat]
        if not len(d):
            continue
        base = np.where(d["strand"] == "+", R_PLUS, R_MINUS)
        width = (d["end"] - d["start"] + 1) / L * 360
        centre = d["start"].map(deg) + width / 2
        op = [1.0 if highlight in (None, n) else 0.25 for n in d["name"]]
        fig.add_trace(go.Barpolar(
            r=[TH * (0.82 if cat == "tRNA" else 1)] * len(d), base=base + (TH * 0.09 if cat == "tRNA" else 0), theta=centre, width=width,
            marker=dict(color=MITO_COLORS[cat], line=dict(color="white", width=0.6), opacity=op), name=cat,
            legendgroup="genes", legendgrouptitle=dict(text="Genes") if cat == "Complex I (NADH dehydrogenase)" else None,
            customdata=np.stack([d["name"], d["start"], d["end"], d["strand"], d["length"], d["product"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,}–%{customdata[2]:,} (%{customdata[3]} strand)"
                          "<br>%{customdata[4]:,} bp<br>%{customdata[5]}<extra></extra>"))
    # labels: big genes inside the bar, tRNAs as one letter outside / inside the rings
    big = feats[(feats["category"] != "tRNA") & (feats["part"] == 0)]
    for r in big.itertuples():
        span = (r.end - r.start + 1) / L * 360
        mid = deg(r.start) + span / 2
        rr = (R_PLUS if r.strand == "+" else R_MINUS) + TH / 2
        if span >= 7:
            fig.add_trace(go.Scatterpolar(r=[rr], theta=[mid], mode="text", text=[r.label],
                                          textfont=dict(color="white", size=12, family="IBM Plex Sans"), hoverinfo="skip", showlegend=False))
        else:
            fig.add_trace(go.Scatterpolar(r=[(R_PLUS + TH + 0.04) if r.strand == "+" else (R_MINUS - 0.04)], theta=[mid], mode="text",
                                          text=[r.label], textfont=dict(color=MITO_COLORS[r.category], size=11), hoverinfo="skip", showlegend=False))
    if show_trna_labels:
        t = feats[(feats["category"] == "tRNA") & (feats["part"] == 0)]
        if len(t):
            mids = t["start"].map(deg) + (t["end"] - t["start"] + 1) / L * 180
            rr = np.where(t["strand"] == "+", R_PLUS + TH + 0.035, R_MINUS - 0.035)
            fig.add_trace(go.Scatterpolar(r=rr, theta=mids, mode="text", text=t["label"], hoverinfo="skip", showlegend=False,
                                          textfont=dict(size=9, color="#4a5a60", family="IBM Plex Mono")))
    # GC content ring (deviation from the genome mean)
    if gc is not None and len(gc):
        mean = gc["gc"].mean()
        dev = (gc["gc"] - mean).values
        scale = 0.10 / max(1e-6, np.abs(dev).max())
        w = (gc["end"] - gc["start"] + 1) / L * 360
        mid = gc["start"].map(deg) + w / 2
        fig.add_trace(go.Barpolar(r=np.abs(dev) * scale, base=np.where(dev >= 0, 0.50, 0.50 - np.abs(dev) * scale), theta=mid, width=w,
                                  marker=dict(color=np.where(dev >= 0, "#0b6c77", "#d2961a"), line=dict(width=0)), name="GC content",
                                  customdata=np.stack([gc["start"], gc["end"], gc["gc"] * 100], axis=1),
                                  hovertemplate="GC %{customdata[2]:.1f}%<br>%{customdata[0]:,}–%{customdata[1]:,}<extra></extra>"))
        fig.add_trace(go.Scatterpolar(r=[0.50] * len(th), theta=th, mode="lines", line=dict(color="#c9d5d9", width=0.8),
                                      hoverinfo="skip", showlegend=False))
    # PCR products: one lane and one colour per pair, named in the legend (✓ amplifies, ✗ fails)
    if sites is not None and len(sites):
        found = sites[sites["found"]]
        pal = ["#e07a5f", "#7a3d8f", "#2f6dcf", "#5b8c2a", "#a64d79", "#1f4e79", "#bc4749", "#48a9a6", "#9c6644", "#6d597a"]
        for i, s in enumerate(found.itertuples()):
            base = 0.975 + i * 0.042
            w = (s.end - s.start + 1) / L * 360
            col = pal[i % len(pal)]
            ok = s.status == "amplified"
            fig.add_trace(go.Barpolar(r=[0.03], base=[base], theta=[deg(s.start) + w / 2], width=[max(w, 0.8)],
                                      marker=dict(color=col, line=dict(width=0), pattern=dict(shape="" if ok else "/", fgcolor="white")),
                                      name=f"{'✓' if ok else '✗'} {s.pair} · {s.product_len} bp", legendgroup="pcr",
                                      legendgrouptitle=dict(text="PCR products") if i == 0 else None,
                                      customdata=[[s.pair, s.start, s.end, s.product_len, s.status, s.f_mm, s.r_mm]],
                                      hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,}–%{customdata[2]:,} · %{customdata[3]} bp"
                                                    "<br>%{customdata[4]} · mismatches F %{customdata[5]}, R %{customdata[6]}<extra></extra>"))
    # scale ticks every kb
    step = 1000 if L > 6000 else 500
    ticks = np.arange(0, L, step)
    fig.add_trace(go.Scatterpolar(r=[0.44] * len(ticks), theta=[deg(p + 1) for p in ticks], mode="text",
                                  text=["0" if p == 0 else f"{p // 1000} kb" if step == 1000 else f"{p / 1000:g} kb" for p in ticks],
                                  textfont=dict(size=9, color="#7c8d93"), hoverinfo="skip", showlegend=False))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1.38]), angularaxis=dict(visible=False, rotation=90, direction="clockwise"),
                   bgcolor="rgba(0,0,0,0)", hole=0),
        height=760, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white", font=dict(family="IBM Plex Sans, Helvetica, Arial"),
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=11), itemclick="toggle", groupclick="toggleitem",
                    tracegroupgap=14), barmode="overlay",
        annotations=[dict(text=f"<i>{title}</i><br><span style='font-size:13px;color:#587078'>{subtitle}</span>", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False, font=dict(size=17, color="#0e1e25"), xanchor="center")])
    return fig


def mito_linear(feats: pd.DataFrame, L: int, sites: pd.DataFrame | None = None) -> go.Figure:
    """Linear map with zoom: + strand on top, − strand below, PCR products underneath."""
    fig = go.Figure()
    lanes = {"+": 2, "−": 1}
    for cat, col in MITO_COLORS.items():
        d = feats[feats["category"] == cat]
        if not len(d):
            continue
        fig.add_trace(go.Bar(y=d["strand"].map(lanes), x=d["end"] - d["start"] + 1, base=d["start"] - 1, orientation="h",
                             marker=dict(color=col, line=dict(color="white", width=0.5)), name=cat, width=0.7,
                             text=d["label"], textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=11),
                             customdata=np.stack([d["name"], d["start"], d["end"], d["strand"], d["length"]], axis=1),
                             hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,}–%{customdata[2]:,} (%{customdata[3]})"
                                           "<br>%{customdata[4]:,} bp<extra></extra>"))
    ticks = {2: "+ strand", 1: "− strand"}
    if sites is not None and len(sites):
        found = sites[sites["found"]].reset_index(drop=True)
        for i, s in found.iterrows():
            y = -0.2 - i * 0.55
            ticks[y] = s["pair"]
            fig.add_trace(go.Bar(y=[y], x=[s["end"] - s["start"] + 1], base=[s["start"] - 1], orientation="h", width=0.35,
                                 marker=dict(color=PCR_COLORS.get(s["status"], "#555")), showlegend=False, name=s["pair"],
                                 hovertemplate=f"<b>{s['pair']}</b><br>{s['start']:,}–{s['end']:,} · {s['product_len']} bp<br>{s['status']}<extra></extra>"))
    fig.update_yaxes(tickvals=list(ticks), ticktext=list(ticks.values()), showgrid=False, zeroline=False)
    fig.update_xaxes(range=[0, L], title_text="position (bp)", showgrid=True, gridcolor="#eef3f4", rangeslider=dict(visible=True, thickness=0.06))
    return _layout(fig, 340 + 26 * (0 if sites is None else int(sites["found"].sum())), barmode="overlay",
                   legend=dict(orientation="h", y=1.15, font=dict(size=11)))


# ---------------------------------------------------------------- in silico qPCR (step 5)
QPCR_COLORS = {"detected": BAD, "at_risk": WARN, "blocked": GOOD, "not_testable": GREY, "no_sequences": "#d8dde0"}


def qpcr_margin(sp: pd.DataFrame, target: str, n: int = 40) -> go.Figure:
    """Fewest mismatches (F + R + probe) of each non-target species, closest first."""
    d = sp[(~sp["is_target"]) & sp["result"].isin(["detected", "at_risk", "blocked"])].copy()
    d = d.sort_values(["min_total_mm", "min_distance_pct"]).head(n).iloc[::-1]
    fig = go.Figure()
    for col, lab, c in (("min_f_mm", "forward", ACCENT), ("min_r_mm", "reverse", GOLD), ("min_p_mm", "probe", "#7a3d8f")):
        if col in d and d[col].notna().any():
            fig.add_bar(y=d["species"], x=d[col].fillna(0), name=lab, orientation="h", marker_color=c,
                        hovertemplate="<i>%{y}</i><br>" + lab + ": %{x} mismatches<extra></extra>")
    fig.add_scatter(y=d["species"], x=d["min_total_mm"] + 0.35, mode="markers", name="outcome",
                    marker=dict(color=[QPCR_COLORS.get(r, GREY) for r in d["result"]], size=10, symbol="circle"),
                    customdata=d["result"], hovertemplate="<i>%{y}</i>: %{customdata}<extra></extra>", showlegend=False)
    fig.update_layout(barmode="stack", xaxis_title="mismatches in the closest record (dot: red detected · amber at risk · green blocked)",
                      yaxis=dict(tickfont=dict(size=10)),
                      title=dict(text=f"Closest species to <i>{target}</i> at the oligo sites", x=0, font=dict(size=13)))
    return _layout(fig, max(320, 17 * len(d) + 110), legend=dict(orientation="h", y=1.06))


def _iupac_ok(o: str, t: str) -> bool:
    from .utils import IUPAC
    return t in IUPAC.get(o, "ACGT") or t in "N?"


def qpcr_rows(hits: pd.DataFrame, target: str, n: int = 30) -> pd.DataFrame:
    """Closest covered record of each species (target first), for the mismatch map."""
    h = hits[hits["covered"].astype(bool)].copy()
    h["_t"] = h["species"] != target
    h = h.sort_values(["_t", "total_mm", "distance_to_target"]).drop_duplicates("species")
    return pd.concat([h[~h["_t"]], h[h["_t"]].head(n)])


def qpcr_mismatch_map(hits: pd.DataFrame, assay: dict, target: str, n: int = 30, w: int = 5) -> go.Figure:
    rows = qpcr_rows(hits, target, n)
    blocks = [("F", assay["forward"], "f_site", True)]
    if assay.get("probe"):
        blocks.append(("P", assay["probe"], "p_site", False))
    blocks.append(("R", assay["reverse"], "r_site", True))
    xs, z, txt = [], [], []
    for lab, olig, col, is_primer in blocks:
        xs += [f"{lab}{i + 1}{b}" for i, b in enumerate(olig)] + ([f"gap{lab}"] if lab != blocks[-1][0] else [])
    for r in rows.itertuples():
        zr, tr = [], []
        for lab, olig, col, is_primer in blocks:
            site = str(getattr(r, col) or "")
            for i, b in enumerate(olig):
                t = site[i] if i < len(site) else "?"
                ok = _iupac_ok(b, t)
                zr.append(0 if ok else (2 if is_primer and i >= len(olig) - w else 1))
                tr.append("·" if ok else t)
            if lab != blocks[-1][0]:
                zr.append(None); tr.append("")
        z.append(zr); txt.append(tr)
    ylab = [f"{'▶ ' if r.species == target else ''}{r.species} · {r.outcome.replace('_', ' ')}" for r in rows.itertuples()]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ylab, text=txt, texttemplate="%{text}", textfont=dict(size=10, family="IBM Plex Mono, monospace"),
                               colorscale=[[0, "#eef3f4"], [0.5, "#f2b8b0"], [1, "#c43d33"]], zmin=0, zmax=2, showscale=False,
                               hoverinfo="skip", xgap=1, ygap=1))
    ticks = [x if not x.startswith("gap") else "" for x in xs]
    fig.update_xaxes(tickvals=xs, ticktext=[t[len(t.rstrip('ACGTRYSWKMBDHVN')) - 0:] if t else "" for t in ticks], side="top",
                     tickfont=dict(size=10, family="IBM Plex Mono, monospace"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    names = "   ".join(f"{lab}: {len(o)} nt" for lab, o, _, _ in blocks)
    return _layout(fig, max(260, 19 * len(rows) + 110), title=dict(text=f"Oligo sites in the closest record of each species ({names})", x=0, font=dict(size=13)))


def qpcr_distance(sp: pd.DataFrame, target: str, intra_pct: float | None, n: int = 40) -> go.Figure:
    d = sp[(~sp["is_target"]) & sp["min_distance_pct"].notna()].sort_values("min_distance_pct").head(n).iloc[::-1]
    fig = go.Figure(go.Bar(y=d["species"], x=d["min_distance_pct"], orientation="h",
                           marker_color=[QPCR_COLORS.get(r, GREY) for r in d["result"]], customdata=d["result"],
                           hovertemplate="<i>%{y}</i><br>%{x:.2f}% from the target · %{customdata}<extra></extra>"))
    if intra_pct is not None:
        fig.add_vline(x=intra_pct, line=dict(color=ACCENT, dash="dash"), annotation_text="largest within target",
                      annotation_position="top right")
    fig.update_layout(xaxis_title="distance of the amplicon (between the primers) to the nearest target record (%)",
                      yaxis=dict(tickfont=dict(size=10)),
                      title=dict(text=f"How different is the amplicon from <i>{target}</i>? (colour = qPCR outcome)", x=0, font=dict(size=13)))
    return _layout(fig, max(320, 17 * len(d) + 110))


def qpcr_compare_bars(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=rows["assay"], y=rows["nontarget_detected"], name="other species detected", marker_color=BAD)
    fig.add_bar(x=rows["assay"], y=rows["nontarget_at_risk"], name="other species at risk", marker_color=WARN)
    fig.add_scatter(x=rows["assay"], y=rows["target_detected_pct"], name="target records detected (%)", yaxis="y2",
                    mode="markers+lines", marker=dict(size=11, color=ACCENT))
    fig.update_layout(barmode="stack", yaxis_title="species", yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="target detected (%)"))
    return _layout(fig, 380, legend=dict(orientation="h", y=1.12))


def compare_bar(rows: pd.DataFrame, col: str, title: str, color: str = ACCENT) -> go.Figure:
    d = rows.sort_values(col, ascending=False)
    fig = go.Figure(go.Bar(x=d["primer"], y=d[col], marker_color=color, text=d[col].map(lambda v: f"{v:.0f}%"), textposition="outside"))
    fig.update_layout(yaxis_title="% of testable species", yaxis_range=[0, 108], title=dict(text=title, x=0, font=dict(size=13)))
    return _layout(fig, 340)


# ---------------------------------------------------------------- dark ("Night bench") version for the app
# The figures above are light, for the HTML and PDF reports. The app passes every figure through night(),
# which swaps the light colours for the dark theme's (same meaning: green = amplified/blocked, red = fails ...).
NIGHT = {"bg": "#111A22", "ink": "#E6EDF3", "ink2": "#C9D4DD", "muted": "#8B9BA8", "grid": "#1E2A35", "line": "#263442"}
_NIGHT_SWAP = {
    ACCENT: "#5FD4C8", GOLD: "#F2B84B", BAD: "#F47067", WARN: "#E3A33B", GOOD: "#56D364", GREY: "#6E7F8D",
    "#3c9a4d": "#56D364", "#2f6dcf": "#6CB6FF", "#cf4237": "#F47067", "#7a3d8f": "#B392F0",
    "#0e1e25": "#E6EDF3", "#4a5a60": "#A9B6C1", "#7c8d93": "#8B9BA8",
    "#eef3f4": "#1A2530", "#f2f4f5": "#16202A", "#c9d5d9": "#2A3845", "#d8dde0": "#3A4754", "#b9c2c5": "#4A5866",
    "#f6d5d1": "#3A2426", "#f2b8b0": "#6B3A35",
    "white": "#111A22", "rgba(255,255,255,.85)": "rgba(17,26,34,.9)",
}


def night(fig: go.Figure) -> go.Figure:
    """Return a dark copy of a figure made by this module (the original is left as it is)."""
    import json as _json
    import re as _re
    txt = fig.to_json()
    pat = _re.compile("|".join(_re.escape('"' + k + '"') for k in sorted(_NIGHT_SWAP, key=len, reverse=True)), _re.I)
    lower = {k.lower(): v for k, v in _NIGHT_SWAP.items()}
    txt = pat.sub(lambda m: '"' + lower[m.group(0)[1:-1].lower()] + '"', txt)
    out = go.Figure(_json.loads(txt))
    axis = dict(gridcolor=NIGHT["grid"], zerolinecolor=NIGHT["line"], linecolor=NIGHT["line"],
                tickfont_color=NIGHT["muted"], title_font_color=NIGHT["muted"])
    out.update_layout(template="plotly_dark", paper_bgcolor=NIGHT["bg"], plot_bgcolor=NIGHT["bg"],
                      font_color=NIGHT["ink2"], legend_font_color=NIGHT["ink2"], title_font_color=NIGHT["ink"],
                      hoverlabel=dict(bgcolor="#16202A", bordercolor=NIGHT["line"], font_color=NIGHT["ink"]))
    out.update_xaxes(**axis)
    out.update_yaxes(**axis)
    if out.layout.polar:
        out.update_polars(bgcolor=NIGHT["bg"])
    return out
