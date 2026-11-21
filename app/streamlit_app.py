"""ispcr app. Start with:  ./run_app.sh   (or ./env/bin/streamlit run app/streamlit_app.py)"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .` (e.g. on Streamlit Cloud)

import numpy as np
import pandas as pd
import streamlit as st

from ispcr import config, exclude, paths, plots, qc
from ispcr import selections as SEL
from ispcr.utils import read_fasta, read_species_list

st.set_page_config(page_title="ispcr · in silico PCR studio", page_icon=":material/genetics:", layout="wide")
paths.init_layout()

st.markdown("""<style>
/* ---- white theme: colours come from .streamlit/config.toml, these are the extras ---- */
:root{--bg:#FFFFFF;--panel:#FFFFFF;--panel-2:#F6F8F9;--rail:#F6F8F9;--line:#E3E8EB;--line-2:#D5DDE1;
--ink:#0F1A1F;--ink-2:#33454D;--muted:#5A6B73;--dim:#7A8A91;--accent:#0B6C77;--accent-ink:#FFFFFF;--accent-soft:#E3F1F2;
--amber:#B5761A;--red:#C43D33;--green:#2C7A51;--mono:'IBM Plex Mono',Menlo,monospace}
.block-container{padding-top:1.2rem;max-width:1500px}
h1,h2,h3{letter-spacing:-.01em;font-weight:600}
/* page header */
.ph{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;padding:4px 0 16px;border-bottom:1px solid var(--line);margin-bottom:20px}
.ph .eyebrow{font:500 .72rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}
.ph h1{font-size:1.6rem;font-weight:600;margin:0;padding:0;line-height:1.15;color:var(--ink)}
.ph .sub{color:var(--muted);margin-top:6px;font-size:.92rem;max-width:76ch}
.ph .chips{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.chip{font:.74rem var(--mono);padding:5px 10px;border-radius:6px;border:1px solid var(--line-2);background:var(--panel);color:var(--muted);white-space:nowrap}
.chip b{color:var(--ink);font-weight:500}
/* metric cards: one joined strip */
[data-testid="stMetric"]{background:var(--panel);box-shadow:0 1px 2px rgba(15,26,31,.04);border:1px solid var(--line);border-radius:10px;padding:12px 14px 10px}
[data-testid="stMetricLabel"] p{font:500 .7rem var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums;color:var(--ink)}
/* panels: bordered containers, expanders, tabs */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel)}
[data-testid="stExpander"] details{background:var(--panel);border-color:var(--line)!important}
[data-testid="stTabs"] [role="tablist"]{gap:4px;border-bottom:1px solid var(--line)}
[data-testid="stTabs"] button[role="tab"]{padding:6px 12px;border-radius:6px 6px 0 0}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:var(--accent-soft)}
/* buttons: primary is solid accent with dark text */
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"],[data-testid="stBaseButton-primary"]{color:var(--accent-ink)!important;font-weight:600}
[data-testid="stBaseButton-secondary"]{background:transparent;border-color:var(--line-2)}
[data-testid="stBaseButton-secondary"]:hover{border-color:var(--accent);color:var(--accent)}
/* tables */
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
/* alerts a bit quieter */
[data-testid="stAlert"]{border:1px solid var(--line-2)}
/* sidebar */
section[data-testid="stSidebar"]{background:var(--rail)}
section[data-testid="stSidebar"] .brand{display:flex;gap:10px;align-items:center;margin:2px 0 16px}
section[data-testid="stSidebar"] .brand .t{font-weight:600;font-size:1.1rem;letter-spacing:-.01em;line-height:1;color:var(--ink)}
section[data-testid="stSidebar"] .brand .s{font:.66rem var(--mono);color:var(--muted);letter-spacing:.08em;margin-top:4px}
section[data-testid="stSidebar"] .sec{font:500 .66rem var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin:18px 0 6px}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a{border-radius:6px;padding:5px 10px;margin:1px 0}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a span{color:var(--ink-2)}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{background:#EAF0F2}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"]{background:var(--accent-soft);box-shadow:inset 2px 0 0 var(--accent)}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] span{color:var(--ink)}
section[data-testid="stSidebar"] [data-testid="stPageLink"] span[data-testid="stIconMaterial"]{color:var(--accent)}
.net{display:inline-flex;align-items:center;gap:8px;font:.74rem var(--mono);color:var(--muted);padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
.net i{width:7px;height:7px;border-radius:50%;display:inline-block}
.prog{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin:8px 0 4px}.prog span{display:block;height:100%;background:var(--accent)}
.mono{font-family:var(--mono);font-size:.88rem;color:var(--ink-2)}
div.mono{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;line-height:1.7}
.note{font-size:.84rem;color:var(--muted)}
.note b{color:var(--ink-2)}
</style>""", unsafe_allow_html=True)

LOGO = """<svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#0B6C77"/>
<path d="M8 10h10l-3-3M24 22H14l3 3" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M8 16h16" stroke="#E0A93B" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="2 2.6"/></svg>"""


# ---------------------------------------------------------------- helpers
def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def read_csv(path: str, _m: float, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


@st.cache_data(show_spinner=False)
def read_fa(path: str, _m: float) -> list[tuple[str, str]]:
    return read_fasta(Path(path))


@st.cache_data(ttl=600, show_spinner=False)
def is_online() -> bool:
    from ispcr.ncbi import online
    return online()


def csv(p: Path, **kw) -> pd.DataFrame:
    return read_csv(str(p), mtime(p), **kw)


def fa(p: Path):
    return read_fa(str(p), mtime(p))


def fasta_text(recs) -> str:
    return "".join(f">{n}\n{s}\n" for n, s in recs)


@st.cache_data(show_spinner=False, max_entries=20)
def realign_window(rows: tuple, start: int, end: int, spans: tuple):
    """Re-align only the shown records over columns start..end (1-based) with MAFFT, so gaps opened by
    records that are not shown disappear. Returns (records, column profile, primer spans in the new columns, dropped ids)."""
    import tempfile
    from ispcr.align import run_mafft
    from ispcr.stats import alignment_matrix, column_profile
    from ispcr.utils import write_fasta
    segs = [(n, s[start - 1:end]) for n, s in rows]
    keep = [(n, seg.replace("-", "")) for n, seg in segs if len(seg.replace("-", "")) >= 5]
    dropped = [n for n, seg in segs if len(seg.replace("-", "")) < 5]
    if len(keep) < 2:
        raise ValueError("At least two of the shown records need bases in this window to re-align.")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.fasta", Path(tmp) / "out.fasta"
        write_fasta(inp, keep)
        new = dict(run_mafft(inp, out, extra=["--auto"]))  # no reverse-complementing: the pieces are already in one orientation
    new = [(n, new[n]) for n, _ in keep]
    prof = column_profile(alignment_matrix([q for _, q in new]))
    old = dict(segs)

    def new_col(n: str, col: int):  # original column (1-based) -> new column (1-based) for record n
        seg = old[n]
        i = col - start
        if i < 0 or i >= len(seg) or seg[i] == "-":
            return None
        k = len(seg[:i].replace("-", ""))  # bases before it
        q = dict(new)[n]
        cnt = -1
        for j, c in enumerate(q):
            if c != "-":
                cnt += 1
                if cnt == k:
                    return j + 1
        return None

    mapped = []
    for name, a, b in spans:
        a, b = max(a, start), min(b, end)
        if a > b:
            continue
        for n, _ in new:
            x, y = new_col(n, a), new_col(n, b)
            if x and y:
                mapped.append((name, x, y))
                break
    return new, prof, mapped, dropped



DARK_CHARTS = False


def chart(where, fig, **kw):
    """Plotly chart. Set DARK_CHARTS = True (with the dark theme in .streamlit/config.dark.toml.bak) for dark figures."""
    kw.setdefault("theme", None)
    return where.plotly_chart(plots.night(fig) if DARK_CHARTS else fig, **kw)


def drop_gap_columns(recs):
    if not recs:
        return recs
    L = max(len(s) for _, s in recs)
    keep = [i for i in range(L) if any(i < len(s) and s[i] != "-" for _, s in recs)]
    return [(n, "".join(s[i] if i < len(s) else "-" for i in keep)) for n, s in recs]


def status_badge(df: pd.DataFrame, col: str, colors: dict):
    return df.style.map(lambda v: f"color:{colors.get(v, '#333')};font-weight:600", subset=[col])


def download_csv(df: pd.DataFrame, name: str, label: str = "Download CSV", key: str | None = None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", icon=":material/download:", key=key)


def need_alignment(a: Path, msg: str) -> bool:
    """The alignment and its metadata both exist (an interrupted alignment can leave only aligned.fasta)."""
    if (a / "aligned.fasta").exists() and not (a / "aligned_metadata.csv").exists():
        st.warning(f"The {marker} alignment is incomplete (aligned_metadata.csv is missing). "
                   "Re-run 'Extract → QC → align' in step 1.", icon=":material/sync_problem:")
        return False
    return need(a / "aligned.fasta", msg)


def need(path: Path, msg: str) -> bool:
    if not path.exists():
        st.info(msg, icon=":material/info:")
        return False
    return True


def reverse_input(label: str, default: str, key: str, where=None) -> str:
    """Reverse-primer text box with a button that reverse-complements what is typed (AACT -> AGTT, IUPAC codes too).
    The box is keyed by its starting value, so choosing another primer pair still fills in that pair's primer."""
    from ispcr.utils import revcomp
    k = f"rev|{key}|{default}"
    if k not in st.session_state:
        st.session_state[k] = default

    def flip():
        st.session_state[k] = revcomp(st.session_state[k].strip().replace(" ", ""))

    c_in, c_bt = (where or st).columns([3, 1.25], vertical_alignment="bottom", gap="small")
    val = c_in.text_input(label, key=k)
    c_bt.button("Rev comp", key=f"rcbtn|{k}", on_click=flip, icon=":material/swap_horiz:", width="stretch",
                disabled=not val, help="Replace the sequence with its reverse complement, e.g. AACT → AGTT. Click again to undo.")
    return val


def header(step: int, title: str, sub: str = ""):
    chips = f"<span class='chip'>dataset <b>{ds}</b></span><span class='chip'>marker <b>{marker}</b></span>" if ds != "—" else ""
    if ds != "—":
        chips += f"<span class='chip'>mode <b>{'qPCR' if MODE == 'qpcr' else 'PCR'}</b></span>"
    st.markdown(f"<div class='ph'><div><div class='eyebrow'>Step {step} of 9 · {'qPCR' if MODE == 'qpcr' else 'PCR'} mode</div><h1>{title}</h1>"
                f"<div class='sub'>{sub}</div></div><div class='chips'>{chips}</div></div>", unsafe_allow_html=True)


def rerun_runs(runs: list[str]):
    """Re-run PCR (and resolution, when it existed) for saved runs, e.g. after exclusions changed."""
    from ispcr.pcr import PrimerPair, run_pcr
    from ispcr.resolution import run_resolution
    for r in runs:
        s = json.loads((paths.pcr_dir(ds, marker, r) / "summary.json").read_text())
        pp = s["primer_pair"]
        rules = {k: s["rules"][k] for k in ("max_mismatches", "max_3prime_mismatches", "three_prime_window") if k in s["rules"]}
        run_pcr(ds, marker, PrimerPair(pp["name"], pp["forward"], pp["reverse"], pp["product_min"], pp["product_max"],
                                       pp.get("marker", marker), pp.get("reference", "")), rules=rules)
        if (paths.resolution_dir(ds, marker, r) / "summary.json").exists():
            try:
                run_resolution(ds, marker, r)
            except ValueError:
                pass


def stale_runs() -> list[str]:
    cur = sorted(qc.excluded_ids(ds, marker))
    out = []
    for r in paths.list_pcr_runs(ds, marker):
        s = json.loads((paths.pcr_dir(ds, marker, r) / "summary.json").read_text())
        if sorted(s.get("excluded_records", [])) != cur:
            out.append(r)
    return out


# ---------------------------------------------------------------- saved selections (the PDF report uses them)
def sget(key: str, default=None, scoped: bool = True):
    return SEL.get(ds, key, marker if scoped else None, default) if ds != "—" else default


def sput(key: str, value, scoped: bool = True):
    if ds != "—":
        SEL.put(ds, key, list(value) if isinstance(value, tuple) else value, marker if scoped else None)


def idx(options: list, saved, fallback=0):
    return options.index(saved) if saved in options else fallback


def keep(options: list, saved, fallback: list) -> list:
    s = [x for x in (saved or []) if x in options]
    return s or fallback


def set_mode(m: str):
    """PCR (metabarcoding) or qPCR (species-specific): chosen in the sidebar, saved per dataset."""
    if ds != "—" and m in ("pcr", "qpcr"):
        SEL.put(ds, "mode", m)


def mode_changed():
    m = st.session_state.get("mode_ctl")
    if m is None:  # clicking the selected option again would clear it: keep the saved mode
        st.session_state["mode_ctl"] = SEL.mode(ds)
    else:
        set_mode(m)
        st.session_state["mode_jump"] = True  # on step 4, go straight to the other mode's step 4


def mode_switch(this: str) -> bool:
    """True when this page belongs to the current mode. Otherwise show how to switch."""
    if MODE == this:
        st.session_state.pop("mode_jump", None)
        return True
    if st.session_state.pop("mode_jump", False):
        st.switch_page(next(pg for pg, s in zip(pages, STEPS) if s[3] == MODE))
    lab, other = ("qPCR", "PCR") if this == "qpcr" else ("PCR", "qPCR")
    c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
    c1.info(f"This step belongs to **{lab} mode**; you are in {other} mode. The mode is chosen in the sidebar, "
            f"and the workflow, design, comparison, map and PDF report follow it.", icon=":material/swap_horiz:")
    if c2.button(f"Switch to {lab} mode", type="primary", icon=":material/swap_horiz:", width="stretch", key=f"modebtn|{this}"):
        set_mode(this)
        st.session_state.pop("mode_ds", None)  # makes the sidebar selector reload the saved mode
        st.rerun()
    return False


# ---------------------------------------------------------------- sidebar: brand + project
if "ds_pending" in st.session_state:
    st.session_state["ds"] = st.session_state.pop("ds_pending")
elif "ds" not in st.session_state and SEL.last_dataset() in paths.list_datasets():  # reopen the last dataset
    st.session_state["ds"] = SEL.last_dataset()
with st.sidebar:
    st.markdown(f"<div class='brand'>{LOGO}<div><div class='t'>ispcr</div><div class='s'>IN SILICO PCR STUDIO</div></div></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sec'>Project</div>", unsafe_allow_html=True)
    datasets = paths.list_datasets()
    ds = st.selectbox("Dataset", datasets or ["—"], key="ds", help="A dataset is a resolved species list (step 1).")
    known_markers = list(config.settings()["markers"]) + ["mitogenome"]
    have = paths.list_markers(ds) if ds != "—" else []
    m_opts = sorted(set(have) | set(known_markers), key=lambda m: (m not in have, m))
    if "marker" not in st.session_state and ds != "—" and SEL.get(ds, "marker") in m_opts:
        st.session_state["marker"] = SEL.get(ds, "marker")
    marker = st.selectbox("Marker", m_opts, key="marker", help="Markers with prepared data are listed first.")
    if ds != "—" and st.session_state.get("mode_ds") != ds:  # load the saved mode when the dataset changes
        st.session_state["mode_ctl"] = SEL.mode(ds)
        st.session_state["mode_ds"] = ds
    st.segmented_control("Mode", ["pcr", "qpcr"], key="mode_ctl", width="stretch", disabled=ds == "—",
                         format_func=lambda m: "PCR" if m == "pcr" else "qPCR",
                         on_change=mode_changed,
                         help="**PCR**: metabarcoding, amplify as many species as possible and tell them apart.  \n"
                              "**qPCR**: species-specific assay, detect one target and nothing else.  \n"
                              "The workflow below changes with the mode.")
cfg = config.settings()
MODE = (st.session_state.get("mode_ctl") or SEL.mode(ds)) if ds != "—" else "pcr"
if ds != "—":
    sput("marker", marker, scoped=False)
    SEL.set_last_dataset(ds)


def step_done() -> list[bool]:
    if ds == "—":
        return [False] * 10
    a = paths.alignment_dir(ds, marker)
    rep = paths.ROOT / "results" / "reports" / paths.safe(ds) / paths.safe(marker)
    pr, qr = paths.list_pcr_runs(ds, marker), paths.list_qpcr_runs(ds, marker)
    q = MODE == "qpcr"
    return [(paths.taxonomy_dir(ds) / "species_resolved.csv").exists(),
            (paths.qc_dir(ds, marker) / "qc_summary.json").exists(),
            (a / "aligned.fasta").exists(),
            bool(pr), bool(qr),
            bool(qr) if q else any((paths.resolution_dir(ds, marker, r) / "summary.json").exists() for r in pr),
            bool(paths.list_qpcr_design_runs(ds, marker)) if q else bool(paths.list_design_runs(ds, marker)),
            bool(sget("compare_qpcr_runs" if q else "compare_runs")),
            any(paths.genbank_dir("mitogenome").glob("*.gb")),
            rep.exists() and any(rep.glob("*.pdf"))]


# ================================================================ 1 species & download
def page_species():
    header(1, "Species & download", "Check the names, fetch sequences from NCBI once, and reuse them offline.")
    online = is_online()
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        st.subheader("Species list")
        lists = sorted(p.name for p in paths.species_lists_dir().glob("*") if p.suffix in (".txt", ".csv"))
        src = st.segmented_control("Source", ["Saved list", "Paste names", "Upload file"], default="Saved list")
        names, list_name = [], ""
        if src == "Saved list" and lists:
            pick = st.selectbox("List in data/species_lists/", lists)
            names, list_name = read_species_list(paths.species_lists_dir() / pick), Path(pick).stem
        elif src == "Paste names":
            txt = st.text_area("One Latin name per line", height=220, placeholder="Gadus morhua\nPollachius virens")
            list_name = st.text_input("Name for this list", "my_species")
            names = [n for n in (l.strip() for l in txt.splitlines()) if n and not n.startswith("#")]
        elif src == "Upload file":
            up = st.file_uploader("Text or CSV file (first column = names)", type=["txt", "csv"])
            if up:
                list_name = Path(up.name).stem
                raw = up.getvalue().decode(errors="ignore")
                names = [l.split(",")[0].strip() for l in raw.splitlines() if l.strip() and not l.startswith("#")]
        recheck = st.checkbox("Re-check names online", value=False, disabled=not online,
                              help="Off: a list that was already checked is reused without internet.")
        st.caption(f"{len(names)} names")
        if names and st.button("Resolve names", type="primary", icon=":material/fact_check:"):
            from ispcr.taxonomy import resolve
            if src != "Saved list":
                (paths.species_lists_dir() / f"{paths.safe(list_name)}.txt").write_text("\n".join(names) + "\n")
            prev = paths.taxonomy_dir(paths.safe(list_name)) / "species_resolved.csv"
            if not online and not prev.exists():
                st.error("No internet connection, and this list has not been checked before. Open a project file (sidebar → Save / open project), or connect.")
            else:
                with st.spinner("Checking names in WoRMS and NCBI Taxonomy..."):
                    resolve(names, paths.safe(list_name), force=recheck)
                st.session_state["ds_pending"] = paths.safe(list_name)
                st.rerun()
    with c2:
        st.subheader("Name check")
        if ds == "—":
            st.info("Resolve a species list to create a dataset.")
            return
        tax = csv(paths.taxonomy_dir(ds) / "species_resolved.csv", dtype={"ncbi_taxid": str})
        vc = tax["status"].value_counts()
        m = st.columns(4)
        for col, k, lab in zip(m, ["ok", "synonym", "check", "not_found"], ["accepted", "synonym updated", "to check", "not found"]):
            col.metric(lab, int(vc.get(k, 0)))
        show = tax[["input_name", "accepted_name", "family", "order", "ncbi_taxid", "status", "note"]]
        st.dataframe(status_badge(show, "status", {"ok": "#2c7a51", "synonym": "#0b6c77", "check": "#b5761a", "not_found": "#c43d33"}),
                     height=330, width="stretch", hide_index=True)

    st.subheader("Sequences")
    if not online:
        st.info("Offline: nothing will be downloaded. 'Use downloaded files' builds the tables from what is already in data/genbank/.",
                icon=":material/cloud_off:")
    d1, d2, d3, d4 = st.columns([1.4, 1, 1, 1])
    targets = d1.multiselect("Targets", ["mitogenome"] + list(cfg["markers"]),
                             default=["mitogenome", marker] if marker != "mitogenome" else ["mitogenome"])
    mx_m = d2.number_input("Max mitogenomes / species", 1, 200, cfg["download"]["max_mitogenomes_per_species"])
    mx_g = d3.number_input("Max gene records / species", 1, 500, cfg["download"]["max_gene_records_per_species"])
    force = d4.checkbox("Download again", disabled=not online, help="Ignore files already downloaded.")
    label = "Download (only what is missing)" if online else "Use downloaded files"
    if st.button(label, type="primary", disabled=not targets, icon=":material/cloud_download:" if online else ":material/folder_open:"):
        from ispcr.fetch import fetch_dataset
        for t in targets:
            bar = st.progress(0.0, text=f"{t}: starting")
            fetch_dataset(ds, t, mx_m if t == "mitogenome" else mx_g, force,
                          progress=lambda i, n, sp, t=t, bar=bar: bar.progress(min(1.0, i / max(n, 1)), text=f"{t}: {sp}"),
                          offline=not online)
        st.success("Done.")
    for t in ["mitogenome"] + list(cfg["markers"]):
        p = paths.sequences_dir(ds, t) / f"download_summary_{t}.csv"
        if p.exists():
            df = csv(p)
            with st.expander(f"{t}: {int(df['downloaded'].sum())} records, {int((df['downloaded'] == 0).sum())} species with none"):
                st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Add your own sequence files · move work to another computer", icon=":material/inventory_2:"):
        st.markdown("<div class='note'>Downloads are cached in <span class='mono'>data/genbank/</span> and are never fetched twice. "
                    "To move a dataset to another computer, or keep a copy of your work, use <b>Save / open project</b> "
                    "in the sidebar: one .zip with the downloads, every run and the choices made in the app.</div>",
                    unsafe_allow_html=True)
        e3 = st.container()
        with e3:
            st.markdown("**Add your own sequence files**")
            tgt = st.selectbox("They contain", ["mitogenome"] + list(cfg["markers"]), index=0, key="imp_tgt",
                               help="GenBank files keep their annotation, so genes can be cut out of mitogenomes. "
                                    "FASTA files are used as they are (whole sequence = this marker).")
            ups = st.file_uploader("GenBank (.gb, .gbk) or FASTA files", accept_multiple_files=True, key="seq_up",
                                   type=["gb", "gbk", "genbank", "gbff", "fasta", "fa", "fas", "fna", "txt"])
            if ups and st.button("Add to downloads", icon=":material/add_circle:"):
                from ispcr.bundle import import_sequences
                res = pd.concat([import_sequences(ds, tgt, u.name, u.getvalue()).assign(file=u.name) for u in ups])
                ok = res["species"].notna().sum()
                st.success(f"Added {ok} of {len(res)} records.")
                if ok < len(res):
                    st.warning("Records whose species could not be matched to this dataset were not added:")
                    st.dataframe(res[res["species"].isna()], hide_index=True, width="stretch")

    st.subheader(f"Prepare {marker}")
    st.caption("Cuts the marker out of gene records, mitogenomes and imported files, removes bad records, and aligns with MAFFT. Works offline.")
    if st.button(f"Extract → QC → align {marker}", icon=":material/play_arrow:"):
        from ispcr.align import align
        from ispcr.extract import extract
        from ispcr.qc import run_qc
        with st.status("Preparing sequences...", expanded=True) as s:
            st.write("Extracting..."); extract(ds, marker)
            st.write("Quality control..."); run_qc(ds, marker)
            st.write("Aligning with MAFFT..."); align(ds, marker)
            s.update(label="Done. Continue with step 2.", state="complete")


# ================================================================ 2 sequences & QC
def page_sequences():
    header(2, "Sequences & QC", "What was downloaded, what quality control removed, and which records look suspicious.")
    q = paths.qc_dir(ds, marker)
    if not need(q / "qc_summary.json", "No QC results yet. Run 'Prepare' in step 1."):
        return
    s = json.loads((q / "qc_summary.json").read_text())
    m = st.columns(5)
    m[0].metric("records kept", f"{s['kept']:,}", f"-{s['removed']} removed", delta_color="off")
    m[1].metric("species with data", f"{s['species_with_data']} / {s['species_total']}")
    m[2].metric("exact duplicates", s["exact_duplicates"])
    flags_p = paths.alignment_dir(ds, marker) / "misid_flags.csv"
    flags = csv(flags_p) if flags_p.exists() else pd.DataFrame()
    m[3].metric("flagged records", len(flags))
    m[4].metric("length limits", f"{s['length_limits'][0]}–{s['length_limits'][1]} bp")
    if s["species_without_data"]:
        st.warning("No sequences for: " + ", ".join(f"*{x}*" for x in s["species_without_data"]))
    counts = csv(q / "species_counts.csv")
    meta = csv(q / "clean_metadata.csv")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        chart(st, plots.species_counts(counts[counts["n_clean"] > 0]), width="stretch")
    with c2:
        chart(st, plots.length_hist(meta), width="stretch")
        st.markdown("**Removed by QC**")
        st.write(s["removed_by_reason"] or "nothing removed")
        st.markdown("**Possibly misidentified** · exclude them in step 3 → *Exclude records*")
        st.dataframe(flags, width="stretch", hide_index=True, height=220)
    st.subheader("All records")
    sp = st.multiselect("Filter species", sorted(meta["species"].unique()))
    view = meta[meta["species"].isin(sp)] if sp else meta
    st.dataframe(view[["seq_id", "species", "family", "source", "extraction", "length", "country", "voucher", "duplicate_of", "definition"]],
                 width="stretch", hide_index=True, height=360)
    download_csv(view, f"{ds}_{marker}_records.csv")
    rem = csv(q / "removed.csv")
    if len(rem):
        with st.expander(f"{len(rem)} removed records"):
            st.dataframe(rem, width="stretch", hide_index=True)


# ================================================================ 3 alignment
def primer_spans_from_runs(runs: list[str]) -> list[tuple[str, int, int]]:
    spans = []
    for r in runs:
        if r.startswith("qPCR "):  # species-specific assay: F, R and probe columns are in its summary
            sp_ = paths.qpcr_dir(ds, marker, r[5:]) / "summary.json"
            if sp_.exists():
                for part in json.loads(sp_.read_text()).get("site_columns", "").split(";"):
                    if part.strip():
                        d, rng = part.strip().split(" ")
                        a, b = (int(x) for x in rng.split("-"))
                        spans.append((f"{r[5:]} {d}", a, b))
            continue
        h = csv(paths.pcr_dir(ds, marker, r) / "hits.csv")
        if "site_columns" in h and h["site_columns"].notna().any():
            sc = h["site_columns"].dropna().iloc[0]
            for part in str(sc).split(";"):
                d, rng = part.strip().split(" ")
                a, b = (int(x) for x in rng.split("-"))
                spans.append((f"{r} {d}", a, b))
    return spans


@st.fragment
def tab_alignment(recs, meta, cols, L, excl):
    runs = paths.list_pcr_runs(ds, marker) + [f"qPCR {r}" for r in paths.list_qpcr_runs(ds, marker)]
    t1, t2, t3, t4, t5 = st.columns([1.6, 1, 1, 1, 1])
    cur = sget("qpcr_run") if MODE == "qpcr" else sget("pcr_run")
    cur = f"qPCR {cur}" if MODE == "qpcr" and cur else cur
    show_runs = t1.multiselect("Primer sites from PCR / qPCR runs", runs,
                               default=keep(runs, sget("align_runs"), [cur] if cur in runs else runs[:1]))
    sput("align_runs", show_runs)
    spans = primer_spans_from_runs(show_runs)
    var_only = t2.toggle("Variable sites only", value=bool(sget("al_var_only", False)))
    one_per = t3.toggle("One per species", value=bool(sget("al_one_per", True)), help="Show one record per species: the one covering most of the window")
    sort_fam = t4.toggle("Sort by taxonomy", value=bool(sget("al_sort_fam", True)))
    show_ex = t5.toggle("Show excluded", value=bool(sget("al_show_ex", False)), help="Excluded records are drawn with a ✕ in their label.")
    sput("al_var_only", var_only); sput("al_one_per", one_per); sput("al_sort_fam", sort_fam); sput("al_show_ex", show_ex)

    # --- column window: slider and number boxes stay in sync
    k = f"{ds}|{marker}"
    ks, ke, kr = f"s|{k}", f"e|{k}", f"r|{k}"
    if kr not in st.session_state:
        d0 = (max(1, spans[0][1] - 10), min(L, spans[0][2] + 140)) if spans else (1, min(L, 150))
        w0 = sget("align_window")
        if w0 and 1 <= w0[0] <= w0[1] <= L:
            d0 = tuple(w0)
        st.session_state[kr] = d0
        st.session_state[ks], st.session_state[ke] = d0

    def from_numbers():
        a, b = int(st.session_state[ks]), int(st.session_state[ke])
        a, b = max(1, min(a, L)), max(1, min(b, L))
        if a > b:
            a, b = b, a
        st.session_state[ks], st.session_state[ke], st.session_state[kr] = a, b, (a, b)

    def from_slider():
        st.session_state[ks], st.session_state[ke] = st.session_state[kr]

    def shift(direction: int):
        a, b = st.session_state[kr]
        w = b - a + 1
        a = max(1, min(L - w + 1, a + direction * w))
        st.session_state[kr] = (a, a + w - 1)
        from_slider()

    def jump():
        lab = st.session_state.get(f"j|{k}")
        for name, a, b in spans:
            if name == lab:
                w = st.session_state[kr][1] - st.session_state[kr][0] + 1
                s0 = max(1, a - 10)
                st.session_state[kr] = (s0, min(L, s0 + w - 1))
                from_slider()

    def fit():
        a, b = min(x for _, x, _ in spans), max(y for _, _, y in spans)
        st.session_state[kr] = (a, b)
        from_slider()

    c1, c2, c3, c4, c6, c5 = st.columns([1, 1, 0.6, 0.6, 1.2, 1.6], vertical_alignment="bottom")
    c1.number_input("From column", 1, L, key=ks, on_change=from_numbers, step=1)
    c2.number_input("To column", 1, L, key=ke, on_change=from_numbers, step=1)
    c3.button("◀", on_click=shift, args=(-1,), help="Previous window", width="stretch")
    c4.button("▶", on_click=shift, args=(1,), help="Next window", width="stretch")
    c6.button("Fit to primers", on_click=fit, disabled=not spans, icon=":material/fit_width:", width="stretch",
              help="Show the alignment from the first base of the forward primer to the last base of the reverse primer. "
                   "With several runs selected above, it spans all of them." if spans else "Choose a PCR or qPCR run above first.")
    if spans:
        c5.selectbox("Jump to primer site", [s[0] for s in spans], index=None, placeholder="Choose a site", key=f"j|{k}", on_change=jump)
    st.slider("Columns", 1, L, key=kr, on_change=from_slider, label_visibility="collapsed")
    rng = st.session_state[kr]
    sput("align_window", rng)

    all_sp = sorted(meta["species"].unique())
    if f"sp|{k}" not in st.session_state:
        st.session_state[f"sp|{k}"] = keep(all_sp, sget("align_species"), [])
    sel_sp = st.multiselect("Only these species", all_sp, key=f"sp|{k}",
                            help="Also available in the Tree tab: build a tree for only these species. The PDF report uses this choice.")
    sput("align_species", sel_sp)
    rows = [(n, s) for n, s in recs if n in meta.index and (show_ex or n not in excl)]
    if sel_sp:
        rows = [r for r in rows if meta.loc[r[0], "species"] in sel_sp]
    if one_per:
        best = {}  # per species, the record covering most of the window
        for n, s in rows:
            sp = meta.loc[n, "species"]
            cov = sum(c != "-" for c in s[rng[0] - 1:rng[1]])
            if sp not in best or cov > best[sp][0]:
                best[sp] = (cov, (n, s))
        rows = [v[1] for v in best.values()]
    if sort_fam:
        rows.sort(key=lambda r: (str(meta.loc[r[0], "order"]), str(meta.loc[r[0], "family"]), meta.loc[r[0], "species"]))
    total = len(rows)
    shown = rows[:400]
    lab_of = lambda n: f"{'✕ ' if n in excl else ''}{meta.loc[n, 'species']} · {n}"
    r1, r2 = st.columns([1.3, 3], vertical_alignment="center")
    realign = r1.toggle("Re-align shown records", key=f"ra|{k}", disabled=len(shown) < 2,
                        help="Runs MAFFT on only the records shown, over the columns in the window (use 'Fit to primers' "
                             "for the amplicon). Gaps that only other species needed disappear. The main alignment is not changed.")
    realigned = None
    if realign:
        try:
            with st.spinner(f"Re-aligning {len(shown)} records over columns {rng[0]}–{rng[1]}..."):
                realigned = realign_window(tuple(shown), int(rng[0]), int(rng[1]), tuple(spans))
        except Exception as err:  # noqa: BLE001
            st.error(f"Re-alignment failed: {err}")
    if realigned:
        new, prof, mspans, dropped = realigned
        r2.caption(f"Re-aligned: {len(new)} records, columns {rng[0]}–{rng[1]} of the main alignment became {len(prof)} columns "
                   f"(numbered from 1 below)." + (f" Left out (no bases in the window): {', '.join(dropped)}." if dropped else ""))
        labels = [lab_of(n) for n, _ in new]
        fams = [meta.loc[n, "family"] for n, _ in new]
        chart(st, plots.alignment_view(new, labels, 1, len(prof), var_only, prof, mspans, fams), width="stretch",
                        config={"displaylogo": False}, key=f"alv_re|{k}")
    else:
        labels = [lab_of(n) for n, _ in shown]
        fams = [meta.loc[n, "family"] for n, _ in shown]
        chart(st, plots.alignment_view(shown, labels, rng[0], rng[1], var_only, cols, spans, fams), width="stretch",
                        config={"displaylogo": False}, key=f"alv|{k}")
    if total > len(shown):
        st.caption(f"Showing the first {len(shown)} of {total} records. Narrow with 'Only these species' or 'One record per species'.")

    st.markdown("**Download what is shown**" + (" · one record per species" if one_per else ""))
    tag = ("one_per_species" if one_per else "all_records") + ("_selected" if sel_sp else "")
    if realigned:
        named0 = [(f"{n}|{meta.loc[n, 'species'].replace(' ', '_')}", q) for n, q in realigned[0]]
        st.download_button(f"Re-aligned · {len(named0)} records · columns {rng[0]}–{rng[1]}", fasta_text(named0),
                           icon=":material/download:", file_name=f"{ds}_{marker}_{tag}_realigned_cols{rng[0]}-{rng[1]}.fasta")
    d1, d2, d3 = st.columns(3)
    full = drop_gap_columns(rows)
    win = drop_gap_columns([(n, s[rng[0] - 1:rng[1]]) for n, s in rows])
    named = lambda rr: [(f"{n}|{meta.loc[n, 'species'].replace(' ', '_')}", s) for n, s in rr]
    d1.download_button(f"{len(rows)} records · full length", fasta_text(named(full)), icon=":material/download:",
                       file_name=f"{ds}_{marker}_{tag}_aligned.fasta", width="stretch")
    d2.download_button(f"{len(rows)} records · columns {rng[0]}–{rng[1]}", fasta_text(named(win)), icon=":material/download:",
                       file_name=f"{ds}_{marker}_{tag}_cols{rng[0]}-{rng[1]}.fasta", width="stretch")
    d3.download_button("Whole alignment (all records)", fasta_text(recs), icon=":material/download:",
                       file_name=f"{ds}_{marker}_aligned_all.fasta", width="stretch")
    st.caption("Headers are `accession|Genus_species`. Columns that are gaps in every downloaded record are removed.")


def tab_tree(meta, L, excl):
    from ispcr import phylo
    st.markdown("<div class='note'>Neighbour-joining tree from the alignment (K2P or p-distance), midpoint-rooted. "
                "Excluded records are left out. With all records, species that do not form one group point to misannotated "
                "or problematic sequences.</div>", unsafe_allow_html=True)
    k = f"{ds}|{marker}"
    # The Alignment tab refreshes on its own (st.fragment), so labels here stay fixed and the current
    # window / species are read when the tree is built.
    with st.form("tree_form", border=True):
        c0, c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1.3, 0.9, 0.8, 1])
        which = c0.radio("Species", ["All species", "Only species selected in the Alignment tab"],
                         index=1 if st.session_state.get(f"sp|{k}") else 0,
                         help="Pick species with 'Only these species' in the Alignment tab.")
        t_opts = sget("tree_opts") or {}
        rec_opts, reg_opts, boot_opts = ["One per species", "All records"], ["Whole alignment", "Columns shown in the Alignment tab"], [0, 100, 200, 500]
        mode = c1.radio("Records", rec_opts, index=idx(rec_opts, t_opts.get("records")), horizontal=False)
        region = c2.radio("Region", reg_opts, index=idx(reg_opts, t_opts.get("region")),
                          help="The From/To columns of the Alignment tab when you press Build tree.")
        boot = c3.selectbox("Bootstrap", boot_opts, index=idx(boot_opts, t_opts.get("bootstrap"), 1 if mode == "One per species" else 0),
                            help="Replicates. Support ≥ 50 % is shown on branches.")
        model = c4.selectbox("Distance", ["k2p", "p"], index=idx(["k2p", "p"], t_opts.get("model")))
        minc = c5.slider("Min. coverage", 0.0, 1.0, float(t_opts.get("min_coverage", 0.7)), 0.05, help="Leave out records covering less than this share of the region.")
        go = st.form_submit_button("Build tree", type="primary", icon=":material/account_tree:")
    if go:
        sput("tree_opts", {"records": mode, "region": region, "bootstrap": boot, "model": model, "min_coverage": minc})
        rng = st.session_state.get(f"r|{k}", (1, L))
        sel_sp = st.session_state.get(f"sp|{k}") or []
        if which.startswith("Only") and not sel_sp:
            st.warning("No species are selected in the Alignment tab, so the tree uses all species.", icon=":material/info:")
        bar = st.progress(0.0, text="Computing distances")
        try:
            only = sel_sp if (sel_sp and which.startswith("Only")) else None
            info = phylo.build_tree(ds, marker, "species" if mode == "One per species" else "all",
                                    None if region.startswith("Whole") else rng, model, boot, min_coverage=minc, species=only,
                                    progress=lambda f: bar.progress(f, text=f"Bootstrap {int(f * boot)}/{boot}"))
            st.session_state[f"tree|{k}"] = info["name"]
            bar.empty()
        except ValueError as e:
            bar.empty(); st.error(str(e))
    trees = phylo.list_trees(ds, marker)
    if not trees:
        st.info("No tree yet. Choose the options above and press Build tree.")
        return
    cur = st.session_state.get(f"tree|{k}") or sget("tree")
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Tree", trees, index=trees.index(cur) if cur in trees else 0)
    sput("tree", pick)
    lab_opts = ["species", "species + record", "record"]
    lab = c2.segmented_control("Labels", lab_opts, default=sget("tree_labels") if sget("tree_labels") in lab_opts else "species + record")
    if lab:
        sput("tree_labels", lab)
    tree, info = phylo.load_tree(ds, marker, pick)
    if info.get("species_selected"):
        st.caption("Species in this tree: " + ", ".join(f"*{x}*" for x in info["species_selected"]))
    m = st.columns(4)
    m[0].metric("tips", info.get("n_tips", len(tree.get_terminals())))
    m[1].metric("region", info.get("region", "—"))
    m[2].metric("bootstrap", info.get("bootstrap", 0))
    m[3].metric("species not in one group", len(info.get("not_monophyletic", [])) if info.get("mode") == "all" else "—")
    flagged = set(csv(paths.alignment_dir(ds, marker) / "misid_flags.csv")["seq_id"]) if (paths.alignment_dir(ds, marker) / "misid_flags.csv").exists() else set()
    chart(st, plots.tree_figure(tree, meta, highlight=flagged, label_mode=lab or "species + record"), width="stretch",
                    config={"displaylogo": False})
    st.caption("Red outline = flagged as possibly misidentified in QC. Branch numbers = bootstrap support (%).")
    nm = info.get("not_monophyletic", [])
    if nm:
        st.markdown("**Species whose records do not form one group**")
        df = pd.DataFrame(nm)
        st.dataframe(df, width="stretch", hide_index=True)
        odd = [x for r in nm for x in str(r.get("records_outside_own_group", "")).split("; ") if x]
        if odd and st.button(f"Exclude the {len(odd)} records that sit outside their species' group", icon=":material/block:"):
            exclude.add(ds, marker, [{"seq_id": i, "species": meta.loc[i, "species"], "reason": "outside own species in tree"} for i in odd])
            st.rerun()
    left = info.get("short_records_left_out", 0)
    if left:
        st.caption(f"{left} records were left out because they cover less than {info.get('min_coverage', 0.7):.0%} of the region.")
    d = paths.trees_dir(ds, marker)
    st.download_button("Download Newick", (d / f"{pick}.nwk").read_bytes(), file_name=f"{ds}_{marker}_{pick}.nwk", icon=":material/download:")


def tab_exclude(meta, excl):
    st.markdown("<div class='note'>Excluded records stay in the alignment file but are left out of the tree, in silico PCR, "
                "resolution statistics and primer design. You can put them back at any time.</div>", unsafe_allow_html=True)
    flags_p = paths.alignment_dir(ds, marker) / "misid_flags.csv"
    flags = csv(flags_p) if flags_p.exists() else pd.DataFrame(columns=["seq_id", "species", "reason"])
    if "reason" not in flags:  # flag files from older versions
        flags = flags.assign(reason="closer to another genus than to its own species")
    ex = exclude.load(ds, marker)
    m = st.columns(3)
    m[0].metric("excluded", len(ex))
    m[1].metric("flagged by QC", len(flags))
    m[2].metric("records kept", len(meta) - len(excl))
    f1, f2, f3 = st.columns([1.6, 1, 1], vertical_alignment="bottom")
    sp = f1.multiselect("Species", sorted(meta["species"].unique()), key="ex_sp")
    only = f2.toggle("Only flagged or excluded", value=True)
    if f3.button(f"Exclude all {len(flags)} flagged", disabled=not len(flags), icon=":material/block:"):
        exclude.add(ds, marker, [{"seq_id": r.seq_id, "species": r.species, "reason": f"QC flag: {r.reason}"} for r in flags.itertuples()])
        st.rerun()
    t = meta.reset_index()[["seq_id", "species", "family", "source", "length", "definition"]].copy()
    t["QC flag"] = t["seq_id"].map(flags.set_index("seq_id")["reason"] if len(flags) else pd.Series(dtype=object)).fillna("")
    t["exclude"] = t["seq_id"].isin(excl)
    t["reason"] = t["seq_id"].map(ex.set_index("seq_id")["reason"] if len(ex) else pd.Series(dtype=object)).fillna("")
    if sp:
        t = t[t["species"].isin(sp)]
    elif only:
        t = t[(t["QC flag"] != "") | t["exclude"]]
    with st.form("ex_form", border=False):
        ed = st.data_editor(t, hide_index=True, width="stretch", height=420, key="ex_editor",
                            disabled=["seq_id", "species", "family", "source", "length", "definition", "QC flag"],
                            column_config={"exclude": st.column_config.CheckboxColumn("exclude", help="Tick to leave this record out"),
                                           "reason": st.column_config.TextColumn("reason", help="Why (optional)"),
                                           "definition": st.column_config.TextColumn("definition", width="large")})
        save = st.form_submit_button("Save exclusions", type="primary", icon=":material/save:")
    if save:
        add = ed[ed["exclude"] & ~ed["seq_id"].isin(excl)]
        back = ed[~ed["exclude"] & ed["seq_id"].isin(excl)]
        if len(add):
            exclude.add(ds, marker, [{"seq_id": r.seq_id, "species": r.species, "reason": r.reason or "excluded by hand"} for r in add.itertuples()])
        if len(back):
            exclude.remove(ds, marker, back["seq_id"].tolist())
        st.toast(f"{len(add)} excluded, {len(back)} put back")
        st.rerun()


def page_alignment():
    header(3, "Alignment", "Inspect the aligned records, build a tree, and leave out misannotated sequences.")
    a = paths.alignment_dir(ds, marker)
    if not need_alignment(a, "No alignment yet. Run 'Prepare' in step 1."):
        return
    recs = fa(a / "aligned.fasta")
    meta = csv(a / "aligned_metadata.csv").set_index("seq_id")
    cols = csv(a / "columns.csv")
    L = len(cols)
    excl = qc.excluded_ids(ds, marker)
    m = st.columns(5)
    m[0].metric("records", f"{len(recs):,}"); m[1].metric("excluded", len(excl)); m[2].metric("columns", f"{L:,}")
    m[3].metric("variable columns", f"{int((cols['entropy'] > 0).sum()):,}"); m[4].metric("species", meta["species"].nunique())
    stale = stale_runs()
    if stale:
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.warning(f"Exclusions changed since these PCR runs: {', '.join(stale)}. Their results still include the old set.",
                   icon=":material/sync_problem:")
        if c2.button("Re-run them now", icon=":material/refresh:"):
            with st.spinner("Re-running PCR and resolution..."):
                rerun_runs(stale)
            st.rerun()
    t1, t2, t3 = st.tabs([":material/view_week: Alignment", ":material/account_tree: Tree",
                          f":material/block: Exclude records ({len(excl)})"])
    with t1:
        tab_alignment(recs, meta, cols, L, excl)
    with t2:
        tab_tree(meta, L, excl)
    with t3:
        tab_exclude(meta, excl)


# ================================================================ 4 in silico PCR
def logo_png(sites: list[str], title: str):
    import logomaker
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = [s for s in sites if isinstance(s, str) and set(s) <= set("ACGT")]
    if len(sites) < 2:
        return None
    mat = logomaker.alignment_to_matrix(sites, to_type="information")
    fig, ax = plt.subplots(figsize=(max(5, .32 * len(sites[0])), 1.8), facecolor="white")
    logomaker.Logo(mat, ax=ax, color_scheme={"A": "#3c9a4d", "C": "#2f6dcf", "G": "#d2961a", "T": "#cf4237"})
    ax.set_title(title, fontsize=9, loc="left", ); ax.set_ylabel("bits", fontsize=8)
    ax.set_xticks(range(len(sites[0]))); ax.set_xticklabels(range(1, len(sites[0]) + 1), fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()


def show_pcr_run(name: str, where: str = "view"):
    from ispcr.report import load_run
    run = load_run(ds, marker, name)
    s, hits, sp = run["summary"], run["hits"], run["species"]
    if sorted(s.get("excluded_records", [])) != sorted(qc.excluded_ids(ds, marker)):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.warning("Exclusions changed since this run, so these numbers use the old set of records.", icon=":material/sync_problem:")
        if c2.button("Re-run", key=f"rr_{where}_{name}", icon=":material/refresh:"):
            with st.spinner("Re-running..."):
                rerun_runs([name])
            st.rerun()
    st.markdown(f"<div class='mono'>F&nbsp;{s['primer_pair']['forward']}<br>R&nbsp;{s['primer_pair']['reverse']}</div>", unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric("species amplified", f"{s['species_amplified']} / {s['species_testable']}", help="testable = has at least one record covering both primer sites")
    m[1].metric("sequences amplified", f"{s['seqs_amplified']:,} / {s['seqs_testable']:,}")
    m[2].metric("records not covering a site", s["status_counts"].get("site_not_covered", 0))
    m[3].metric("median insert", f"{s['median_insert_len'] or 0:.0f} bp")
    m[4].metric("Tm F / R", f"{s['forward']['tm_min']:.0f} / {s['reverse']['tm_min']:.0f} °C",
                help=f"F {s['forward']['tm_min']}–{s['forward']['tm_max']} °C, R {s['reverse']['tm_min']}–{s['reverse']['tm_max']} °C; heterodimer ΔG {s['heterodimer_dg']} kcal/mol")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        chart(st, plots.status_by_species(sp), width="stretch", key=f"{where}|{name}|c1")
    with c2:
        if s["species_fail"]:
            st.error("Fails: " + ", ".join(f"*{x}*" for x in s["species_fail"]))
        if s["species_not_testable"]:
            st.warning("Not testable (records end before a primer site): " + ", ".join(f"*{x}*" for x in s["species_not_testable"]))
        if s["species_no_sequences"]:
            st.info("No sequences: " + ", ".join(f"*{x}*" for x in s["species_no_sequences"]))
        chart(st, plots.insert_length_hist(hits), width="stretch", key=f"{where}|{name}|c2")
    fam = sp.set_index("species")["family"]
    h1, h2 = st.columns(2)
    if len(run["mmF"]):
        chart(h1, plots.mismatch_heatmap(run["mmF"], s["primer_pair"]["forward"], "Forward primer: share of records with a mismatch", family=fam), width="stretch", key=f"{where}|{name}|hF")
        chart(h2, plots.mismatch_heatmap(run["mmR"], s["primer_pair"]["reverse"], "Reverse primer: share of records with a mismatch", family=fam), width="stretch", key=f"{where}|{name}|hR")
    tested = hits[hits["status"].isin(["amplified", "fail_mismatch", "fail_length"])]
    l1, l2 = st.columns(2)
    for col, site, lab in ((l1, "f_site", "Forward binding sites (template, primer orientation)"), (l2, "r_site", "Reverse binding sites")):
        png = logo_png(tested[site].tolist(), lab)
        if png:
            col.image(png, width="stretch")
    st.subheader("Per species")
    st.dataframe(status_badge(sp, "result", {"amplified": "#2c7a51", "fails": "#c43d33", "not_testable": "#9aa9ae", "no_sequences": "#9aa9ae"}),
                 width="stretch", hide_index=True, height=320)
    st.subheader("Per record")
    stat = st.multiselect("Status", sorted(hits["status"].unique()), key=f"st_{where}_{name}", default=[x for x in ["fail_mismatch", "fail_length", "no_site"] if x in set(hits["status"])])
    view = hits[hits["status"].isin(stat)] if stat else hits
    st.dataframe(view.drop(columns=["insert"], errors="ignore"), width="stretch", hide_index=True, height=320)
    download_csv(hits, f"{ds}_{marker}_{name}_hits.csv", key=f"dl_{where}_{name}")


def page_pcr():
    header(4, "In silico PCR", "Test a primer pair against every record: binding-site mismatches, 3′-end rule and product length.")
    if not mode_switch("pcr"):
        return
    if not need(paths.qc_dir(ds, marker) / "clean.fasta", "Prepare sequences first (step 1)."):
        return
    lib = config.primers()
    runs = paths.list_pcr_runs(ds, marker)
    tab_run, tab_view = st.tabs([":material/play_arrow: Run a primer pair", f":material/insights: Results ({len(runs)} runs)"])
    with tab_run:
        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1:
            opts = ["Custom…"] + [k for k, v in lib.items() if str(v.get("marker", "")).lower() in ("", marker.lower())] + \
                   [k for k, v in lib.items() if str(v.get("marker", "")).lower() not in ("", marker.lower())]
            choice = st.selectbox("Primer pair", opts, index=1 if len(opts) > 1 else 0)
            e = lib.get(choice, {})
            name = st.text_input("Run name", choice if choice != "Custom…" else "my_pair")
            fwd = st.text_input("Forward 5′→3′", e.get("forward", ""))
            rev = reverse_input("Reverse 5′→3′", e.get("reverse", ""), f"pcr|{choice}")
            pmin, pmax = st.slider("Expected product length incl. primers (bp)", 20, 2000,
                                   (int(e.get("product_min", 80)), int(e.get("product_max", 400))))
            if e.get("reference"):
                st.caption(e["reference"])
        with c2:
            pc = cfg["pcr"]
            r1, r2, r3 = st.columns(3)
            mm = r1.number_input("Max mismatches per primer", 0, 10, pc["max_mismatches"])
            mm3 = r2.number_input("Max mismatches at 3′ end", 0, 5, pc["max_3prime_mismatches"])
            w = r3.number_input("3′ end window (bases)", 1, 10, pc["three_prime_window"])
            if fwd and rev:
                from ispcr import thermo
                pf, pr = thermo.properties(fwd), thermo.properties(rev)
                st.dataframe(pd.DataFrame([pf, pr], index=["forward", "reverse"]), width="stretch")
                st.caption(f"Heterodimer ΔG {thermo.heterodimer_dg(fwd, rev)} kcal/mol (more negative = stronger primer-dimer)")
            b1, b2 = st.columns(2)
            go = b1.button("Run in silico PCR", type="primary", disabled=not (fwd and rev), icon=":material/play_arrow:", width="stretch")
            save = b2.button("Save pair to library", disabled=not (fwd and rev), icon=":material/bookmark_add:", width="stretch")
            n_ex = len(qc.excluded_ids(ds, marker))
            if n_ex:
                st.caption(f"{n_ex} excluded records are left out (step 3).")
        if save:
            config.save_primer(name, {"marker": marker, "forward": fwd, "reverse": rev, "product_min": pmin, "product_max": pmax,
                                      "reference": "user-defined"})
            st.success(f"Saved '{name}' to config/primers.yaml")
        if go:
            from ispcr.pcr import PrimerPair, run_pcr
            with st.spinner("Searching binding sites in every record..."):
                run_pcr(ds, marker, PrimerPair(name, fwd, rev, pmin, pmax, marker),
                        rules={"max_mismatches": mm, "max_3prime_mismatches": mm3, "three_prime_window": w})
            st.session_state["pcr_view"] = paths.safe(name)
            sput("pcr_run", paths.safe(name))
            st.success("Done. Results below and in the Results tab.")
            show_pcr_run(paths.safe(name), "run")
    with tab_view:
        if runs:
            cur = st.session_state.get("pcr_view") or sget("pcr_run")
            pick = st.selectbox("Run", runs, index=idx(runs, cur), help="The PDF report shows this run in its in silico PCR section.")
            sput("pcr_run", pick)
            show_pcr_run(pick)
        else:
            st.info("No runs yet.")


# ================================================================ 5 in silico qPCR
QPCR_GUIDE = """
**What makes a qPCR assay specific to one species?** The checks below are run on every record of every species in your list.

| What | Why it matters | Rule used here (settings.yaml → `qpcr`) |
|---|---|---|
| **Every target record matches** primers and probe | Haplotypes of the target with mismatches give late or no signal (false negatives) | target *detected*: ≤ 2 mismatches per primer, none in the last 5 bases at the 3′ end, ≤ 1 in the probe; best is 0 everywhere |
| **Mismatches at the primer 3′ end** in other species | The polymerase needs a matched 3′ end; 3′ mismatches block extension, mismatches near the 5′ end barely matter | *blocked*: ≥ 2 mismatches in the last 5 bases of one primer |
| **Total mismatches** in other species | Several mismatches anywhere lower binding and efficiency | *blocked*: ≥ 4 in one primer, or ≥ 5 in both primers together |
| **Probe mismatches** | A probe that does not bind gives no fluorescence even if something amplifies; MGB probes can discriminate single mismatches | *blocked*: ≥ 3 probe mismatches |
| **At risk** | Between *detected* and *blocked*: late (high Cq) amplification is possible at high DNA amounts | check these species in the lab (tissue DNA) |
| **Short amplicon** | eDNA is degraded; short products are detected more often | 50–150 bp |
| **Oligo chemistry** | Primers 18–25 nt, Tm 57–63 °C within 2 °C of each other, GC 35–65 %, 1–3 G/C in the last 5 bases, no runs of 4+ bases, no strong hairpins/dimers; probe Tm 5–10 °C above the primers, no G at the 5′ end, more C than G | oligo checks table |
| **Species not in your list** | Close relatives that are not in the list (or not in GenBank) cannot be checked here | also run Primer-BLAST on the final assay and test tissue of close relatives |

The rules follow MIQE (Bustin et al. 2009), mismatch studies (Kwok et al. 1990; Stadhouders et al. 2010), eDNA assay
specificity (Wilcox et al. 2013) and the eDNA assay validation scale (Thalinger et al. 2021, levels 1–2 are in silico).
"""
QOUT = {"detected": "#c43d33", "at_risk": "#b5761a", "blocked": "#2c7a51", "not_testable": "#9aa9ae", "no_sequences": "#9aa9ae",
        "not_covered": "#9aa9ae"}


def rerun_qpcr(name: str):
    from ispcr import qpcr
    s = json.loads((paths.qpcr_dir(ds, marker, name) / "summary.json").read_text())
    a = s["assay"]
    qpcr.evaluate(ds, marker, qpcr.Assay(name, a["forward"], a["reverse"], a.get("probe", ""), s["target"], marker,
                                         a.get("probe_type", "TaqMan"), a.get("reference", "")), s["target"], s["rules"])


def show_qpcr_run(name: str, where: str = "view"):
    from ispcr import qpcr
    run = qpcr.load_run(ds, marker, name)
    s, hits, sp, chk = run["summary"], run["hits"], run["species"], run["checks"]
    a, target = s["assay"], s["target"]
    if sorted(s.get("excluded_records", [])) != sorted(qc.excluded_ids(ds, marker)):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.warning("Exclusions changed since this check, so it uses the old set of records.", icon=":material/sync_problem:")
        if c2.button("Re-run", key=f"qrr_{where}_{name}", icon=":material/refresh:"):
            rerun_qpcr(name)
            st.rerun()
    v = s["verdict"]
    box = st.success if v == "specific" else st.warning if v.startswith("specific") else st.error
    box(f"**{v.capitalize()}** for *{target}* · {s['target_detected']} of {s['target_covered']} target records detected · "
        f"{len(s['nontarget_detected'])} other species detected, {len(s['nontarget_at_risk'])} at risk, "
        f"{s['nontarget_blocked']} blocked (of {s['nontarget_species_tested']} testable)",
        icon=":material/verified:" if v == "specific" else ":material/warning:")
    cols = dict(p.strip().split(" ") for p in s["site_columns"].split(";"))
    st.markdown("<div class='mono'>" + "<br>".join(
        f"{lab}&nbsp;{seq}&nbsp;&nbsp;<span class='note'>alignment columns {cols.get(k, '')}</span>"
        for lab, seq, k in (("F", a["forward"], "F"), ("R", a["reverse"], "R"), ("P", a.get("probe", ""), "P")) if seq) +
        f"<br><span class='note'>probe: {a.get('probe_type', 'none')}</span></div>", unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("target records detected", f"{s['target_detected']} / {s['target_covered']}",
                help=f"{s['target_perfect']} match every oligo perfectly. Records not reaching every oligo site are not counted.")
    m[1].metric("other species detected", len(s["nontarget_detected"]), help="would amplify and give a probe signal")
    m[2].metric("other species at risk", len(s["nontarget_at_risk"]), help="late or weak amplification possible")
    m[3].metric("other species blocked", f"{s['nontarget_blocked']} / {s['nontarget_species_tested']}")
    m[4].metric("closest species", f"{s['closest_total_mm'] if s['closest_total_mm'] is not None else '—'} mm",
                help=f"{s['closest_species']}: fewest mismatches over F + R + probe")
    m[5].metric("amplicon", f"{s['amplicon_len'] or 0:.0f} bp")
    if s["nontarget_detected"]:
        st.error("Detected (cross-reaction): " + ", ".join(f"*{x}*" for x in s["nontarget_detected"]))
    if s["nontarget_at_risk"]:
        st.warning("At risk: " + ", ".join(f"*{x}*" for x in s["nontarget_at_risk"]))
    if s["nontarget_not_testable"]:
        st.info("Not testable (records do not reach the sites): " + ", ".join(f"*{x}*" for x in s["nontarget_not_testable"]))
    c1, c2 = st.columns([1.25, 1])
    with c1:
        chart(st, plots.qpcr_margin(sp, target), width="stretch", key=f"{where}|{name}|margin")
    with c2:
        st.markdown(f"**Oligo checks** · {s['checks_passed']} of {s['checks_total']} pass")
        st.dataframe(chk.style.map(lambda x: f"color:{'#2c7a51' if x else '#c43d33'};font-weight:600", subset=["ok"]),
                     hide_index=True, width="stretch", height=420)
    chart(st, plots.qpcr_mismatch_map(hits, a, target), width="stretch", config={"displaylogo": False}, key=f"{where}|{name}|map")
    st.caption("Closest record of each species at the primer and probe sites, in oligo orientation (5′→3′). "
               "· = same base as the oligo; letters = mismatch; dark red = mismatch in the primer 3′ window.")
    st.subheader("Per species")
    spv = sp.drop(columns=["genus"], errors="ignore").copy()
    tg = spv["is_target"].astype(bool)
    spv.loc[tg, "result"] = "target: " + spv.loc[tg, "result"].astype(str)
    st.dataframe(status_badge(spv, "result", {**QOUT, "target: detected": "#2c7a51", "target: at_risk": "#b5761a", "target: blocked": "#c43d33"}),
                 hide_index=True, width="stretch", height=340)
    download_csv(sp, f"{ds}_{marker}_{name}_species.csv", key=f"dls_{where}_{name}")
    st.subheader("Per record")
    oc = st.multiselect("Outcome", ["detected", "at_risk", "blocked", "not_covered"], default=["detected", "at_risk"], key=f"qo_{where}_{name}")
    view = hits[hits["outcome"].isin(oc)] if oc else hits
    st.dataframe(view, hide_index=True, width="stretch", height=320)
    download_csv(hits, f"{ds}_{marker}_{name}_records.csv", key=f"dlr_{where}_{name}")


def page_qpcr():
    from ispcr import qpcr
    header(4, "In silico qPCR", "Is a primer + probe assay specific to one species? Detected in every record of the target, blocked in every other species.")
    if not mode_switch("qpcr"):
        with st.expander("What makes an assay species-specific", icon=":material/help:"):
            st.markdown(QPCR_GUIDE)
        return
    a = paths.alignment_dir(ds, marker)
    if not need_alignment(a, "Prepare and align sequences first (step 1)."):
        return
    meta = csv(a / "aligned_metadata.csv")
    species = sorted(meta["species"].unique())
    tgt = st.selectbox("Target species", species, index=idx(species, sget("qpcr_target"), None), placeholder="Choose the species to detect",
                       help="Saved for steps 5–9 (target vs others, design, report).")
    if tgt:
        sput("qpcr_target", tgt)
    runs = paths.list_qpcr_runs(ds, marker)
    t_run, t_view, t_guide = st.tabs([":material/play_arrow: Check an assay", f":material/insights: Results ({len(runs)})",
                                      ":material/help: What makes an assay species-specific"])
    with t_guide:
        st.markdown(QPCR_GUIDE)
    with t_run:
        if not tgt:
            st.info("Choose the target species first.")
        else:
            lib = qpcr.library()
            opts = ["Custom…"] + [k for k, v in lib.items() if v.get("target") == tgt] + [k for k, v in lib.items() if v.get("target") != tgt]
            c1, c2 = st.columns([1, 1.3], gap="large")
            with c1:
                choice = st.selectbox("Assay", opts, index=1 if len(opts) > 1 and lib[opts[1]].get("target") == tgt else 0,
                                      help="Assays in config/qpcr_assays.yaml. Designed assays are saved there from step 6.")
                e = lib.get(choice, {})
                name = st.text_input("Run name", choice if choice != "Custom…" else f"{tgt.split()[0][:1]}{tgt.split()[-1][:4]}_{marker}_assay")
                fwd = st.text_input("Forward primer 5′→3′", e.get("forward", ""))
                rev = reverse_input("Reverse primer 5′→3′", e.get("reverse", ""), f"qpcr|{choice}")
                ptype = st.segmented_control("Probe", ["TaqMan", "MGB", "none"], default=e.get("probe_type", "TaqMan") or "TaqMan",
                                             help="none = SYBR Green (no probe)") or "TaqMan"
                probe = st.text_input("Probe 5′→3′", e.get("probe", "") or "", disabled=ptype == "none")
                if e.get("target") and e.get("target") != tgt:
                    st.caption(f"This assay was made for *{e['target']}*; it will be checked against *{tgt}*.")
            with c2:
                R = qpcr.rules()
                st.markdown("**Rules** (per record)")
                r1, r2, r3 = st.columns(3)
                over = {
                    "detect_primer_mm": r1.number_input("Detected: max mm per primer", 0, 10, R["detect_primer_mm"]),
                    "detect_primer_3prime": r2.number_input("Detected: max mm at 3′ end", 0, 5, R["detect_primer_3prime"]),
                    "detect_probe_mm": r3.number_input("Detected: max probe mm", 0, 10, R["detect_probe_mm"]),
                    "block_primer_3prime": r1.number_input("Blocked: mm at one 3′ end ≥", 1, 10, R["block_primer_3prime"]),
                    "block_primer_mm": r2.number_input("Blocked: mm in one primer ≥", 1, 15, R["block_primer_mm"]),
                    "block_pair_mm": r3.number_input("Blocked: mm in both primers ≥", 1, 20, R["block_pair_mm"]),
                    "block_probe_mm": r1.number_input("Blocked: probe mm ≥", 1, 15, R["block_probe_mm"]),
                    "three_prime_window": r2.number_input("3′ window (bases)", 1, 10, R["three_prime_window"]),
                }
                st.caption("Anything between detected and blocked is at risk. Defaults are in settings.yaml → qpcr.rules.")
                b1, b2 = st.columns(2)
                go_ = b1.button("Check assay", type="primary", disabled=not (fwd and rev), icon=":material/play_arrow:", width="stretch")
                save = b2.button("Save to qPCR library", disabled=not (fwd and rev), icon=":material/bookmark_add:", width="stretch")
                n_ex = len(qc.excluded_ids(ds, marker))
                if n_ex:
                    st.caption(f"{n_ex} excluded records are left out (step 3).")
            if save:
                qpcr.save_assay(name, {"target": tgt, "marker": marker, "forward": fwd, "reverse": rev,
                                       "probe": probe if ptype != "none" else "", "probe_type": ptype,
                                       "reference": e.get("reference", "user-defined")})
                st.success(f"Saved '{name}' to config/qpcr_assays.yaml")
            if go_:
                try:
                    with st.spinner("Reading every record at the primer and probe sites..."):
                        qpcr.evaluate(ds, marker, qpcr.Assay(name, fwd, rev, probe if ptype != "none" else "", tgt, marker, ptype,
                                                             e.get("reference", "")), tgt, over)
                    sput("qpcr_run", paths.safe(name))
                    st.session_state["qpcr_view"] = paths.safe(name)
                    show_qpcr_run(paths.safe(name), "run")
                except ValueError as err:
                    st.error(str(err))
    with t_view:
        runs = paths.list_qpcr_runs(ds, marker)
        if not runs:
            st.info("No checks yet.")
        else:
            cur = st.session_state.get("qpcr_view") or sget("qpcr_run")
            pick = st.selectbox("Assay check", runs, index=idx(runs, cur), help="The PDF report shows this check.")
            sput("qpcr_run", pick)
            show_qpcr_run(pick)


# ================================================================ 6 resolution
def page_resolution():
    if MODE == "qpcr":
        return resolution_qpcr()
    header(5, "Resolution", "Can the amplicon tell species apart? Variation within species against distance to the nearest other species.")
    runs = paths.list_pcr_runs(ds, marker)
    if not runs:
        st.info("Run in silico PCR first (step 4).")
        return
    c1, c2, c3 = st.columns([1.2, 1, 1], vertical_alignment="bottom")
    pick = c1.selectbox("PCR run", runs, index=idx(runs, sget("resolution_run") or sget("pcr_run")),
                        help="The PDF report shows the resolution of this run.")
    sput("resolution_run", pick)
    model = c2.selectbox("Distance", ["k2p", "p"], index=idx(["k2p", "p"], sget("resolution_model") or cfg["resolution"]["distance_model"]))
    sput("resolution_model", model)
    rdir = paths.resolution_dir(ds, marker, pick)
    if c3.button("Compute / recompute", type="primary", icon=":material/calculate:") or not (rdir / "summary.json").exists():
        from ispcr.resolution import run_resolution
        try:
            with st.spinner("Aligning amplicons and computing distances..."):
                run_resolution(ds, marker, pick, model)
        except ValueError as e:
            st.warning(str(e)); return
    s = json.loads((rdir / "summary.json").read_text())
    gap = csv(rdir / "barcode_gap.csv")
    lv_opts = ["species", "genus", "family"]
    level = st.segmented_control("Level", lv_opts, default=sget("resolution_level") if sget("resolution_level") in lv_opts else "species") or "species"
    sput("resolution_level", level)
    m = st.columns(4)
    m[0].metric(f"identifiable to {level}", f"{s[f'resolved_{level}']} / {s['n_species']}", help="barcode gap: nearest other-group species is further than the largest within-species distance")
    m[1].metric("unique amplicons", f"{s['unique_amplicon']} / {s['n_species']}", help="species whose amplicon sequences are not shared with any other species")
    m[2].metric("median gap", f"{100 * (s['median_gap'] or 0):.2f} %")
    m[3].metric("groups with identical amplicons", len(s["identical_groups"]))
    if s["identical_groups"]:
        st.warning("Identical amplicons: " + " · ".join(", ".join(f"*{x}*" for x in g) for g in s["identical_groups"]))
    c1, c2 = st.columns(2)
    chart(c1, plots.barcode_gap_scatter(gap, level), width="stretch")
    chart(c2, plots.distance_hist(csv(rdir / "distance_histogram.csv")), width="stretch")
    st.dataframe(gap, width="stretch", hide_index=True, height=360)
    download_csv(gap, f"{ds}_{marker}_{pick}_barcode_gap.csv")
    if (rdir / "tree_nj.nwk").exists():
        st.download_button("Download NJ tree of amplicons (Newick)", (rdir / "tree_nj.nwk").read_bytes(), file_name=f"{pick}_tree.nwk",
                           icon=":material/download:")


def resolution_qpcr():
    from ispcr import qpcr
    header(5, "Target vs other species", "How different is the target's amplicon from every other species, and does that agree with the primer and probe mismatches?")
    runs = paths.list_qpcr_runs(ds, marker)
    if not runs:
        st.info("Check an assay first (step 4).")
        return
    pick = st.selectbox("Assay check", runs, index=idx(runs, sget("qpcr_run")), help="Shared with step 4 and the PDF report.")
    sput("qpcr_run", pick)
    run = qpcr.load_run(ds, marker, pick)
    s, sp = run["summary"], run["species"]
    nt = sp[(~sp["is_target"]) & sp["min_distance_pct"].notna()]
    intra = s.get("target_max_intra_pct") or 0
    same = nt[nt["min_distance_pct"] <= 0]
    m = st.columns(4)
    m[0].metric("largest distance within the target", f"{intra:.2f} %", help="between the primers, K2P")
    m[1].metric("nearest other species", f"{nt['min_distance_pct'].min():.2f} %" if len(nt) else "—",
                help=", ".join(nt.nsmallest(1, "min_distance_pct")["species"]) if len(nt) else "")
    m[2].metric("species with an identical amplicon", len(same), help="the amplicon alone cannot tell them from the target")
    m[3].metric("species further than the target's own variation", int((nt["min_distance_pct"] > intra).sum()), help="barcode gap for the target")
    if len(same):
        st.warning("Amplicon identical to some target records in: " + ", ".join(f"*{x}*" for x in same["species"])
                   + ". Here only primer/probe mismatches keep the assay specific, and a probe signal cannot be confirmed by sequencing the product.")
    chart(st, plots.qpcr_distance(sp, s["target"], intra), width="stretch")
    st.dataframe(nt.sort_values("min_distance_pct")[["species", "family", "result", "min_distance_pct", "min_total_mm", "min_f_mm", "min_r_mm",
                                                     "min_p_mm", "closest_record"]], hide_index=True, width="stretch", height=340)


# ================================================================ 7 primer design
def editor(pairs: pd.DataFrame):
    from ispcr.pcr import PrimerPair, mismatch_matrix, run_on_sequences
    st.subheader("Primer editor")
    ids = pairs["pair_id"].tolist()
    pid = st.selectbox("Start from candidate", ids)
    row = pairs.set_index("pair_id").loc[pid]
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    fwd = c1.text_input("Forward 5′→3′ (IUPAC allowed)", row["forward"], key=f"ef_{pid}")
    rev = reverse_input("Reverse 5′→3′", row["reverse"], f"er_{pid}", where=c2)
    est = int(row["product_len_est"])
    pmin, pmax = c3.slider("Product length", 20, 2000, (max(20, int(est * .7)), int(est * 1.4) + 20), key=f"el_{pid}")
    a = paths.alignment_dir(ds, marker)
    excl = qc.excluded_ids(ds, marker)
    recs = [r for r in fa(a / "aligned.fasta") if r[0] not in excl]
    meta = csv(a / "aligned_metadata.csv").set_index("seq_id")
    pair = PrimerPair("edited", fwd, rev, pmin, pmax, marker)
    with st.spinner("Testing against every record..."):
        hits = run_on_sequences([(n, s.replace("-", "")) for n, s in recs], pair, dict(recs), thermo_binding=False)
    hits = hits.join(meta[["species", "family"]], on="seq_id")
    amp = hits[hits["status"] == "amplified"]
    testable = hits[hits["status"] != "site_not_covered"]
    from ispcr import thermo
    pf, pr = thermo.properties(fwd), thermo.properties(rev)
    m = st.columns(5)
    m[0].metric("species amplified", f"{amp['species'].nunique()} / {testable['species'].nunique()}")
    m[1].metric("records amplified", f"{len(amp)} / {len(testable)}")
    m[2].metric("Tm F", f"{pf['tm_min']}–{pf['tm_max']} °C")
    m[3].metric("Tm R", f"{pr['tm_min']}–{pr['tm_max']} °C")
    m[4].metric("degeneracy", f"{pf['degeneracy']} × {pr['degeneracy']}")
    fam = meta.drop_duplicates("species").set_index("species")["family"]
    h1, h2 = st.columns(2)
    mmF, mmR = mismatch_matrix(hits, "_f_vec", len(pair.forward)), mismatch_matrix(hits, "_r_vec", len(pair.reverse))
    if len(mmF):
        chart(h1, plots.mismatch_heatmap(mmF, pair.forward, "Forward mismatches", family=fam), width="stretch")
        chart(h2, plots.mismatch_heatmap(mmR, pair.reverse, "Reverse mismatches", family=fam), width="stretch")
    fails = testable[testable["status"] != "amplified"].groupby("species").agg(
        records=("seq_id", "count"), status=("status", "first"), f_mismatches=("f_mismatches", "first"), r_mismatches=("r_mismatches", "first"))
    if len(fails):
        st.markdown("**Records that fail** (mismatch notation: position, primer base > template base)")
        st.dataframe(fails, width="stretch")
    n1, n2 = st.columns([1, 2], vertical_alignment="bottom")
    name = n1.text_input("Name", f"{pid}_edited")
    if n2.button("Save to primer library and run full PCR + resolution", type="primary", icon=":material/bookmark_add:"):
        from ispcr.pcr import run_pcr
        from ispcr.resolution import run_resolution
        config.save_primer(name, {"marker": marker, "forward": pair.forward, "reverse": pair.reverse,
                                  "product_min": pmin, "product_max": pmax, "reference": "designed with ispcr"})
        with st.spinner("Running..."):
            run_pcr(ds, marker, PrimerPair(name, pair.forward, pair.reverse, pmin, pmax, marker))
            try:
                run_resolution(ds, marker, name)
            except ValueError as e:
                st.warning(str(e))
        st.success(f"Saved as '{name}'. See steps 4 and 6.")


def page_design():
    if MODE == "qpcr":
        return design_qpcr()
    header(6, "Primer design", "Find conserved sites around a variable region, rank pairs by coverage and resolution, then refine by hand.")
    a = paths.alignment_dir(ds, marker)
    if not need_alignment(a, "Prepare and align sequences first (step 1)."):
        return
    d = cfg["design"]
    with st.expander("Search settings", expanded=not paths.list_design_runs(ds, marker), icon=":material/tune:"):
        c = st.columns(4)
        p = {"product_min": c[0].number_input("Product min (bp, incl. primers)", 40, 3000, d["product_min"]),
             "product_max": c[0].number_input("Product max", 40, 3000, d["product_max"]),
             "primer_min_len": c[1].number_input("Primer min length", 14, 35, d["primer_min_len"]),
             "primer_max_len": c[1].number_input("Primer max length", 14, 35, d["primer_max_len"]),
             "tm_min": c[2].number_input("Tm min (°C)", 40, 75, d["tm_min"]),
             "tm_max": c[2].number_input("Tm max (°C)", 40, 80, d["tm_max"]),
             "max_degeneracy": c[3].number_input("Max degeneracy (fold)", 1, 512, d["max_degeneracy"]),
             "min_base_freq": c[3].number_input("Min base frequency for degeneracy", 0.0, 0.5, float(d["min_base_freq"]), 0.01)}
        ev = st.number_input("Evaluate top N pairs with the full PCR + barcode gap", 0, 50, 10)
        n_ex = len(qc.excluded_ids(ds, marker))
        if n_ex:
            st.caption(f"{n_ex} excluded records are left out (step 3).")
        if st.button("Find candidates", type="primary", icon=":material/search:"):
            from ispcr.design import design
            bar = st.progress(0.0, text="Starting")
            try:
                design(ds, marker, p, ev, progress=lambda msg, f: bar.progress(min(1.0, f), text=msg))
                bar.progress(1.0, text="Done")
            except ValueError as e:
                st.error(str(e))
    runs = paths.list_design_runs(ds, marker)
    runs = [r for r in runs if (paths.design_dir(ds, marker, r) / "pairs.csv").exists()]
    if not runs:
        st.info("No design runs yet.")
        return
    pick = st.selectbox("Design run", runs, index=idx(runs, sget("design_run")), help="The PDF report lists the top pairs of this run.")
    sput("design_run", pick)
    rd = paths.design_dir(ds, marker, pick)
    pairs = csv(rd / "pairs.csv")
    cons = csv(rd / "conservation.csv")
    fwd = csv(rd / "forward_candidates.csv") if (rd / "forward_candidates.csv").exists() else None
    rev = csv(rd / "reverse_candidates.csv") if (rd / "reverse_candidates.csv").exists() else None
    sel = st.session_state.get("design_sel")
    hl = None
    if sel in set(pairs["pair_id"]):
        r = pairs.set_index("pair_id").loc[sel]
        hl = (r["f_cols"], r["r_cols"])
    chart(st, plots.conservation_plot(cons, fwd, rev, hl), width="stretch")
    st.subheader("Candidate pairs")
    st.caption("species_coverage = species amplified / species with records covering both sites (alignment-based); "
               "resolution_unique = share of amplified species whose amplicon is not shared with another species.")
    show = pairs[["pair_id", "forward", "reverse", "f_cols", "r_cols", "product_len_est", "species_coverage", "seq_coverage",
                  "resolution_unique", "tm_f", "tm_r", "deg_f", "deg_r", "heterodimer_dg", "score"]]
    ev = st.dataframe(show, width="stretch", hide_index=True, height=300, on_select="rerun", selection_mode="single-row")
    if ev and ev.selection.rows:
        st.session_state["design_sel"] = show.iloc[ev.selection.rows[0]]["pair_id"]
    if (rd / "evaluated_pairs.csv").exists():
        st.markdown("**Top pairs checked with the full in silico PCR and barcode-gap statistics**")
        st.dataframe(csv(rd / "evaluated_pairs.csv"), width="stretch", hide_index=True)
    download_csv(pairs, f"{ds}_{marker}_{pick}_pairs.csv")
    st.divider()
    editor(pairs)


def design_qpcr():
    from ispcr import qpcr
    header(6, "Assay design", "Primers and probe that match every record of the target and carry mismatches, at the primer 3′ ends, in every other species.")
    a = paths.alignment_dir(ds, marker)
    if not need_alignment(a, "Prepare and align sequences first (step 1)."):
        return
    species = sorted(csv(a / "aligned_metadata.csv")["species"].unique())
    tgt = st.selectbox("Target species", species, index=idx(species, sget("qpcr_target"), None), placeholder="Choose the species to detect")
    if not tgt:
        return
    sput("qpcr_target", tgt)
    d = cfg.get("qpcr", {}).get("design", {})
    with st.expander("Design settings", expanded=not paths.list_qpcr_design_runs(ds, marker), icon=":material/tune:"):
        c = st.columns(4)
        params = {"product_min": c[0].number_input("Product min (bp)", 40, 1000, int(d.get("product_min", 60))),
                  "product_max": c[0].number_input("Product max (bp)", 40, 1000, int(d.get("product_max", 150))),
                  "primer_min_len": c[1].number_input("Primer min length", 15, 30, int(d.get("primer_min_len", 18))),
                  "primer_max_len": c[1].number_input("Primer max length", 15, 35, int(d.get("primer_max_len", 25))),
                  "probe_type": c[2].selectbox("Probe", ["TaqMan", "MGB", "none"], index=["TaqMan", "MGB", "none"].index(d.get("probe_type", "TaqMan"))),
                  "target_max_mismatches": c[2].number_input("Allowed mismatches to target records", 0, 3, int(d.get("target_max_mismatches", 0))),
                  "min_target_coverage": c[3].number_input("Min share of target records covering a site", 0.1, 1.0, float(d.get("min_target_coverage", 0.5)), 0.05)}
        top = c[3].number_input("Assays to check in full", 5, 50, 20)
        st.caption("Primer Tm, GC, 3′ G/C, runs and probe rules come from settings.yaml → qpcr.checks.")
        if st.button("Design assays", type="primary", icon=":material/search:"):
            bar = st.progress(0.0, text="Starting")
            try:
                qpcr.design_assays(ds, marker, tgt, params, top, progress=lambda msg, f: bar.progress(min(1.0, f), text=msg))
                runs_ = paths.list_qpcr_design_runs(ds, marker)
                sput("qpcr_design_run", runs_[0] if runs_ else None)
            except ValueError as e:
                bar.empty(); st.error(str(e))
    runs = [r for r in paths.list_qpcr_design_runs(ds, marker) if r.startswith(paths.safe(tgt) + "_")]
    if not runs:
        st.info(f"No designs for {tgt} yet.")
        return
    pick = st.selectbox("Design run", runs, index=idx(runs, sget("qpcr_design_run")), help="The PDF report lists the assays of this run.")
    sput("qpcr_design_run", pick)
    D = csv(paths.qpcr_design_dir(ds, marker, pick) / "assays.csv")
    st.caption("Sorted by target records detected, then other species detected and at risk, then mismatches in the closest species. "
               "Select a row to check it in full and save it.")
    show = D.drop(columns=["species_not_blocked_by_primers", "margin_primers"], errors="ignore")
    ev = st.dataframe(show, hide_index=True, width="stretch", height=380, on_select="rerun", selection_mode="single-row")
    row = D.iloc[ev.selection.rows[0]] if ev and ev.selection.rows else D.iloc[0]
    st.markdown(f"**{row['assay_id']}** · {row['verdict']}")
    n1, n2 = st.columns([1, 2], vertical_alignment="bottom")
    name = n1.text_input("Save as", f"{tgt.split()[0][:1]}{tgt.split()[-1][:4]}_{marker}_{row['assay_id']}")
    if n2.button("Save to qPCR library and check it (step 4)", type="primary", icon=":material/bookmark_add:"):
        probe = row["probe"] if isinstance(row.get("probe"), str) else ""
        ptype = json.loads((paths.qpcr_design_dir(ds, marker, pick) / "params.json").read_text())["design"].get("probe_type", "TaqMan")
        qpcr.save_assay(name, {"target": tgt, "marker": marker, "forward": row["forward"], "reverse": row["reverse"], "probe": probe,
                               "probe_type": ptype if probe else "none", "reference": f"designed with ispcr ({pick} {row['assay_id']})"})
        with st.spinner("Checking..."):
            qpcr.evaluate(ds, marker, qpcr.Assay(name, row["forward"], row["reverse"], probe, tgt, marker, ptype))
        sput("qpcr_run", paths.safe(name))
        st.session_state["qpcr_view"] = paths.safe(name)
        st.success(f"Saved as '{name}'. Full results in step 4.")
    download_csv(D, f"{ds}_{marker}_{pick}_assays.csv")


# ================================================================ 8 compare & export
def page_compare():
    if MODE == "qpcr":
        return compare_qpcr()
    header(7, "Compare & export", "Published and designed primer pairs side by side, and an HTML report with every figure.")
    runs = paths.list_pcr_runs(ds, marker)
    if not runs:
        st.info("Run in silico PCR first (step 4).")
        return
    from ispcr.report import build_report, compare_runs
    pick = st.multiselect("Primer pairs", runs, default=keep(runs, sget("compare_runs"), runs),
                          help="The PDF report compares these pairs.")
    sput("compare_runs", pick)
    if not pick:
        return
    stale = [r for r in stale_runs() if r in pick]
    if stale:
        st.warning(f"Exclusions changed since: {', '.join(stale)}. Re-run them in step 3 or 4 before comparing.", icon=":material/sync_problem:")
    comp = compare_runs(ds, marker, pick)
    b1, b2 = st.columns(2)
    chart(b1, plots.compare_bar(comp, "species_amplified_pct", "Species amplified"), width="stretch")
    if "species_resolved_pct" in comp:
        chart(b2, plots.compare_bar(comp.fillna({"species_resolved_pct": 0}), "species_resolved_pct",
                                          "Species resolved (barcode gap)", plots.GOLD), width="stretch")
    st.dataframe(comp, width="stretch", hide_index=True)
    missing = [r for r in pick if not (paths.resolution_dir(ds, marker, r) / "summary.json").exists()]
    if missing:
        st.caption("No resolution statistics yet for: " + ", ".join(missing) + " (compute them in step 5).")
    download_csv(comp, f"{ds}_{marker}_primer_comparison.csv")
    if st.button("Build HTML report", type="primary", icon=":material/description:"):
        with st.spinner("Writing report..."):
            out = build_report(ds, marker, pick)
        st.success(f"Saved: {out}")
        st.download_button("Download report", Path(out).read_bytes(), file_name=Path(out).name, mime="text/html", icon=":material/download:")


def compare_qpcr():
    from ispcr import qpcr
    header(7, "Compare assays", "Species-specific assays side by side: target records detected against other species detected or at risk.")
    runs = paths.list_qpcr_runs(ds, marker)
    if not runs:
        st.info("Check an assay first (step 4).")
        return
    pick = st.multiselect("Assay checks", runs, default=keep(runs, sget("compare_qpcr_runs"), runs), help="The PDF report compares these assays.")
    sput("compare_qpcr_runs", pick)
    if not pick:
        return
    comp = qpcr.compare_runs(ds, marker, pick)
    chart(st, plots.qpcr_compare_bars(comp), width="stretch")
    st.dataframe(comp, hide_index=True, width="stretch")
    download_csv(comp, f"{ds}_{marker}_qpcr_comparison.csv")


# ================================================================ 9 mitogenome map
@st.cache_data(show_spinner=False)
def mito_record(species: str, acc: str, _m: float):
    from ispcr import mitomap
    for r in mitomap.records(species):
        if r.id == acc:
            return str(r.seq), mitomap.features(r), r.annotations.get("organism", species), r.description
    return None


@st.cache_data(show_spinner=False)
def mito_accessions(species: str, _m: float) -> list[tuple[str, int]]:
    from ispcr import mitomap
    recs = sorted(mitomap.records(species), key=lambda r: (not r.id.startswith("NC_"), r.id))
    return [(r.id, len(r.seq)) for r in recs]


@st.cache_data(show_spinner="Reading all mitogenomes...")
def mito_overview(species: tuple[str, ...], _m: float) -> pd.DataFrame:
    from ispcr import mitomap
    rows = []
    for sp in species:
        recs = sorted(mitomap.records(sp), key=lambda r: (not r.id.startswith("NC_"), r.id))
        if not recs:
            continue
        r = recs[0]
        f = mitomap.features(r)
        f = f[(f["category"] != "tRNA") & (f["part"] == 0)]
        row = {"species": sp, "record": r.id, "genome": len(r.seq),
               "GC %": round(100 * sum(b in "GC" for b in str(r.seq).upper()) / len(r.seq), 1)}
        for g in f.itertuples():
            row[g.name] = g.length
        row["gene order"] = mitomap.gene_order(mitomap.features(r))
        rows.append(row)
    return pd.DataFrame(rows)


def page_mitomap():
    from ispcr import mitomap
    from ispcr.utils import revcomp
    header(8, "Mitogenome map", "Gene layout of a complete mitochondrial genome, with where each primer pair amplifies.")
    gdir = paths.genbank_dir("mitogenome")
    species = mitomap.cached_species(ds if ds != "—" else None)
    if not species:
        st.info("No mitogenomes downloaded yet. Download 'mitogenome' in step 1.", icon=":material/info:")
        return
    tab_map, tab_all = st.tabs([":material/donut_large: Map", ":material/table_chart: Across species"])
    with tab_map:
        c1, c2, c3 = st.columns([1.4, 1.2, 1], vertical_alignment="bottom")
        default = species.index("Gadus morhua") if "Gadus morhua" in species else 0
        if MODE == "qpcr" and sget("qpcr_target") in species and not sget("mito_species"):
            default = species.index(sget("qpcr_target"))
        sp = c1.selectbox("Species", species, index=idx(species, sget("mito_species"), default), help="The PDF report shows this genome.")
        sput("mito_species", sp)
        gb = gdir / f"{paths.safe(sp)}.gb"
        accs = mito_accessions(sp, mtime(gb))
        acc_ids = [a for a, _ in accs]
        acc = c2.selectbox("Record", acc_ids, index=idx(acc_ids, sget("mito_record")), format_func=lambda a: f"{a} · {dict(accs)[a]:,} bp"
                           + ("  (RefSeq)" if a.startswith("NC_") else ""))
        sput("mito_record", acc)
        v_opts = ["Circular", "Linear", "Both"]
        view = c3.segmented_control("View", v_opts, default=sget("mito_view", scoped=False) if sget("mito_view", scoped=False) in v_opts else "Circular") or "Circular"
        sput("mito_view", view, scoped=False)
        o1, o2, o3, o4 = st.columns([2, 1.1, 0.8, 0.8], vertical_alignment="bottom")
        if MODE == "qpcr":
            from ispcr import qpcr
            lib = {k: {**v, "product_min": 0, "product_max": 3000} for k, v in qpcr.library().items()}
            pick = o1.multiselect("Show where these qPCR assays amplify", list(lib), default=keep(list(lib), sget("mito_assays"), list(lib)[:4]))
            sput("mito_assays", pick)
        else:
            lib = config.primers()
            pick = o1.multiselect("Show where these primer pairs amplify", list(lib), default=keep(list(lib), sget("mito_pairs"), list(lib)[:4]))
            sput("mito_pairs", pick)
        res = mito_record(sp, acc, mtime(gb))
        if res is None:
            st.error("Record not found."); return
        seq, feats, organism, desc = res
        genes = sorted(feats.loc[feats["category"] != "tRNA", "name"].unique())
        hl = o2.selectbox("Highlight gene", ["—"] + genes, index=idx(["—"] + genes, sget("mito_highlight", scoped=False)))
        show_gc = o3.toggle("GC ring", value=bool(sget("mito_gc", True, scoped=False)))
        show_t = o4.toggle("tRNA labels", value=bool(sget("mito_trna", True, scoped=False)))
        sput("mito_highlight", hl, scoped=False); sput("mito_gc", show_gc, scoped=False); sput("mito_trna", show_t, scoped=False)
        sites = mitomap.primer_sites(seq, {k: lib[k] for k in pick}) if pick else None
        L = len(seq)
        gcp = 100 * sum(b in "GC" for b in seq.upper()) / L
        n = feats[feats["part"] == 0]["category"].value_counts()
        m = st.columns(5)
        m[0].metric("genome length", f"{L:,} bp")
        m[1].metric("GC content", f"{gcp:.1f} %")
        m[2].metric("protein-coding genes", int(sum(v for k, v in n.items() if k.startswith("Complex"))))
        m[3].metric("rRNA / tRNA genes", f"{int(n.get('rRNA', 0))} / {int(n.get('tRNA', 0))}")
        cr = feats[feats["category"] == "Control region"]
        m[4].metric("control region", f"{int(cr['length'].iloc[0]):,} bp" if len(cr) else "not annotated")
        if not len(feats):
            st.warning("This record has no gene annotation, so only the length can be shown.")
        if view in ("Circular", "Both"):
            fig = plots.mito_circle(feats, L, organism, f"{acc} · {L:,} bp · GC {gcp:.1f}%",
                                    mitomap.gc_windows(seq) if show_gc else None, sites, show_t, None if hl == "—" else hl)
            chart(st, fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"format": "svg", "filename": f"{acc}_map"}})
            st.caption("Outer ring: genes on the + (heavy) strand · inner ring: − strand · centre ring: GC content above (teal) "
                       "or below (gold) the genome average · arcs outside: PCR products (✓ amplifies; hatched ✗ fails the mismatch or length rules). "
                       "Click legend entries to hide groups; the camera icon saves an SVG.")
        if view in ("Linear", "Both"):
            chart(st, plots.mito_linear(feats, L, sites), width="stretch", config={"displaylogo": False})
            st.caption("Drag the slider under the map, or draw a box, to zoom into a region.")
        if sites is not None and len(sites):
            st.markdown("**Primer pairs on this genome**")
            show = sites.copy()
            if "start" in show:
                show["genes covered"] = [", ".join(feats[(feats["end"] >= r.start) & (feats["start"] <= r.end)]["label"].unique())
                                         if r.found else "" for r in show.itertuples()]
            st.dataframe(show, hide_index=True, width="stretch")
        st.markdown("**Genes**")
        tbl = feats[feats["part"] == 0][["label", "name", "category", "start", "end", "strand", "length", "product"]]
        st.dataframe(tbl, hide_index=True, width="stretch", height=300)
        d1, d2, d3 = st.columns([1.2, 1, 1], vertical_alignment="bottom")
        g = d1.selectbox("Gene sequence to download", tbl["name"].tolist())
        parts = feats[feats["name"] == g].sort_values("part")
        gs = "".join(seq[r.start - 1:r.end] for r in parts.itertuples())
        if len(parts) and parts["strand"].iloc[0] == "−":
            gs = revcomp(gs)
        d2.download_button(f"{g} ({len(gs)} bp) FASTA", f">{acc}|{sp.replace(' ', '_')}|{g}\n{gs}\n", file_name=f"{acc}_{paths.safe(g)}.fasta",
                           icon=":material/download:", width="stretch")
        d3.download_button("Whole genome FASTA", f">{acc}|{sp.replace(' ', '_')}\n{seq}\n", file_name=f"{acc}.fasta",
                           icon=":material/download:", width="stretch")
    with tab_all:
        st.markdown("<div class='note'>One record per species (RefSeq first). Gene lengths in bp; differences in gene order point to "
                    "rearranged genomes or annotation errors.</div>", unsafe_allow_html=True)
        ov = mito_overview(tuple(species), max(mtime(p) for p in gdir.glob("*.gb")))
        if not len(ov):
            return
        common = ov["gene order"].mode().iloc[0]
        ov["standard order"] = ov["gene order"] == common
        m = st.columns(4)
        m[0].metric("species with a mitogenome", len(ov))
        m[1].metric("genome length", f"{ov['genome'].min():,}–{ov['genome'].max():,} bp")
        m[2].metric("GC content", f"{ov['GC %'].min()}–{ov['GC %'].max()} %")
        m[3].metric("gene order differs", int((~ov["standard order"]).sum()), help="Protein-coding and rRNA genes, read from 12S; "
                    "differences are real rearrangements or annotation errors in the GenBank record")
        gene_cols = [c for c in ["12S", "16S", "ND1", "ND2", "COI", "COII", "ATP8", "ATP6", "COIII", "ND3", "ND4L", "ND4", "ND5", "ND6",
                                 "CYTB", "Control region"] if c in ov]
        import plotly.graph_objects as go
        z = ov.set_index("species")[gene_cols]
        rel = (z - z.median()) / z.median()
        fig = go.Figure(go.Heatmap(z=rel.values, x=gene_cols, y=z.index, customdata=z.values, zmid=0, colorscale="RdBu_r", zmin=-0.3, zmax=0.3,
                                   colorbar=dict(title="vs median", tickformat=".0%"),
                                   hovertemplate="<i>%{y}</i><br>%{x}: %{customdata} bp<br>%{z:+.0%} vs median<extra></extra>", xgap=1, ygap=1))
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        fig.update_xaxes(side="top", tickfont=dict(size=11))
        fig.update_layout(height=max(400, 16 * len(z) + 120), margin=dict(l=10, r=10, t=70, b=10), template="plotly_white",
                          title=dict(text="Gene length compared with the median across species", font=dict(size=13), x=0))
        chart(st, fig, width="stretch")
        st.dataframe(ov, hide_index=True, width="stretch", height=360)
        download_csv(ov, f"{ds}_mitogenome_overview.csv")


# ================================================================ 10 PDF report
def page_report():
    from ispcr import pdfreport
    from ispcr.phylo import list_trees
    header(9, "PDF report", "Every step in one PDF, built from what you chose in steps 1–8. Change a choice here and it is changed in that step too.")
    q = MODE == "qpcr"
    o = pdfreport.default_options(ds, marker)
    titles = pdfreport.QSECTION_TITLES if q else pdfreport.SECTION_TITLES
    st.markdown(f"<div class='note'>Mode: <b>{'in silico qPCR' if q else 'in silico PCR'}</b> (switch it in the sidebar). "
                f"The report has the {'qPCR' if q else 'PCR'} versions of steps 4–7.</div>", unsafe_allow_html=True)
    SECS = pdfreport.SECTIONS
    secs = st.multiselect("Sections", SECS, default=keep(SECS, sget("report_sections", scoped=False), SECS), format_func=lambda s: titles[s])
    sput("report_sections", secs, scoped=False)
    pr, qr = paths.list_pcr_runs(ds, marker), paths.list_qpcr_runs(ds, marker)
    a = paths.alignment_dir(ds, marker)
    all_sp = sorted(csv(a / "aligned_metadata.csv")["species"].unique()) if (a / "aligned_metadata.csv").exists() else []
    trees = list_trees(ds, marker) if (a / "aligned.fasta").exists() else []
    opts = {"mode": MODE, "sections": secs}
    with st.container(border=True):
        st.markdown("**Choices used in the report**")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("<div class='note'>STEP 3 · ALIGNMENT</div>", unsafe_allow_html=True)
            site_opts = pr + [f"qPCR {r}" for r in qr]
            opts["align_runs"] = st.multiselect("Primer sites shown", site_opts, default=keep(site_opts, o["align_runs"], []))
            opts["align_species"] = st.multiselect("Species (empty = all, one record each)", all_sp, default=keep(all_sp, o["align_species"], []))
            t = st.selectbox("Tree", ["none"] + trees, index=idx(["none"] + trees, o["tree"]))
            opts["tree"] = None if t == "none" else t
            sput("align_runs", opts["align_runs"]); sput("align_species", opts["align_species"]); sput("tree", opts["tree"])
        with c2:
            if q:
                st.markdown("<div class='note'>STEPS 4–7 · qPCR</div>", unsafe_allow_html=True)
                opts["qpcr_run"] = st.selectbox("qPCR assay check (steps 4–5)", qr, index=idx(qr, o["qpcr_run"]) if qr else None)
                qd = paths.list_qpcr_design_runs(ds, marker)
                opts["qpcr_design_run"] = st.selectbox("Assay design run (step 6)", qd, index=idx(qd, o["qpcr_design_run"]) if qd else None)
                opts["compare_qpcr_runs"] = st.multiselect("Assays compared (step 7)", qr, default=keep(qr, o["compare_qpcr_runs"], qr))
                sput("qpcr_run", opts["qpcr_run"]); sput("qpcr_design_run", opts["qpcr_design_run"])
                sput("compare_qpcr_runs", opts["compare_qpcr_runs"])
            else:
                st.markdown("<div class='note'>STEPS 4–7 · PCR</div>", unsafe_allow_html=True)
                opts["pcr_runs"] = st.multiselect("In silico PCR runs (step 4)", pr, default=keep(pr, o["pcr_runs"], pr[:1]))
                res_ok = [r for r in pr if (paths.resolution_dir(ds, marker, r) / "summary.json").exists()]
                opts["resolution_runs"] = st.multiselect("Resolution (step 5)", res_ok, default=keep(res_ok, o["resolution_runs"], res_ok[:1]))
                dr = [r for r in paths.list_design_runs(ds, marker) if (paths.design_dir(ds, marker, r) / "pairs.csv").exists()]
                opts["design_run"] = st.selectbox("Primer design run (step 6)", dr, index=idx(dr, o["design_run"]) if dr else None)
                opts["compare_runs"] = st.multiselect("Pairs compared (step 7)", pr, default=keep(pr, o["compare_runs"], pr))
                if opts["pcr_runs"]:
                    sput("pcr_run", opts["pcr_runs"][0])
                if opts["resolution_runs"]:
                    sput("resolution_run", opts["resolution_runs"][0])
                sput("design_run", opts["design_run"]); sput("compare_runs", opts["compare_runs"])
            opts["design_top"] = st.segmented_control("Design: top candidates", [10, 20], default=int(sget("design_top", 20) or 20)) or 20
            sput("design_top", opts["design_top"])
        with c3:
            from ispcr import mitomap
            st.markdown("<div class='note'>STEP 8 · MITOGENOME MAP</div>", unsafe_allow_html=True)
            msp = mitomap.cached_species(ds)
            opts["mito_species"] = st.selectbox("Species", msp, index=idx(msp, o["mito_species"]) if msp else None)
            if opts["mito_species"]:
                gb = paths.genbank_dir("mitogenome") / f"{paths.safe(opts['mito_species'])}.gb"
                accs = [x for x, _ in mito_accessions(opts["mito_species"], mtime(gb))]
                opts["mito_record"] = st.selectbox("Record", accs, index=idx(accs, o["mito_record"]))
            if q:
                from ispcr import qpcr
                lib = list(qpcr.library())
                opts["mito_assays"] = st.multiselect("qPCR assays drawn", lib, default=keep(lib, o["mito_assays"], [o["qpcr_run"]] if o["qpcr_run"] in lib else lib[:4]))
                sput("mito_assays", opts["mito_assays"])
            else:
                lib = list(config.primers())
                opts["mito_pairs"] = st.multiselect("Primer pairs drawn", lib, default=keep(lib, o["mito_pairs"], lib[:4]))
                sput("mito_pairs", opts["mito_pairs"])
            sput("mito_species", opts["mito_species"]); sput("mito_record", opts.get("mito_record"))
    if st.button("Generate PDF", type="primary", icon=":material/picture_as_pdf:", disabled=not secs):
        bar = st.progress(0.0, text="Starting")
        try:
            out = pdfreport.build_pdf(ds, marker, opts, progress=lambda m_, f: bar.progress(min(1.0, f), text=m_))
            st.session_state["last_pdf"] = str(out)
            bar.empty()
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(f"The report could not be written: {e}")
    last = st.session_state.get("last_pdf")
    if last and Path(last).exists():
        st.success(f"Saved: {last}", icon=":material/check_circle:")
        st.download_button(f"Download {Path(last).name}", Path(last).read_bytes(), file_name=Path(last).name, mime="application/pdf",
                           type="primary", icon=":material/download:")
    rd = paths.ROOT / "results" / "reports" / paths.safe(ds) / paths.safe(marker)
    old = sorted(rd.glob("*.pdf"), key=lambda p_: p_.stat().st_mtime, reverse=True) if rd.exists() else []
    if old:
        with st.expander(f"Earlier reports ({len(old)})", icon=":material/history:"):
            for p_ in old[:10]:
                c1, c2 = st.columns([3, 1], vertical_alignment="center")
                c1.markdown(f"<span class='mono'>{p_.name}</span> <span class='note'>{p_.stat().st_size / 1e6:.1f} MB</span>", unsafe_allow_html=True)
                c2.download_button("Download", p_.read_bytes(), file_name=p_.name, mime="application/pdf", key=f"dl_{p_.name}",
                                   icon=":material/download:", width="stretch")


# ---------------------------------------------------------------- primer library (not a step)
@st.dialog("Remove from library?")
def confirm_remove(kind: str, names: list[str]):
    from ispcr import qpcr
    fname = "primers.yaml" if kind == "pcr" else "qpcr_assays.yaml"
    st.markdown(f"This removes **{len(names)}** {'primer pair' if kind == 'pcr' else 'assay'}{'s' if len(names) != 1 else ''} "
                f"from `config/{fname}`:")
    st.markdown("\n".join(f"- `{n}`" for n in names))
    st.caption(f"Results you already ran are kept. The previous file is saved as config/{fname}.bak, so you can undo one removal.")
    c1, c2 = st.columns(2)
    if c1.button("Remove", type="primary", icon=":material/delete:", width="stretch"):
        gone = config.delete_primers(names) if kind == "pcr" else qpcr.delete_assays(names)
        st.session_state["lib_msg"] = f"Removed {len(gone)} from config/{fname}: {', '.join(gone)}"
        st.session_state.pop(f"lib_ed_{kind}", None)
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


def library_table(kind: str, lib: dict, cols: list[str]):
    if not lib:
        st.info("Nothing saved yet." if kind == "pcr" else "No qPCR assays saved yet. Save one from step 4 or step 6 in qPCR mode.")
        return
    df = pd.DataFrame([{"name": k, **{c: v.get(c, "") for c in cols}} for k, v in lib.items()])
    c1, c2 = st.columns([1, 2])
    mk = sorted({str(m) for m in df["marker"] if str(m)})
    pick = c1.multiselect("Marker", mk, key=f"lib_mk_{kind}", placeholder="All markers")
    q = c2.text_input("Search", key=f"lib_q_{kind}", placeholder="Name, sequence, target or reference")
    view = df
    if pick:
        view = view[view["marker"].astype(str).isin(pick)]
    if q:
        hit = view.astype(str).apply(lambda r: r.str.contains(q, case=False, regex=False)).any(axis=1)
        view = view[hit]
    st.caption(f"Showing {len(view)} of {len(df)}. Tick rows to remove them.")
    view = view.reset_index(drop=True)
    view.insert(0, "remove", False)
    ed = st.data_editor(view, key=f"lib_ed_{kind}", hide_index=True, width="stretch",
                        disabled=[c for c in view.columns if c != "remove"],
                        column_config={"remove": st.column_config.CheckboxColumn("Remove", width=80),
                                       "name": st.column_config.TextColumn("Name", pinned=True)})
    chosen = ed.loc[ed["remove"], "name"].tolist()
    b1, b2 = st.columns([1, 2.2], vertical_alignment="center")
    if b1.button(f"Remove ({len(chosen)})", disabled=not chosen, icon=":material/delete:", key=f"lib_rm_{kind}", width="stretch"):
        confirm_remove(kind, chosen)
    with b2:
        download_csv(df, f"{'primer_pairs' if kind == 'pcr' else 'qpcr_assays'}.csv", key=f"lib_dl_{kind}")


def page_library():
    from ispcr import qpcr
    pcr_lib, q_lib = config.primers(), qpcr.library()
    st.markdown("<div class='ph'><div><div class='eyebrow'>Library</div><h1>Primer library</h1>"
                "<div class='sub'>Every primer pair and qPCR assay the app knows about. Remove the ones you no longer need; "
                "they disappear from the pickers in steps 4, 8 and 9.</div></div></div>", unsafe_allow_html=True)
    if msg := st.session_state.pop("lib_msg", None):
        st.success(msg, icon=":material/check:")
    markers = {str(v.get("marker", "")) for v in [*pcr_lib.values(), *q_lib.values()] if v.get("marker")}
    m1, m2, m3 = st.columns(3)
    m1.metric("PCR primer pairs", len(pcr_lib))
    m2.metric("qPCR assays", len(q_lib))
    m3.metric("Markers covered", len(markers))
    t1, t2 = st.tabs([f":material/biotech: PCR primer pairs ({len(pcr_lib)})", f":material/my_location: qPCR assays ({len(q_lib)})"])
    with t1:
        library_table("pcr", pcr_lib, ["marker", "forward", "reverse", "product_min", "product_max", "reference"])
    with t2:
        library_table("qpcr", q_lib, ["target", "marker", "forward", "reverse", "probe", "probe_type", "reference"])
    st.caption(f"Stored in `{paths.config_dir()}` (primers.yaml and qpcr_assays.yaml). New ones are added with "
               "“Save to library” in steps 4 and 6.")


# ---------------------------------------------------------------- navigation
STEPS = [
    (page_species, "Species & download", ":material/travel_explore:", "species"),
    (page_sequences, "Sequences & QC", ":material/fact_check:", "sequences"),
    (page_alignment, "Alignment & tree", ":material/view_week:", "alignment"),
    (page_pcr, "In silico PCR", ":material/biotech:", "pcr"),
    (page_qpcr, "In silico qPCR", ":material/my_location:", "qpcr"),
    (page_resolution, "Target vs others" if MODE == "qpcr" else "Resolution", ":material/scatter_plot:", "resolution"),
    (page_design, "Assay design" if MODE == "qpcr" else "Primer design", ":material/design_services:", "design"),
    (page_compare, "Compare assays" if MODE == "qpcr" else "Compare & export", ":material/compare_arrows:", "compare"),
    (page_mitomap, "Mitogenome map", ":material/donut_large:", "mitogenome-map"),
    (page_report, "PDF report", ":material/picture_as_pdf:", "report"),
]


# ---------------------------------------------------------------- sidebar: project files and data folder
def project_box():
    """Export everything of a dataset to one .zip, or open such a file (here or on another computer)."""
    import zipfile
    from ispcr import bundle
    st.markdown("**Save project**")
    st.caption("One .zip with everything: species list, downloads, sequences, alignments, exclusions, trees, "
               "every run, reports and the choices made in the app.")
    all_ds = paths.list_datasets()
    scope = st.radio("What", ["This dataset", "All datasets"], horizontal=True, disabled=not all_ds, label_visibility="collapsed")
    with_dl = st.toggle("Include GenBank downloads", value=True,
                        help="Without them the file is smaller; the app can download them again (needs internet).")
    if st.button("Create project file", icon=":material/archive:", disabled=not all_ds, width="stretch"):
        with st.spinner("Packing..."):
            st.session_state["proj_file"] = str(bundle.export_project(all_ds if scope == "All datasets" else ds, downloads=with_dl))
    f = st.session_state.get("proj_file")
    if f and Path(f).exists():
        st.download_button(f"Download {Path(f).name} ({Path(f).stat().st_size / 1e6:.1f} MB)", Path(f).read_bytes(),
                           file_name=Path(f).name, mime="application/zip", icon=":material/download:", type="primary", width="stretch")
        if not paths.is_hosted():
            st.caption("Also saved in results/exports/ of the data folder.")
    st.divider()
    st.markdown("**Open project**")
    up = st.file_uploader("Project file (.zip)", type=["zip"], key="proj_up",
                          help="A project file, or an older downloads bundle.")
    if not up:
        return
    try:
        man = bundle.read_manifest(up.getvalue())
    except zipfile.BadZipFile:
        st.error("This is not a .zip file.")
        return
    if man:
        st.caption(f"Saved {man.get('created')} · {', '.join(man.get('datasets', []))} · {len(man.get('files', {}))} files"
                   + ("" if man.get("downloads", True) else " · without downloads"))
        here = [d for d in man.get("datasets", []) if d in all_ds]
        if here:
            st.warning(f"**{', '.join(here)}** is already here and will be replaced by the version in the file. "
                       "What is here now is saved first to results/exports/.", icon=":material/warning:")
    else:
        st.caption("Downloads bundle: name check and GenBank files only.")
    if st.button("Open project", type="primary", icon=":material/unarchive:", width="stretch"):
        try:
            res = bundle.import_any(up.getvalue())
        except ValueError as e:
            st.error(str(e))
            return
        st.cache_data.clear()
        msg = f"Opened {', '.join(res['datasets']) or 'bundle'} ({res['files']} files)"
        if res.get("library_added"):
            msg += f" · added to library: {', '.join(res['library_added'])}"
        if res.get("corrupt"):
            msg += f" · {len(res['corrupt'])} damaged files were skipped"
        st.session_state["import_msg"] = msg
        if res["datasets"]:
            st.session_state["ds_pending"] = res["datasets"][0]
            for k in ("marker", "mode_ds", "mode_ctl"):  # reload marker and mode from the imported choices
                st.session_state.pop(k, None)
            st.session_state["restore_page"] = res.get("kind") == "project"
        st.rerun()


def data_folder_box():
    """Where your data and results are kept on this computer. The code (git repository) stays separate."""
    st.caption("Your species lists, downloads, results and primer library live here, separate from the app's code. "
               "Saved in ispcr.local.yaml (not in git).")
    if paths.data_root_source() == "env":
        st.info("Set by the environment variable ISPCR_DATA, so it can't be changed here.")
        return
    new = st.text_input("Folder", str(paths.ROOT), key="data_dir_in")
    p = Path(new).expanduser()
    if p.exists() and not p.is_dir():
        st.error("That is a file, not a folder.")
        return
    has = (p / "data" / "taxonomy").exists()
    st.caption(("Contains ispcr data." if has else "No ispcr data there yet: it starts empty. "
                "To bring your work, save all datasets as a project first and open it after switching.") if p != paths.ROOT else "")
    if st.button("Use this folder", type="primary", disabled=p.resolve() == paths.ROOT, width="stretch"):
        paths.set_root(p)
        st.cache_data.clear()
        for k in ("ds", "marker", "mode_ds", "mode_ctl", "proj_file"):
            st.session_state.pop(k, None)
        st.rerun()


def guarded(fn):
    def run():
        if ds == "—" and fn is not page_species:
            header(STEPS.index(next(s for s in STEPS if s[0] is fn)) + 1, "No dataset yet")
            st.info("Start with step 1: resolve a species list to create a dataset.")
        else:
            fn()
    run.__name__ = fn.__name__
    return run


pages = [st.Page(guarded(f), title=t, icon=i, url_path=u, default=(u == "species")) for f, t, i, u in STEPS]
lib_page = st.Page(page_library, title="Primer library", icon=":material/library_books:", url_path="primer-library")
nav = st.navigation(pages + [lib_page], position="hidden")
done = step_done()
with st.sidebar:
    st.markdown("<div class='sec'>Workflow</div>", unsafe_allow_html=True)
    off = "pcr" if MODE == "qpcr" else "qpcr"
    n = 0
    for pg, ok, (_, t, icon, u) in zip(pages, done, STEPS):
        if u == off:  # the other mode's step is hidden (still reachable by its URL)
            continue
        n += 1
        st.page_link(pg, label=f"{n}  {t}", icon=":material/check_circle:" if ok else icon)
    n_done = sum(ok for ok, (_, _, _, u) in zip(done, STEPS) if u != off)
    st.markdown(f"<div class='prog'><span style='width:{100 * n_done / 9:.0f}%'></span></div>"
                f"<div class='note'>{n_done} of 9 steps done for {marker} · mode: {'in silico qPCR' if MODE == 'qpcr' else 'in silico PCR'}</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sec'>Library</div>", unsafe_allow_html=True)
    from ispcr import qpcr as _q
    st.page_link(lib_page, label=f"Primer library · {len(config.primers()) + len(_q.library())}", icon=":material/library_books:")
    st.markdown("<div class='sec'>Status</div>", unsafe_allow_html=True)
    on = is_online()
    st.markdown(f"<span class='net'><i style='background:{'#2c7a51' if on else '#b5761a'}'></i>"
                f"{'Online · NCBI reachable' if on else 'Offline · using downloaded files'}</span>", unsafe_allow_html=True)
    if not cfg["ncbi"].get("email"):
        st.caption(":material/mail: Add your NCBI email before downloading: "
                   + ("NCBI_EMAIL in the app's secrets." if paths.is_hosted() else "config/settings.local.yaml (ncbi.email) or NCBI_EMAIL."))
    st.markdown("<div class='sec'>Data</div>", unsafe_allow_html=True)
    with st.popover("Save / open project", icon=":material/folder_zip:", width="stretch"):
        project_box()
    if not paths.is_hosted():
        with st.popover("Data folder", icon=":material/folder_open:", width="stretch"):
            data_folder_box()
    st.caption(f"Data folder: `{paths.ROOT}`" if not paths.is_hosted() else
               "Online version: files here are temporary. Save a project file before you leave.")

# after an import: open the page that was open when the project was saved
if ds != "—" and "test_page" not in st.session_state:
    if st.session_state.pop("restore_page", False):
        saved = SEL.get(ds, "page")
        target = next((pg for pg in pages + [lib_page] if pg.title == saved), None)
        if target is not None and saved != nav.title:
            st.switch_page(target)
    else:
        SEL.put(ds, "page", nav.title)
if "import_msg" in st.session_state:
    st.toast(st.session_state.pop("import_msg"), icon=":material/check_circle:")

if "test_page" in st.session_state:  # used by automated tests
    guarded(STEPS[st.session_state["test_page"]][0])()
else:
    nav.run()
