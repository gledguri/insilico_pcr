# ispcr: in silico PCR and primer design for fish mitochondrial markers

The **code** (this repository) and **your data** (species lists, downloads, results, your primer library)
are kept apart: the code goes on GitHub, the data stays on your computer.

## Install and start (your computer)

```bash
git clone <this repository> ispcr && cd ispcr
conda env create -p ./env -f environment.yml    # Python 3.12 + MAFFT + packages
./run_app.sh                                     # opens the app at http://localhost:8501
```

Without conda: install MAFFT (`brew install mafft` / `apt install mafft`), then
`pip install -r requirements.txt && pip install -e .` and `./run_app.sh`.

**NCBI email and key.** NCBI asks for a contact email; an API key is optional (~3x faster downloads).
Put them in `config/settings.local.yaml` (not in git), or set `NCBI_EMAIL` / `NCBI_API_KEY`:

```yaml
ncbi:
  email: "you@example.org"
  api_key: "..."
```

**Data folder.** By default your data goes to `workspace/` inside the repository (ignored by git).
To keep it elsewhere, use **Data folder** in the app's sidebar (saved in `ispcr.local.yaml`, not in git),
or set the environment variable `ISPCR_DATA=/path/to/folder`.

## Save your work and pick it up later (project files)

Sidebar → **Save / open project**:

- **Save project** makes one `.zip` with everything of a dataset (or all datasets): species list, name check,
  GenBank downloads, sequences, QC, alignments, excluded records, trees, every PCR / qPCR / design / resolution
  run, reports, the choices made in each step (runs shown, filters, view options, the page you were on) and your
  primer library. Turn off *Include GenBank downloads* for a smaller file.
- **Open project** restores it, here or on another computer or on the online version, and opens the app on the
  same dataset, marker, mode and page. If that dataset is already there, it is replaced; what was there is first
  saved to `results/exports/<dataset>_before_import_<date>.zip`. Primer pairs and assays you don't have yet are
  added to your library (yours with the same name are kept). Files are checked against checksums.

From Python: `ispcr.bundle.export_project("my_dataset")` and `ispcr.bundle.import_project(Path("file.zip"))`.

## Online version (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On https://share.streamlit.io: *Create app* → this repository, branch `main`, file `app/streamlit_app.py`,
   Python 3.12 (advanced settings).
3. In the app's *Secrets*, add:
   ```toml
   NCBI_EMAIL = "you@example.org"
   NCBI_API_KEY = "..."
   ```
`requirements.txt` installs the Python packages and `packages.txt` installs MAFFT.

The online app cannot see files on your computer. Bring your work in with **Open project**, and take it home with
**Save project** before you leave: the server's files are temporary (wiped when the app restarts) and are shared
by everyone using the same app.

## Folder layout

Repository (code, in git):

```
ispcr/
├── app/streamlit_app.py       the user interface
├── src/ispcr/                 the Python package (importable: `from ispcr import pcr`)
├── config/
│   ├── settings.yaml          default settings: marker definitions, QC, PCR and qPCR rules, design settings
│   ├── settings.local.yaml    your NCBI email/key (not in git)
│   ├── primers.yaml           published primer pairs (copied into your data folder on first start)
│   └── qpcr_assays.yaml       empty template for species-specific assays
├── requirements.txt, packages.txt, environment.yml   installation
├── tests/
└── workspace/                 default data folder (not in git)
```

Data folder (`workspace/` or the folder you chose):

```
<data folder>/
├── config/
│   ├── primers.yaml           your primer library (published pairs + pairs you save)
│   ├── qpcr_assays.yaml       your species-specific qPCR assays (primers + probe + target species)
│   └── settings.yaml          optional: overrides of the default settings
├── data/
│   ├── species_lists/         INPUT: your lists, one Latin name per line (# = comment)
│   ├── taxonomy/<dataset>/    species_resolved.csv  (WoRMS + NCBI name check)
│   ├── genbank/<target>/      raw GenBank downloads, cached per species (.gb + .json)
│   │     mitogenome/  12S/  16S/  COI/  CYTB/
│   ├── sequences/<dataset>/<marker>/   sequences.fasta + metadata.csv (gene cut out of records)
│   ├── qc/<dataset>/<marker>/          clean.fasta, clean_metadata.csv, removed.csv,
│   │                                   species_counts.csv, qc_summary.json
│   └── alignments/<dataset>/<marker>/  aligned.fasta, aligned_metadata.csv, columns.csv
│                                       (entropy per column), misid_flags.csv
├── results/
│   ├── pcr/<dataset>/<marker>/<primer>/        hits.csv (one row per record), species_summary.csv,
│   │                                           mismatch_matrix_forward/reverse.csv, amplicons*.fasta,
│   │                                           summary.json
│   ├── resolution/<dataset>/<marker>/<primer>/ barcode_gap.csv, intra_distances.csv,
│   │                                           distance_histogram.csv, species_min_distance.csv,
│   │                                           amplicons_aligned.fasta, tree_nj.nwk, summary.json
│   ├── design/<dataset>/<marker>/<run_id>/     conservation.csv, forward/reverse_candidates.csv,
│   │                                           pairs.csv, evaluated_pairs.csv, params.json
│   ├── qpcr/<dataset>/<marker>/<assay>/        species-specific assay check: hits.csv (every record),
│   │                                           species_summary.csv, checks.csv, target_amplicons.fasta, summary.json
│   ├── qpcr_design/<dataset>/<marker>/<target>_<time>/   assays.csv, params.json
│   └── reports/<dataset>/                      selections.json (what you chose in the app)
│       └── <marker>/                           report_pcr|qpcr_<time>.pdf (+ .json of its choices),
│                                               report_<time>.html, primer_comparison.csv
│   └── exports/                                project files (.zip) you saved
└── logs/                      one log per command and day
```

A *dataset* is a resolved species list (its file name without extension). A *target* is what is
downloaded (`mitogenome` or a marker). A *marker* is the region analysed (`12S`, `16S`, `COI`, `CYTB`,
or `mitogenome` for whole genomes). Add new markers in `settings.yaml`.

## Steps

| Step | Command | What it does |
|---|---|---|
| Names | `ispcr resolve <list>` | WoRMS match (accepted names, synonyms, fuzzy matches) + NCBI taxid and lineage |
| Download | `ispcr fetch <dataset> --target mitogenome\|12S\|...` | NCBI Nucleotide; RefSeq mitogenomes first; cached |
| Extract | `ispcr extract <dataset> --marker 12S` | cuts the gene out of gene records **and** mitogenomes (GenBank features) |
| QC | `ispcr qc <dataset> --marker 12S` | length limits, ambiguous bases, duplicates, nuclear/predicted records dropped |
| Align | `ispcr align <dataset> --marker 12S` | MAFFT; orientation fixed; entropy per column; flags likely misidentified records |
| PCR | `ispcr pcr <dataset> --marker 12S --primers MiFish-U` or `--fwd .. --rev ..` | binding sites with mismatches on both strands, 3′-end rule, product length, primer3 binding Tm/ΔG |
| Resolution | `ispcr resolution <dataset> --marker 12S --primers MiFish-U` | K2P/p distances on amplicons (primers removed), barcode gap, identical amplicons, NJ tree |
| Design | `ispcr design <dataset> --marker 12S` | conserved sites → degenerate candidates → pairs scored by coverage and resolution → top pairs checked with the full PCR |
| qPCR | `ispcr qpcr <dataset> --marker COI --target "Gadus morhua" --fwd .. --rev .. --probe ..` | species-specific assay: target detected in every record, every other species blocked |
| qPCR design | `ispcr qpcr-design <dataset> --marker COI --target "Gadus morhua"` | primers + probe identical in the target, mismatched (at the 3′ ends) in all other species |
| Report | `ispcr report <dataset> --marker 12S` | HTML report comparing all primer pairs |
| PDF report | `ispcr pdf-report <dataset> --marker 12S` | every step in one PDF, using the choices saved in the app |

## In silico PCR or in silico qPCR (steps 4 and 5)

Steps 4 and 5 each have a switch; only one can be on. With **in silico PCR** (metabarcoding: amplify as many
species as possible and tell them apart), steps 6–8 show resolution, primer design and the comparison of PCR runs.
With **in silico qPCR** (detect one species and nothing else), they show the target-vs-other-species distances, the
design of species-specific assays, and the comparison of qPCR assays. The mitogenome map and the PDF report follow
the switch too.

What is checked for a species-specific assay (rules in `settings.yaml` → `qpcr`):

- **every record of the target is detected**: at most 2 mismatches per primer, none in the last 5 bases at the 3′ end,
  at most 1 in the probe (best: 0 everywhere, shown as "perfect");
- **every other species is blocked**: ≥ 2 mismatches in the 3′ window of one primer, or ≥ 4 in one primer, or ≥ 5 in
  both primers together, or ≥ 3 in the probe. Species in between are **at risk** (late, weak amplification possible);
- **oligo chemistry** (MIQE / TaqMan rules): primers 18–25 nt, Tm 57–63 °C and within 2 °C, GC 35–65 %, 1–3 G/C in
  the last 5 bases, no runs of 4+, no strong hairpins/dimers; probe Tm 5–10 °C above the primers, no 5′ G, C ≥ G;
  amplicon 50–150 bp;
- **amplicon distance to the target** (step 6): species with an amplicon identical to the target rely on primer and
  probe mismatches only.

Records are read at the oligos' alignment columns, so a record that ends before a site is "not covered" rather than
a failure. Species that are not in your list (or not in GenBank) are not checked: run Primer-BLAST on the final assay
and test tissue DNA of close relatives. References: Bustin et al. 2009 (MIQE), Kwok et al. 1990, Stadhouders et al.
2010, Wilcox et al. 2013, Thalinger et al. 2021.

## PDF report (step 10)

Every choice in the app is saved per dataset in `results/reports/<dataset>/selections.json`: the marker, the switch,
the species and primer sites shown in the alignment, the tree, the PCR run or qPCR check you looked at, the
resolution run, the design run, the pairs or assays compared, and the mitogenome (species, record, pairs). Step 10
reads them, lets you change them, and writes one PDF with: name check, sequences & QC, alignment (one record per
species with the F/R (and probe) sites and the bases of every species at each site), in silico PCR (species amplified
and F/R mismatches) or in silico qPCR, resolution, the top 10 or 20 designed pairs/assays, the comparison (species
amplified first, then species resolved), and the mitogenome map.

## Working offline and reusing downloads

Downloads are cached in `data/genbank/` and never fetched twice; a species list that was already checked
is reused without internet. When NCBI cannot be reached, the app switches to the downloaded files
(sidebar shows *Offline*). To move downloads to another computer, or keep a backup:

```bash
ispcr export-downloads norway_marine_fish_100     # -> results/exports/<dataset>_downloads_<date>.zip
ispcr import-downloads path/to/bundle.zip
ispcr fetch norway_marine_fish_100 --target 12S --offline
```

You can also add your own GenBank or FASTA files (step 1 → *Reuse downloads*); each record is matched to a
species of the dataset by its organism or header. FASTA files go to `data/imported/<target>/`.

## Excluding records and trees

Misannotated records can be left out from step 3 onward (alignment page → *Exclude records*, or
`ispcr exclude <dataset> --marker 12S ACC1 ACC2 --reason "..."`, `--flagged` for all QC flags, `--undo`
to put them back). They are listed in `data/alignments/<dataset>/<marker>/excluded.csv` and skipped by the
tree, PCR, resolution and design. Runs made before a change are marked as out of date in the app,
with a button to re-run them.

Trees: alignment page → *Tree*, or `ispcr tree <dataset> --marker 12S --mode species --bootstrap 100`
(`--columns 200-420` for one region). Neighbour joining on K2P distances, midpoint-rooted, saved as Newick
in `data/alignments/<dataset>/<marker>/trees/`. With all records, species that do not form one group are
listed with the records that sit outside it.

## Mitogenome map (step 8)

An interactive circular map of any downloaded mitogenome: genes on the + strand (outer ring) and − strand
(inner ring), coloured by complex, tRNAs as one-letter codes, GC content, and arcs showing where each primer
pair from the library amplifies (✓ / ✗ by the PCR rules). A linear view with zoom, a gene table with FASTA
download per gene, and an *Across species* tab comparing genome length, GC and gene lengths/gene order of all
downloaded mitogenomes. Gene names are standardised (e.g. COX1 → COI, s-rRNA → 12S).

## How PCR success is decided

A record is **amplified** when both primers bind with at most `max_mismatches` mismatches each,
at most `max_3prime_mismatches` in the last `three_prime_window` bases at the 3′ end, and the product
length is in the pair's expected range. Records that stop before a primer site are reported as
**site_not_covered** (decided from the alignment) and are not counted as failures; a species with only
such records is **not testable** rather than failing. Mismatches are written as `position primer>template`
(e.g. `9A>G`, position counted from the 5′ end). Indels in the binding site are not modelled.

Change the rules in `settings.yaml` (defaults) or per run in the app.
