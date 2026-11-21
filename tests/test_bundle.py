"""Project export -> import into an empty data folder gives back exactly the same files."""
import zipfile

from ispcr import bundle, paths, selections


def _make_dataset(root, ds="demo"):
    paths.set_root(root, remember=False)
    (paths.taxonomy_dir(ds) / "species_resolved.csv").write_text("input_name,accepted_name\nGadus morhua,Gadus morhua\n")
    (paths.species_lists_dir() / f"{ds}.txt").write_text("Gadus morhua\n")
    (paths.genbank_dir("12S") / "Gadus_morhua.gb").write_text("LOCUS x\n//\n")
    (paths.genbank_dir("12S") / "Other_fish.gb").write_text("LOCUS y\n//\n")  # not in this dataset
    (paths.qc_dir(ds, "12S") / "clean.fasta").write_text(">a\nACGT\n")
    (paths.trees_dir(ds, "12S") / "t.nwk").write_text("(a,b);")
    (paths.pcr_dir(ds, "12S", "MiFish-U") / "hits.csv").write_text("id,status\na,amplified\n")
    selections.put(ds, "mode", "qpcr")
    selections.put(ds, "pcr_run", "MiFish-U", marker="12S")
    (paths.config_dir() / "qpcr_assays.yaml").write_text("My_assay:\n  target: Gadus morhua\n")


def _snapshot(root):
    skip = ("results/exports", "logs", "config")
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and not p.relative_to(root).as_posix().startswith(skip)}


def test_round_trip(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    _make_dataset(src)
    before = _snapshot(src)
    z = bundle.export_project("demo")
    names = set(zipfile.ZipFile(z).namelist())
    assert "data/genbank/12S/Other_fish.gb" not in names and bundle.MANIFEST in names

    paths.set_root(dst, remember=False)
    res = bundle.import_any(z.read_bytes())
    assert res["kind"] == "project" and res["datasets"] == ["demo"] and not res["corrupt"]
    after = _snapshot(dst)
    expected = {k: v for k, v in before.items() if "Other_fish" not in k}
    assert after == expected
    assert selections.mode("demo") == "qpcr" and selections.get("demo", "pcr_run", "12S") == "MiFish-U"
    assert "My_assay" in res["library_added"]


def test_import_replaces_and_backs_up(tmp_path):
    _make_dataset(tmp_path)
    z = bundle.export_project("demo", downloads=False)
    (paths.pcr_dir("demo", "12S", "Later_run") / "hits.csv").write_text("x\n")  # made after the export
    res = bundle.import_project(z.read_bytes())
    assert not (paths.ROOT / "results/pcr/demo/12S/Later_run").exists()
    assert res["backups"] and "results/pcr/demo/12S/Later_run/hits.csv" in zipfile.ZipFile(res["backups"][0]).namelist()


def test_corrupt_file_is_refused(tmp_path):
    _make_dataset(tmp_path)
    z = bundle.export_project("demo")
    zin = zipfile.ZipFile(z)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, b"tampered" if n.endswith("clean.fasta") else zin.read(n))
    res = bundle.import_project(bad.read_bytes())
    assert any(n.endswith("clean.fasta") for n in res["corrupt"])
