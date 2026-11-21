"""Load settings and the primer library.

Settings are merged in this order (later wins):
  <repo>/config/settings.yaml          defaults (in git)
  <repo>/config/settings.local.yaml    private values for this computer, e.g. NCBI email and key (not in git)
  <data folder>/config/settings.yaml   overrides that travel with your data
  environment: NCBI_EMAIL, NCBI_API_KEY (on Streamlit Cloud, put them in the app's secrets)
The primer library (primers.yaml, qpcr_assays.yaml) lives in <data folder>/config/.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

from . import paths


def _merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _load(p) -> dict:
    if not p.exists():
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    cfg = _load(paths.default_config_dir() / "settings.yaml")
    _merge(cfg, _load(paths.default_config_dir() / "settings.local.yaml"))
    _merge(cfg, _load(paths.ROOT / "config" / "settings.yaml"))
    ncbi = cfg.setdefault("ncbi", {})
    ncbi["email"] = os.environ.get("NCBI_EMAIL") or ncbi.get("email") or ""
    ncbi["api_key"] = os.environ.get("NCBI_API_KEY") or ncbi.get("api_key") or ""
    return cfg


def marker_cfg(marker: str) -> dict[str, Any]:
    markers = settings()["markers"]
    for k, v in markers.items():
        if k.lower() == marker.lower():
            return v
    raise KeyError(f"Marker '{marker}' is not defined in settings.yaml (known: {', '.join(markers)})")


def primers() -> dict[str, dict[str, Any]]:
    p = paths.config_dir() / "primers.yaml"
    if not p.exists():
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


def save_primer(name: str, entry: dict[str, Any]) -> None:
    lib = primers()
    lib[name] = entry
    with open(paths.config_dir() / "primers.yaml", "w") as fh:
        yaml.safe_dump(lib, fh, sort_keys=False)


def delete_primers(names: list[str]) -> list[str]:
    """Remove primer pairs from primers.yaml. A copy of the old file is kept as primers.yaml.bak."""
    p = paths.config_dir() / "primers.yaml"
    lib = primers()
    gone = [n for n in names if n in lib]
    if gone:
        p.with_suffix(".yaml.bak").write_text(p.read_text())
        for n in gone:
            del lib[n]
        with open(p, "w") as fh:
            yaml.safe_dump(lib, fh, sort_keys=False)
    return gone
