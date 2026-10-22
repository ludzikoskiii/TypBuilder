"""Wczytywanie konfiguracji z config.yaml."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for section, key in (("data", "cache_dir"), ("journal", "db_path")):
        value = cfg.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(ROOT / value)
    return cfg


def load_config(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    """Zwróć kopię konfiguracji; ścieżki względne są rozwiązywane względem katalogu projektu."""
    return copy.deepcopy(_load(str(path)))
