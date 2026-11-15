"""Shared configuration loader.

All seeds, paths and thresholds are read from ``config.yaml`` at the repository root.
"""
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load config.yaml once and return it as a dictionary."""
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str) -> Path:
    """Return an absolute path from the ``paths`` section of the config."""
    return PROJECT_ROOT / load_config()["paths"][key]
