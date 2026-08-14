"""Central config loader.

Every script pulls paths / seeds / hyperparameter ranges from
`config/config.yaml` through this module instead of hardcoding values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(*parts: str) -> Path:
    """Resolve a relative project path (e.g. resolve('data','raw','x.csv'))."""
    return PROJECT_ROOT.joinpath(*parts)
