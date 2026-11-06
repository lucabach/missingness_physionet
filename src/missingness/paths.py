"""Single source of truth for where things live on disk.

Every path can be overridden with an environment variable, so the same code runs on a
laptop and on the cluster (e.g. ``export MISSINGNESS_DATA_DIR=/cluster/scratch/$USER/data``).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("MISSINGNESS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path(os.environ.get("MISSINGNESS_OUTPUT_DIR", REPO_ROOT / "outputs"))
SPLITS_DIR = REPO_ROOT / "splits"  # committed: everyone uses the same patient splits
CONFIG_DIR = REPO_ROOT / "configs"
