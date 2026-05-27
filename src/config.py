from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPANIES = ["chipotle", "starbucks", "tesla"]

EMOTION_MODEL = "SamLowe/roberta-base-go_emotions-onnx"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCORED_DIR = DATA_DIR / "scored"
TOPICS_DIR = DATA_DIR / "topics"
AGGREGATED_DIR = DATA_DIR / "aggregated"
BRIEFS_DIR = DATA_DIR / "briefs"
REPORTS_DIR = PROJECT_ROOT / "reports"

_ALL_DATA_DIRS = [RAW_DIR, PROCESSED_DIR, SCORED_DIR, TOPICS_DIR,
                  AGGREGATED_DIR, BRIEFS_DIR, REPORTS_DIR]


def ensure_data_dirs() -> None:
    """Create all data output folders if they do not already exist."""
    for directory in _ALL_DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable with an optional default."""
    return os.getenv(name, default)
