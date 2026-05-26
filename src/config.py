from pathlib import Path

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
