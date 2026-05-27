"""Step 2: Text preprocessing for Reddit and StockTwits data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from api_access.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_data_dirs

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your", "about", "after", "all", "also", "can", "do", "does",
    "just", "like", "more", "not", "now", "out", "over", "than", "up", "would",
}


def combine_title_and_text(row: pd.Series) -> str:
    """Combine title and body text into one field."""
    title = str(row.get("title", "") or "")
    text = str(row.get("text", "") or "")
    return f"{title} {text}".strip()


def clean_readable_text(text: str) -> str:
    """Create readable cleaned text for sentiment and quote extraction."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"u/\w+|r/\w+", " ", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\$([A-Za-z]+)", r"\1", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_lda_text(text: str) -> str:
    """Create a more filtered version of text for LDA topic modeling."""
    text = clean_readable_text(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    return " ".join(tokens)


def preprocess_file(
    input_path: str | Path,
    output_name: str | None = None,
    min_tokens: int = 5,
) -> str:
    """Clean a raw CSV file and save it to data/processed."""
    ensure_data_dirs()
    input_path = Path(input_path)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input file is empty: {input_path}")

    df["combined_text"] = df.apply(combine_title_and_text, axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_readable_text)
    df["lda_text"] = df["combined_text"].apply(make_lda_text)
    df["token_count"] = df["lda_text"].apply(lambda x: len(str(x).split()))

    # Remove duplicates and low-effort rows.
    df = df.drop_duplicates(subset=["clean_text"])
    df = df[df["token_count"] >= min_tokens]

    if output_name is None:
        output_name = input_path.stem + "_clean.csv"

    output_path = PROCESSED_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)


def preprocess_default_files() -> list[str]:
    """Preprocess all CSV files currently in data/raw."""
    ensure_data_dirs()
    outputs = []
    for path in RAW_DATA_DIR.glob("*.csv"):
        outputs.append(preprocess_file(path))
    return outputs
