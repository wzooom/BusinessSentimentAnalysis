from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.api_access.config import PROCESSED_DATA_DIR, ensure_data_dirs

BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "then", "there", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "you", "your", "not", "do", "does", "did",
    "can", "could", "would", "should", "just", "like", "really", "very", "much", "more",
    "most", "get", "got", "one", "two", "also", "still", "even", "think", "people",
    "thing", "things", "actually", "basically", "literally", "op", "tldr", "imo", "imho",
    "afaik", "edit", "update", "deleted", "removed", "stock", "stocks", "share", "shares",
    "market", "company",
}

COMPANY_STOPWORDS = {
    "tesla", "tsla", "elon", "musk",
    "starbucks", "sbux",
    "chipotle", "cmg",
}


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_bert_text(text: str) -> str:
    """Light cleaning only. Natural long-text column for BERT/RoBERTa before softmax."""
    text = "" if pd.isna(text) else str(text)
    text = strip_urls(text)
    text = re.sub(r"\bu/[A-Za-z0-9_-]+", "@user", text)
    text = re.sub(r"\br/[A-Za-z0-9_-]+", "r/sub", text)
    return normalize_spacing(text)


def make_lda_text(text: str, extra_stopwords: Iterable[str] | None = None) -> str:
    """Aggressive cleaning for LDA/topic modeling."""
    text = "" if pd.isna(text) else str(text)
    text = strip_urls(text)
    text = text.lower()
    text = re.sub(r"\$[a-z]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()
    stopwords = set(BASIC_STOPWORDS) | set(COMPANY_STOPWORDS)
    if extra_stopwords:
        stopwords |= {w.lower() for w in extra_stopwords}

    tokens = [tok for tok in tokens if len(tok) >= 3 and tok not in stopwords]
    return " ".join(tokens)


def preprocess_stocktwits_file(raw_path: str | Path, company: str) -> str | None:
    ensure_data_dirs()
    raw_path = Path(raw_path)

    try:
        df = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {raw_path}")
        return None

    if df.empty:
        print(f"Skipping file with 0 rows: {raw_path}")
        return None

    if "text" not in df.columns:
        raise ValueError(f"Expected a text column in {raw_path}")

    df["text"] = df["text"].fillna("").astype(str)
    df = df[~df["text"].str.lower().isin(["", "nan", "[deleted]", "[removed]"])]
    df = df.drop_duplicates(subset=["text"])

    df["bert_text"] = df["text"].apply(make_bert_text)
    df["lda_text"] = df["text"].apply(make_lda_text)

    token_count = df["lda_text"].fillna("").str.split().str.len()
    df = df[token_count >= 3]

    if df.empty:
        print(f"No useful rows after preprocessing: {raw_path}")
        return None

    if "sentiment_label" not in df.columns:
        if "stocktwits_sentiment_label" in df.columns:
            df["sentiment_label"] = df["stocktwits_sentiment_label"].fillna("").astype(str)
        else:
            df["sentiment_label"] = ""

    if "created_datetime" not in df.columns:
        df["created_datetime"] = ""

    if "ticker" not in df.columns:
        df["ticker"] = ""

    processed = pd.DataFrame({
        "company": company,
        "source": "stocktwits",
        "ticker": df["ticker"].fillna("").astype(str),
        "created_datetime": df["created_datetime"].fillna("").astype(str),
        "sentiment_label": df["sentiment_label"].fillna("").astype(str),
        "bert_text": df["bert_text"].fillna("").astype(str),
        "lda_text": df["lda_text"].fillna("").astype(str),
    })

    output_name = raw_path.name.replace(".csv", "_clean.csv")
    output_path = PROCESSED_DATA_DIR / output_name
    processed.to_csv(output_path, index=False)
    print(f"Saved {len(processed)} cleaned rows to {output_path}")
    return str(output_path)


def combine_clean_files(output_name: str = "stocktwits_all_messages_clean.csv") -> str | None:
    files = sorted(PROCESSED_DATA_DIR.glob("stocktwits_*_messages_clean.csv"))
    if not files:
        print("No clean StockTwits files found to combine.")
        return None

    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        print("No readable clean StockTwits files found.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["company", "ticker", "bert_text"])
    output_path = PROCESSED_DATA_DIR / output_name
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} combined clean StockTwits rows to {output_path}")
    return str(output_path)
