from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, ensure_data_dirs

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
    title = str(row.get("title", "") or "")
    text = str(row.get("text", "") or "")
    return f"{title} {text}".strip()


def clean_readable_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"u/\w+|r/\w+", " ", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\$([A-Za-z]+)", r"\1", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_lda_text(text: str) -> str:
    text = clean_readable_text(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    return " ".join(tokens)


def preprocess_file(
    input_path: str | Path,
    output_name: str | None = None,
    min_tokens: int = 5,
) -> str:
    ensure_data_dirs()
    input_path = Path(input_path)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input file is empty: {input_path}")

    df["combined_text"] = df.apply(combine_title_and_text, axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_readable_text)
    df["lda_text"] = df["combined_text"].apply(make_lda_text)
    df["token_count"] = df["lda_text"].apply(lambda x: len(str(x).split()))

    df = df.drop_duplicates(subset=["clean_text"])
    df = df[df["token_count"] >= min_tokens]

    if output_name is None:
        output_name = input_path.stem + "_clean.csv"

    output_path = PROCESSED_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)


def preprocess_default_files() -> list[str]:
    ensure_data_dirs()
    outputs = []
    for path in RAW_DIR.glob("*.csv"):
        outputs.append(preprocess_file(path))
    return outputs


ST_BASIC_STOPWORDS = {
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

ST_COMPANY_STOPWORDS = {
    "tesla", "tsla", "elon", "musk",
    "starbucks", "sbux",
    "chipotle", "cmg",
}


def _st_strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def _st_normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_stocktwits_bert_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = _st_strip_urls(text)
    text = re.sub(r"\bu/[A-Za-z0-9_-]+", "@user", text)
    text = re.sub(r"\br/[A-Za-z0-9_-]+", "r/sub", text)
    return _st_normalize_spacing(text)


def make_stocktwits_lda_text(text: str, extra_stopwords: set[str] | None = None) -> str:
    text = "" if pd.isna(text) else str(text)
    text = _st_strip_urls(text)
    text = text.lower()
    text = re.sub(r"\$[a-z]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    stopwords = set(ST_BASIC_STOPWORDS) | set(ST_COMPANY_STOPWORDS)
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

    df["bert_text"] = df["text"].apply(make_stocktwits_bert_text)
    df["lda_text"] = df["text"].apply(make_stocktwits_lda_text)

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
    output_path = PROCESSED_DIR / output_name
    processed.to_csv(output_path, index=False)
    print(f"Saved {len(processed)} cleaned rows to {output_path}")
    return str(output_path)
