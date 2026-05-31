from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import RAW_DIR, ensure_data_dirs, get_env

BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


def _load_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sentiment_label(message: dict[str, Any]) -> str:
    sentiment = message.get("entities", {}).get("sentiment")
    if isinstance(sentiment, dict):
        return sentiment.get("basic", "") or ""
    return ""


def _normalize_message(message: dict[str, Any], company: str, ticker: str) -> dict[str, Any]:
    user = message.get("user") or {}
    likes = message.get("likes")
    if isinstance(likes, dict):
        likes_total = likes.get("total", 0)
    else:
        likes_total = 0

    return {
        "company": company,
        "source": "stocktwits",
        "ticker": ticker,
        "message_id": message.get("id"),
        "created_datetime": message.get("created_at", ""),
        "username": user.get("username", ""),
        "text": message.get("body", "") or "",
        "likes": likes_total,
        "stocktwits_sentiment_label": _sentiment_label(message),
    }


def fetch_stocktwits_messages(
    company: str,
    ticker: str,
    target: int = 5000,
    delay_seconds: float = 1.5,
    max_rate_limit_wait: int = 600,
    output_name: str | None = None,
) -> str:
    ensure_data_dirs()

    ticker = ticker.upper()
    output_name = output_name or f"stocktwits_{company}_{ticker}_messages.csv"
    output_path = RAW_DIR / output_name

    existing = _load_existing(output_path)
    seen_ids = set()
    if not existing.empty and "message_id" in existing.columns:
        seen_ids = set(existing["message_id"].dropna().astype(str))

    records: list[dict[str, Any]] = []
    max_id = None
    pages = 0
    no_new_pages = 0
    wait_seconds = 30

    if not existing.empty and "message_id" in existing.columns:
        numeric_ids = pd.to_numeric(existing["message_id"], errors="coerce").dropna()
        if not numeric_ids.empty:
            max_id = int(numeric_ids.min()) - 1

    headers = {"User-Agent": get_env("STOCKTWITS_USER_AGENT", "BusinessSentimentAnalysis/1.0")}

    print(f"Starting StockTwits pull for {company} ({ticker}). Existing rows: {len(existing)}. Target: {target}")

    while len(existing) + len(records) < target:
        params: dict[str, Any] = {}
        if max_id is not None:
            params["max"] = max_id

        response = requests.get(
            BASE_URL.format(ticker=ticker),
            headers=headers,
            params=params,
            timeout=30,
        )

        if response.status_code == 429:
            print(f"StockTwits rate limit 429. Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, max_rate_limit_wait)
            continue

        if response.status_code in {403, 404}:
            print(f"StockTwits returned {response.status_code} for {ticker}. Stopping.")
            break

        if response.status_code != 200:
            print(f"StockTwits request failed with {response.status_code}: {response.text[:200]}")
            print(f"Waiting {wait_seconds} seconds and retrying...")
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, max_rate_limit_wait)
            continue

        wait_seconds = 30
        payload = response.json()
        messages = payload.get("messages", [])

        if not messages:
            print("No more messages returned. Stopping.")
            break

        page_new = 0
        oldest_id = None

        for message in messages:
            message_id = message.get("id")
            if message_id is None:
                continue

            oldest_id = message_id
            message_id_key = str(message_id)
            if message_id_key in seen_ids:
                continue

            normalized = _normalize_message(message, company=company, ticker=ticker)
            if not normalized["text"].strip():
                continue

            seen_ids.add(message_id_key)
            records.append(normalized)
            page_new += 1

            if len(existing) + len(records) >= target:
                break

        pages += 1

        if records:
            new_df = pd.DataFrame(records)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["message_id"])
            combined.to_csv(output_path, index=False)
            existing = combined
            records = []

        print(f"Page {pages}: added {page_new} new rows. Total saved: {len(existing)}")

        if page_new == 0:
            no_new_pages += 1
        else:
            no_new_pages = 0

        if no_new_pages >= 3:
            print("Saw 3 pages with no new messages. Stopping.")
            break

        if oldest_id is None:
            break

        try:
            max_id = int(oldest_id) - 1
        except (TypeError, ValueError):
            print("Could not determine next max_id. Stopping.")
            break

        time.sleep(delay_seconds)

    print(f"Finished {company} ({ticker}). Raw file: {output_path}. Rows: {len(existing)}")
    return str(output_path)
