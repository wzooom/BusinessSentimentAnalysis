"""Step 1: Reddit and StockTwits ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests

from src.config import RAW_DIR, ensure_data_dirs, get_env


# ---------- Reddit ----------
# Uses Reddit's public JSON endpoint (no developer credentials required).

def fetch_subreddit_posts(
    subreddit_name: str = "stocks",
    limit: int = 50,
    listing: str = "hot",
    keywords: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fetch posts from a subreddit and return them as a DataFrame."""
    if listing not in {"hot", "new", "top", "rising"}:
        raise ValueError("listing must be one of: hot, new, top, rising")

    url = f"https://www.reddit.com/r/{subreddit_name}/{listing}.json"
    headers = {
        "User-Agent": get_env(
            "REDDIT_USER_AGENT",
            "DataMiningProject/1.0 by u/tobyasfaw",
        )
    }
    params = {"limit": limit}

    response = requests.get(url, headers=headers, params=params, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(
            f"Reddit request failed with status {response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()
    posts = data.get("data", {}).get("children", [])
    normalized_keywords = [kw.lower() for kw in keywords] if keywords else []

    records = []
    for item in posts:
        post = item.get("data", {})
        if post.get("stickied"):
            continue

        title = post.get("title", "") or ""
        text = post.get("selftext", "") or ""
        text_blob = f"{title} {text}".lower()

        if normalized_keywords and not any(kw in text_blob for kw in normalized_keywords):
            continue

        created_utc = post.get("created_utc")
        created_datetime = ""
        if created_utc:
            created_datetime = datetime.fromtimestamp(
                created_utc, tz=timezone.utc
            ).isoformat()

        records.append(
            {
                "source": "reddit",
                "subreddit_or_ticker": subreddit_name,
                "post_id": post.get("id"),
                "created_utc": created_utc,
                "created_datetime": created_datetime,
                "title": title,
                "text": text,
                "score_or_likes": post.get("score"),
                "num_comments": post.get("num_comments"),
                "url": post.get("url"),
                "permalink": "https://www.reddit.com" + post.get("permalink", ""),
            }
        )

    return pd.DataFrame(records)


def save_reddit_posts(
    subreddit_name: str = "stocks",
    limit: int = 50,
    listing: str = "hot",
    keywords: Iterable[str] | None = None,
    output_name: str = "reddit_posts.csv",
) -> str:
    """Fetch Reddit posts and save them to data/raw."""
    ensure_data_dirs()
    df = fetch_subreddit_posts(
        subreddit_name=subreddit_name,
        limit=limit,
        listing=listing,
        keywords=keywords,
    )
    output_path = RAW_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)


# ---------- StockTwits ----------
# Uses StockTwits' public symbol stream endpoint. May rate-limit if called too often.

def fetch_stocktwits_messages(symbol: str = "TSLA", limit: int = 30) -> pd.DataFrame:
    """Fetch recent StockTwits messages for a stock ticker."""
    symbol = symbol.upper().replace("$", "").strip()
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    headers = {
        "User-Agent": get_env("STOCKTWITS_USER_AGENT", "DataMiningProject/1.0")
    }

    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(
            f"StockTwits request failed with status {response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()
    messages = data.get("messages", [])[:limit]

    records = []
    for message in messages:
        created_at = message.get("created_at", "")
        created_utc = ""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_utc = dt.astimezone(timezone.utc).timestamp()
            except ValueError:
                created_utc = ""

        user = message.get("user", {}) or {}
        entities = message.get("entities", {}) or {}
        sentiment = entities.get("sentiment") or {}

        records.append(
            {
                "source": "stocktwits",
                "subreddit_or_ticker": symbol,
                "post_id": message.get("id"),
                "created_utc": created_utc,
                "created_datetime": created_at,
                "title": "",
                "text": message.get("body", "") or "",
                "score_or_likes": message.get("likes", {}).get("total"),
                "num_comments": message.get("conversation", {}).get("replies"),
                "url": f"https://stocktwits.com/{user.get('username', '')}/message/{message.get('id')}",
                "permalink": f"https://stocktwits.com/{user.get('username', '')}/message/{message.get('id')}",
                "stocktwits_sentiment_label": sentiment.get("basic"),
            }
        )

    return pd.DataFrame(records)


def save_stocktwits_messages(
    symbol: str = "TSLA",
    limit: int = 30,
    output_name: str | None = None,
) -> str:
    """Fetch StockTwits messages and save them to data/raw."""
    ensure_data_dirs()
    symbol = symbol.upper().replace("$", "").strip()
    if output_name is None:
        output_name = f"stocktwits_{symbol}_messages.csv"

    df = fetch_stocktwits_messages(symbol=symbol, limit=limit)
    output_path = RAW_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)
