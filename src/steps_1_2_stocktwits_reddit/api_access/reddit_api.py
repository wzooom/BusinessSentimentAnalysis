"""Step 1: Reddit ingestion using Reddit's public JSON endpoint.

I actually don't need Reddit developer credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests

from api_access.config import RAW_DATA_DIR, ensure_data_dirs, get_env


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
    output_path = RAW_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)
