"""Step 1: StockTwits ingestion.

This uses StockTwits' public symbol stream endpoint. It may rate-limit if called too often.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from api_access.config import RAW_DATA_DIR, ensure_data_dirs, get_env


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
    output_path = RAW_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    return str(output_path)
