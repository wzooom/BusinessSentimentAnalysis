"""Command line runner for Step 1 and Step 2.

Examples:
    python main.py --source reddit --subreddit stocks --limit 50
    python main.py --source stocktwits --ticker TSLA --limit 30
    python main.py --source preprocess
    python main.py --source all --subreddit stocks --ticker TSLA --limit 50
"""

from __future__ import annotations

import argparse

from api_access.preprocess import preprocess_default_files, preprocess_file
from api_access.reddit_api import save_reddit_posts
from api_access.stocktwits_api import save_stocktwits_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 and Step 2 data pipeline")
    parser.add_argument(
        "--source",
        choices=["reddit", "stocktwits", "preprocess", "reddit-preprocess", "stocktwits-preprocess", "all"],
        required=True,
        help="Which pipeline stage to run.",
    )
    parser.add_argument("--subreddit", default="stocks", help="Subreddit to scrape")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker for StockTwits")
    parser.add_argument("--limit", type=int, default=50, help="Number of posts/messages")
    parser.add_argument(
        "--listing",
        default="hot",
        choices=["hot", "new", "top", "rising"],
        help="Reddit listing type",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional Reddit keywords, e.g. --keywords TSLA Tesla",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "reddit":
        path = save_reddit_posts(
            subreddit_name=args.subreddit,
            limit=args.limit,
            listing=args.listing,
            keywords=args.keywords,
        )
        print(f"Saved raw Reddit data to {path}")

    elif args.source == "stocktwits":
        path = save_stocktwits_messages(symbol=args.ticker, limit=args.limit)
        print(f"Saved raw StockTwits data to {path}")

    elif args.source == "preprocess":
        paths = preprocess_default_files()
        for path in paths:
            print(f"Saved cleaned data to {path}")

    elif args.source == "reddit-preprocess":
        raw_path = save_reddit_posts(
            subreddit_name=args.subreddit,
            limit=args.limit,
            listing=args.listing,
            keywords=args.keywords,
        )
        print(f"Saved raw Reddit data to {raw_path}")
        clean_path = preprocess_file(raw_path)
        print(f"Saved cleaned Reddit data to {clean_path}")

    elif args.source == "stocktwits-preprocess":
        raw_path = save_stocktwits_messages(symbol=args.ticker, limit=args.limit)
        print(f"Saved raw StockTwits data to {raw_path}")
        clean_path = preprocess_file(raw_path)
        print(f"Saved cleaned StockTwits data to {clean_path}")

    elif args.source == "all":
        reddit_path = save_reddit_posts(
            subreddit_name=args.subreddit,
            limit=args.limit,
            listing=args.listing,
            keywords=args.keywords,
        )
        stocktwits_path = save_stocktwits_messages(symbol=args.ticker, limit=args.limit)
        print(f"Saved raw Reddit data to {reddit_path}")
        print(f"Saved raw StockTwits data to {stocktwits_path}")

        for path in [reddit_path, stocktwits_path]:
            clean_path = preprocess_file(path)
            print(f"Saved cleaned data to {clean_path}")


if __name__ == "__main__":
    main()
