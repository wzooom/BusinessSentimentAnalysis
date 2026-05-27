# Step 1 and Step 2 Pipeline

This project handles the first two parts of the data mining pipeline:

1. **Ingestion**: pull Reddit posts and StockTwits messages.
2. **Preprocessing**: clean the raw text for LDA, sentiment scoring, and quote extraction.

## Setup

```bash
cd api_access_project_steps_1_2_stocktwits
pip install -r requirements.txt
cp .env.example .env
```

## Step 1: Reddit ingestion

```bash
python main.py --source reddit --subreddit stocks --limit 50
```

With keywords:

```bash
python main.py --source reddit --subreddit stocks --keywords TSLA Tesla --limit 50
```

Output:

```text
data/raw/reddit_posts.csv
```

## Step 1: StockTwits ingestion

```bash
python main.py --source stocktwits --ticker TSLA --limit 30
```

Output:

```text
data/raw/stocktwits_TSLA_messages.csv
```

## Step 2: Preprocessing

```bash
python main.py --source preprocess
```

Output:

```text
data/processed/reddit_posts_clean.csv
data/processed/stocktwits_TSLA_messages_clean.csv
```

## Run ingestion and preprocessing together

Reddit:

```bash
python main.py --source reddit-preprocess --subreddit stocks --keywords TSLA Tesla --limit 50
```

StockTwits:

```bash
python main.py --source stocktwits-preprocess --ticker TSLA --limit 30
```

Both:

```bash
python main.py --source all --subreddit stocks --keywords TSLA Tesla --ticker TSLA --limit 50
```

## Important columns

Raw files include:

```text
source, subreddit_or_ticker, post_id, created_datetime, title, text,
score_or_likes, num_comments, url, permalink
```

Processed files add:

```text
combined_text, clean_text, lda_text, token_count
```

Use `clean_text` for sentiment/quotes and `lda_text` for LDA topic modeling.
