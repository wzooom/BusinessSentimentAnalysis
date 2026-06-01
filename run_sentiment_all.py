#!/usr/bin/env python3
"""
Run BERT sentiment analysis on all CSV files in data/processed/
Outputs to data/scored/ with emotion scores added.
"""

import logging
from pathlib import Path
from src.sentiment import run_sentiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# CSV files to process
csv_configs = [
    {
        "company": "tesla",
        "input": "data/processed/stocktwits_tesla_TSLA_messages_clean.csv",
        "output": "data/scored/stocktwits_tesla_TSLA_messages_scored.csv",
    },
    {
        "company": "tesla", 
        "input": "data/processed/stocktwits_TSLA_messages_clean.csv",
        "output": "data/scored/stocktwits_TSLA_messages_scored.csv",
    },
    {
        "company": "reddit",
        "input": "data/processed/reddit_posts_clean.csv",
        "output": "data/scored/reddit_posts_scored.csv",
    },
]

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BERT SENTIMENT ANALYSIS - PROCESS ALL CSV FILES")
    print("="*70 + "\n")
    
    processed = 0
    failed = 0
    
    for config in csv_configs:
        company = config["company"]
        input_path = config["input"]
        output_path = config["output"]
        
        if not Path(input_path).exists():
            print(f"⊘ SKIP: {input_path} (file not found)")
            continue
        
        print(f"\n→ Processing: {input_path}")
        print(f"  Company: {company}")
        print(f"  Output:  {output_path}")
        print(f"  Status:  Running inference...\n")
        
        try:
            result = run_sentiment(
                company=company,
                input_csv_path=input_path,
                output_path=output_path,
                use_chunking=True
            )
            print(f"\n✓ SUCCESS: Saved to {result}")
            processed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {processed} processed, {failed} failed")
    print("="*70 + "\n")
