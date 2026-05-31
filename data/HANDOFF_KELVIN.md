# Handoff: Re-score Tesla with extended history

**Generated:** 2026-05-31
**Author:** Will (via pipeline tooling)

## What changed

Tesla data was previously 2 days (May 26–28). It's now **42 days** (April 19 → May 31), enough for weekly aggregation, WoW comparison, and spike detection. Chipotle and Starbucks are unchanged.

## What you need to do

Re-run RoBERTa go_emotions on the new Tesla cleaned CSV, then regenerate the combined-with-emotions file.

### Input (new — re-score this)

`data/processed/stocktwits_tesla_TSLA_messages_clean.csv`
- 23,548 rows (after dedup + 3-token filter from 35,000 raw)
- Columns: `company, source, ticker, created_datetime, sentiment_label, bert_text, lda_text`
- Same schema you've been using

### Existing combined file (regenerate)

`data/stocktwits_all_messages_clean_combined_with_emotions.csv` (3,584 rows, includes old 2-day Tesla data)

### Procedure

1. Drop existing `company == "tesla"` rows from the combined CSV (~1,143 rows).
2. Run `SamLowe/roberta-base-go_emotions-onnx` on the new Tesla cleaned CSV. Use `bert_text` as input.
3. Append the 28 `emotion_*` columns (`emotion_admiration` ... `emotion_surprise`) to the new Tesla rows.
4. Concat the scored Tesla rows with the unchanged Chipotle + Starbucks rows in the existing combined CSV.
5. Save back to `data/stocktwits_all_messages_clean_combined_with_emotions.csv`.

### No schema change

Same 35 columns, same column order. Downstream LDA + aggregate + brief already work — they just need fresh Tesla emotion data.

## After you push

I'll re-run:
- `python -m main lda --company tesla` (fresh LDA on 23K rows)
- `python -m main aggregate --company tesla` (now with weekly + WoW + spikes)
- `python -m main brief --company tesla`

Pipeline-side changes already in place:
- `src/stocktwits_api.py` (StockTwits pagination, import-fixed)
- `src/preprocess.py` (added `preprocess_stocktwits_file`, `make_stocktwits_bert_text`, `make_stocktwits_lda_text`)
- `src/brief.py` (full implementation, uses `google-genai` SDK)

## Delete this file when done

It's a one-time coordination artifact.
