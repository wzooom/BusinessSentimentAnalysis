# Sentiment Analysis Pipeline

A Python module for extracting 28 emotion labels from text using a RoBERTa-based ONNX model.

## Quick Start

```python
from Pipeline.main import process_sentiment_file

# Process a CSV file
result = process_sentiment_file("Input/reddit_posts_clean.csv")

if result["success"]:
    print(f"✓ Processed {result['input_rows']} rows")
    print(f"✓ Saved to: {result['output_path']}")
    print(f"✓ Added {result['emotions_added']} emotion columns")
```

## Installation

Dependencies are already installed:
- `onnxruntime` - ONNX model inference
- `transformers` - Tokenizer loading
- `pandas` - CSV handling
- `numpy` - Numerical operations

## How It Works

1. **Load Model**: Loads RoBERTa ONNX model and tokenizer
2. **Read CSV**: Reads input CSV with `clean_text` column
3. **Tokenize**: Tokenizes text using RoBERTa tokenizer
4. **Inference**: Runs emotion classification on all texts
5. **Add Columns**: Adds 28 emotion score columns to CSV
6. **Save**: Saves enriched CSV to `Output/` folder
7. **Validate**: Verifies all scores are in [0, 1] range

## Output Format

Input CSV is enriched with 28 new columns (one per emotion):

```
clean_text,emotion_admiration,emotion_amusement,emotion_anger,...,emotion_surprise
"I love this!",0.92,0.05,0.01,...,0.12
"That makes me sad",0.03,0.02,0.08,...,0.65
```

Column naming: `emotion_{emotion_name}`

All scores are confidence values from 0.0 to 1.0.

## 28 Emotion Labels

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise

## API

### Main Function

```python
process_sentiment_file(
    input_csv_path,           # Path to CSV with 'clean_text' column
    output_folder="Output",   # Where to save processed CSV
    model_path=None,          # Optional: path to model.onnx
    tokenizer_dir=None,       # Optional: path to tokenizer directory
    config_path=None          # Optional: path to config.json
)
```

Returns: Dictionary with processing results

### Batch Processing

```python
from Pipeline.main import batch_process_sentiment_files

results = batch_process_sentiment_files([
    "Input/reddit_posts_clean.csv",
    "Input/stocktwits_TSLA_messages_clean.csv"
])
```

### Low-Level Functions

For more control, use individual functions:

```python
from Pipeline.model_setup import initialize_pipeline
from Pipeline.sentiment_pipeline import (
    load_csv, infer_emotions, add_emotion_columns, save_processed_csv
)

# Setup
pipeline = initialize_pipeline()
model = pipeline['model']
tokenizer = pipeline['tokenizer']
id2label = pipeline['id2label']

# Process
df = load_csv("Input/reddit_posts_clean.csv")
texts = df["clean_text"].tolist()
emotion_scores = infer_emotions(texts, model, tokenizer, id2label)
df_with_emotions = add_emotion_columns(df, emotion_scores)
output_path = save_processed_csv(df_with_emotions, "Output")
```

## Testing

Run the test script to verify everything works:

```bash
python Pipeline/test_pipeline.py
```

## Files

- `model_setup.py` - Model and tokenizer loading
- `sentiment_pipeline.py` - CSV processing and inference
- `main.py` - Main orchestration functions
- `test_pipeline.py` - Quick test script
- `USAGE_GUIDE.py` - Detailed usage examples (open as Python comments)
- `README.md` - This file

## Requirements

- Python 3.11+
- Input CSV must have a `clean_text` column
- Model files in `Model/` folder:
  - `model.onnx`
  - `tokenizer.json`
  - `vocab.json`
  - `special_tokens_map.json`
  - `config.json`

## Performance

- ~1-3 seconds for 10 texts
- ~30-60 seconds for 1000 texts
- Runs on CPU (no GPU needed)

## Output Location

Processed CSVs are saved to `Output/` folder with naming:
- Input: `Input/reddit_posts_clean.csv`
- Output: `Output/reddit_posts_clean_with_emotions.csv`

**Note**: Input folder is never modified

## Troubleshooting

**Error: 'clean_text' column not found**
- Ensure input CSV has exactly this column name

**Error: Model/tokenizer not found**
- Check Model/ folder contains all required files

**Processing is slow**
- Normal on CPU. For 10,000 rows: 2-5 minutes

## Example Workflow

```python
from Pipeline.main import process_sentiment_file

# Process reddit posts
result1 = process_sentiment_file("Input/reddit_posts_clean.csv")
print(f"Reddit: {result1['output_path']}")

# Process stocktwits messages
result2 = process_sentiment_file("Input/stocktwits_TSLA_messages_clean.csv")
print(f"Stocktwits: {result2['output_path']}")

# Both CSVs now enriched with emotion scores in Output/ folder
```

---

For detailed usage examples, see `Pipeline/USAGE_GUIDE.py`
