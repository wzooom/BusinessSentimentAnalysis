"""
SENTIMENT ANALYSIS PIPELINE - USAGE GUIDE
==========================================

This pipeline extracts 28 emotion labels from text using a RoBERTa-based ONNX model.
"""

# ============================================================================
# QUICK START
# ============================================================================

from Pipeline.main import process_sentiment_file

# Process a single CSV file
result = process_sentiment_file("Input/stocktwits_all_messages_clean_combined.csv")

if result["success"]:
    print(f"✓ Processed {result['input_rows']} rows")
    print(f"✓ Saved to: {result['output_path']}")
    print(f"✓ Added {result['emotions_added']} emotion columns")
else:
    print(f"✗ Error: {result['error']}")


# ============================================================================
# MAIN FUNCTION - process_sentiment_file()
# ============================================================================

def process_sentiment_file(
    input_csv_path,
    output_folder="Output",
    model_path=None,
    tokenizer_dir=None,
    config_path=None,
):
    """
    Process a CSV file through the sentiment analysis pipeline.
    
    **WORKFLOW:**
    1. Initialize BERT model + tokenizer
    2. Read CSV and validate 'clean_text' column
    3. Extract text from 'clean_text' column
    4. Run emotion inference on all texts
    5. Add 28 emotion score columns
    6. Save enriched CSV to Output/ folder
    7. Validate output quality
    
    **PARAMETERS:**
    
    input_csv_path (required)
        Path to CSV file with 'clean_text' column
        Example: "Input/reddit_posts_clean.csv"
    
    output_folder (optional, default: "Output")
        Where to save the processed CSV
        Output file named: {original_name}_with_emotions.csv
        Example: Output/reddit_posts_clean_with_emotions.csv
    
    model_path (optional, default: "Model/model.onnx")
        Path to ONNX model file
    
    tokenizer_dir (optional, default: "Model/")
        Directory containing tokenizer files
    
    config_path (optional, default: "Model/config.json")
        Path to model config with emotion labels
    
    **RETURNS:**
    dict with keys:
        - success (bool): True if processing completed
        - output_path (Path): Path to saved CSV file
        - input_rows (int): Number of rows processed
        - output_rows (int): Number of rows in output
        - emotions_added (int): Number of emotion columns (28)
        - validation (dict): Output validation results
        - error (str): Error message if failed
    
    **EXAMPLE USAGE:**
    
    # Basic usage
    result = process_sentiment_file("Input/reddit_posts_clean.csv")
    
    # With custom output folder
    result = process_sentiment_file(
        "Input/stocktwits_TSLA_messages_clean.csv",
        output_folder="ProcessedData"
    )
    
    # Check results
    if result["success"]:
        print(f"Saved to: {result['output_path']}")
        if result["validation"]["valid"]:
            print("✓ All validations passed")
    else:
        print(f"Error: {result['error']}")
    """
    pass  # See Pipeline/main.py for implementation


# ============================================================================
# OUTPUT FORMAT
# ============================================================================

"""
Output CSV Structure:
=====================

Original Columns: All columns from input CSV (including 'clean_text')

New Emotion Columns: 28 new columns added, one per emotion
    emotion_admiration: 0.0 - 1.0 (confidence score)
    emotion_amusement: 0.0 - 1.0
    emotion_anger: 0.0 - 1.0
    emotion_annoyance: 0.0 - 1.0
    emotion_approval: 0.0 - 1.0
    emotion_caring: 0.0 - 1.0
    emotion_confusion: 0.0 - 1.0
    emotion_curiosity: 0.0 - 1.0
    emotion_desire: 0.0 - 1.0
    emotion_disappointment: 0.0 - 1.0
    emotion_disapproval: 0.0 - 1.0
    emotion_disgust: 0.0 - 1.0
    emotion_embarrassment: 0.0 - 1.0
    emotion_excitement: 0.0 - 1.0
    emotion_fear: 0.0 - 1.0
    emotion_gratitude: 0.0 - 1.0
    emotion_grief: 0.0 - 1.0
    emotion_joy: 0.0 - 1.0
    emotion_love: 0.0 - 1.0
    emotion_nervousness: 0.0 - 1.0
    emotion_neutral: 0.0 - 1.0
    emotion_optimism: 0.0 - 1.0
    emotion_pride: 0.0 - 1.0
    emotion_realization: 0.0 - 1.0
    emotion_relief: 0.0 - 1.0
    emotion_remorse: 0.0 - 1.0
    emotion_sadness: 0.0 - 1.0
    emotion_surprise: 0.0 - 1.0

EXAMPLE ROW:
clean_text: "I love this stock!"
emotion_admiration: 0.12
emotion_amusement: 0.05
emotion_anger: 0.02
...
emotion_joy: 0.78
emotion_love: 0.82
emotion_neutral: 0.03
...

Note: All scores sum to approximately 1.0 across all emotions
(they are normalized softmax probabilities from the model)
"""


# ============================================================================
# BATCH PROCESSING
# ============================================================================

from Pipeline.main import batch_process_sentiment_files

# Process multiple CSV files
input_files = [
    "Input/reddit_posts_clean.csv",
    "Input/stocktwits_TSLA_messages_clean.csv",
]

batch_result = batch_process_sentiment_files(input_files)

print(f"Total files: {batch_result['total_files']}")
print(f"Successful: {batch_result['successful']}")
print(f"Failed: {batch_result['failed']}")

for idx, result in enumerate(batch_result['results']):
    if result['success']:
        print(f"✓ File {idx+1}: {result['output_path']}")
    else:
        print(f"✗ File {idx+1}: {result['error']}")


# ============================================================================
# ADVANCED: USING INDIVIDUAL FUNCTIONS
# ============================================================================

from Pipeline.model_setup import initialize_pipeline, load_emotion_labels
from Pipeline.sentiment_pipeline import (
    load_csv,
    infer_emotions,
    add_emotion_columns,
    save_processed_csv,
    validate_output_csv,
)

# Initialize pipeline
pipeline = initialize_pipeline()
model = pipeline['model']
tokenizer = pipeline['tokenizer']
id2label = pipeline['id2label']

# Load CSV
df = load_csv("Input/reddit_posts_clean.csv")

# Extract texts
texts = df["clean_text"].tolist()

# Run inference
emotion_scores = infer_emotions(texts, model, tokenizer, id2label)

# Add emotion columns
df_with_emotions = add_emotion_columns(df, emotion_scores)

# Save output
output_path = save_processed_csv(df_with_emotions, "Output", "reddit_posts_clean.csv")

# Validate
validation = validate_output_csv(output_path)
print(f"Valid: {validation['valid']}")
print(f"Issues: {validation['issues']}")


# ============================================================================
# EMOTION LABELS (28 TOTAL)
# ============================================================================

emotions = [
    "admiration",       # 0
    "amusement",        # 1
    "anger",            # 2
    "annoyance",        # 3
    "approval",         # 4
    "caring",           # 5
    "confusion",        # 6
    "curiosity",        # 7
    "desire",           # 8
    "disappointment",   # 9
    "disapproval",      # 10
    "disgust",          # 11
    "embarrassment",    # 12
    "excitement",       # 13
    "fear",             # 14
    "gratitude",        # 15
    "grief",            # 16
    "joy",              # 17
    "love",             # 18
    "nervousness",      # 19
    "neutral",          # 20
    "optimism",         # 21
    "pride",            # 22
    "realization",      # 23
    "relief",           # 24
    "remorse",          # 25
    "sadness",          # 26
    "surprise",         # 27
]


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
ERROR: 'clean_text' column not found
SOLUTION: Ensure input CSV has a column named exactly 'clean_text'

ERROR: Score values out of range [0, 1]
SOLUTION: This shouldn't happen. Report this as a bug. Scores are normalized.

ERROR: ONNX model not found
SOLUTION: Ensure Model/model.onnx exists in workspace

ERROR: Tokenizer files not found
SOLUTION: Ensure Model/ folder contains:
  - tokenizer.json
  - config.json
  - special_tokens_map.json
  - vocab.json

ERROR: Processing very slow
SOLUTION: This is normal. For 1000 rows: ~15-30 seconds
         For 10000 rows: ~2-5 minutes
         Model runs on CPU (slower) but no GPU required.
"""


# ============================================================================
# TESTING
# ============================================================================

"""
Run the test script to verify everything works:

    python Pipeline/test_pipeline.py

This processes Input/reddit_posts_clean.csv with a quick test and validates output.
"""
