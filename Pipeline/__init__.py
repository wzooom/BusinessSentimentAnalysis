"""
Sentiment Analysis Pipeline using RoBERTa ONNX Model
Extracts emotion scores from text data using a pretrained BERT model.
"""

from .model_setup import load_onnx_model, load_tokenizer, initialize_pipeline
from .sentiment_pipeline import load_csv, infer_emotions, add_emotion_columns, save_processed_csv
from .main import process_sentiment_file

__all__ = [
    "load_onnx_model",
    "load_tokenizer",
    "initialize_pipeline",
    "load_csv",
    "infer_emotions",
    "add_emotion_columns",
    "save_processed_csv",
    "process_sentiment_file",
]
