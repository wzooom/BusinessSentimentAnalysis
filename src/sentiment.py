import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import onnxruntime as rt

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def get_model_path():
    """Get the path to the Model directory."""
    return PROJECT_ROOT / "Model"


def load_onnx_model(model_path=None):
    """Load ONNX model for inference."""
    if model_path is None:
        model_path = get_model_path() / "model.onnx"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        
        session = rt.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        logger.info(f"✓ ONNX model loaded from {model_path}")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")


def load_tokenizer(tokenizer_dir=None):
    """Load tokenizer from model directory."""
    if tokenizer_dir is None:
        tokenizer_dir = get_model_path()
    else:
        tokenizer_dir = Path(tokenizer_dir)
    
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
            trust_remote_code=False
        )
        logger.info(f"✓ Tokenizer loaded from {tokenizer_dir}")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")


def load_emotion_labels(config_path=None):
    """Load emotion label mapping from config."""
    if config_path is None:
        config_path = get_model_path() / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if "id2label" not in config:
            raise ValueError("id2label mapping not found in config.json")
        
        id2label = {int(k): v for k, v in config["id2label"].items()}
        logger.info(f"✓ Loaded {len(id2label)} emotion labels")
        return id2label
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse config.json: {e}")


def initialize_pipeline(model_path=None, tokenizer_dir=None, config_path=None):
    """Initialize sentiment analysis pipeline with all components."""
    try:
        logger.info("Initializing sentiment analysis pipeline...")
        
        model = load_onnx_model(model_path)
        tokenizer = load_tokenizer(tokenizer_dir)
        id2label = load_emotion_labels(config_path)
        
        num_labels = len(id2label)
        logger.info(f"✓ Pipeline initialized ({num_labels} emotion classes)")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label,
            "num_labels": num_labels,
        }
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


def preprocess_text(text, max_length=512):
    """Clean and prepare text for tokenization."""
    if not isinstance(text, str):
        return ""
    return text.strip()


def chunk_text(text, chunk_size=400, chunk_overlap=50):
    """Chunk a single text into smaller pieces."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    chunks_with_pos = []
    current_pos = 0
    
    for chunk in chunks:
        start_pos = text.find(chunk, current_pos)
        if start_pos != -1:
            chunks_with_pos.append({
                'content': chunk,
                'start_char': start_pos
            })
            current_pos = start_pos + len(chunk)
        else:
            chunks_with_pos.append({
                'content': chunk,
                'start_char': -1
            })
    
    return chunks_with_pos


def chunk_texts(texts, chunk_size=400, chunk_overlap=50):
    """Chunk multiple texts and return flat list with source tracking."""
    all_chunks = []
    
    for source_idx, text in enumerate(texts):
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk['source_idx'] = source_idx
            chunk['chunk_idx'] = chunk_idx
            all_chunks.append(chunk)
    
    return all_chunks


def infer_emotions(texts, model, tokenizer, id2label, batch_size=8, max_length=512, 
                   use_chunking=True, chunk_size=400, chunk_overlap=50):
    """Run emotion inference on texts using BERT model."""
    if not texts:
        raise ValueError("Texts list is empty")
    
    num_labels = len(id2label)
    
    try:
        if use_chunking:
            logger.info(f"Chunking texts (chunk_size={chunk_size}, overlap={chunk_overlap})...")
            chunks_list = chunk_texts(texts, chunk_size, chunk_overlap)
            chunk_contents = [c['content'] for c in chunks_list]
            logger.info(f"✓ Created {len(chunks_list)} chunks from {len(texts)} texts")
        else:
            chunk_contents = texts
            chunks_list = [{'source_idx': i, 'chunk_idx': 0, 'content': t} for i, t in enumerate(texts)]
        
        chunk_emotion_scores = {id2label[i]: [] for i in range(num_labels)}
        chunk_lengths = []
        
        logger.info(f"Running inference on {len(chunk_contents)} chunks (batch_size={batch_size})...")
        
        for batch_idx in range(0, len(chunk_contents), batch_size):
            batch_texts = chunk_contents[batch_idx:batch_idx + batch_size]
            batch_texts = [preprocess_text(t, max_length) for t in batch_texts]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )
            
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            ort_outputs = model.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            for label_id in range(num_labels):
                emotion_name = id2label[label_id]
                scores_for_emotion = probs[:, label_id].tolist()
                chunk_emotion_scores[emotion_name].extend(scores_for_emotion)
            
            for text in batch_texts:
                chunk_lengths.append(len(text))
            
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(chunk_contents) + batch_size - 1) // batch_size
            logger.info(f"  Processed batch {batch_num}/{total_batches}")
        
        logger.info(f"✓ Inference complete. Extracted {num_labels} emotion scores")
        
        if use_chunking:
            logger.info("Aggregating chunk scores...")
            emotion_scores = aggregate_chunk_scores(
                chunk_emotion_scores,
                chunks_list,
                chunk_lengths,
                len(texts),
                id2label
            )
        else:
            emotion_scores = chunk_emotion_scores
        
        return emotion_scores
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Failed to run inference: {e}")


def aggregate_chunk_scores(chunk_scores, chunks_list, chunk_lengths, num_original_texts, id2label):
    """Aggregate chunk scores back to original texts using weighted averaging."""
    num_labels = len(id2label)
    aggregated_scores = {id2label[i]: [] for i in range(num_labels)}
    
    chunks_by_source = {}
    for chunk_idx, chunk_meta in enumerate(chunks_list):
        source_idx = chunk_meta['source_idx']
        if source_idx not in chunks_by_source:
            chunks_by_source[source_idx] = []
        chunks_by_source[source_idx].append(chunk_idx)
    
    for source_idx in range(num_original_texts):
        chunk_indices = chunks_by_source.get(source_idx, [])
        
        if not chunk_indices:
            for emotion_name in aggregated_scores.keys():
                aggregated_scores[emotion_name].append(0.0)
            continue
        
        text_chunk_lengths = np.array([chunk_lengths[idx] for idx in chunk_indices])
        total_length = np.sum(text_chunk_lengths)
        weights = text_chunk_lengths / total_length if total_length > 0 else np.ones_like(text_chunk_lengths) / len(text_chunk_lengths)
        
        for emotion_name in aggregated_scores.keys():
            emotion_scores = np.array([chunk_scores[emotion_name][idx] for idx in chunk_indices])
            weighted_avg = np.sum(emotion_scores * weights)
            aggregated_scores[emotion_name].append(float(weighted_avg))
    
    return aggregated_scores


def add_emotion_columns(df, emotion_scores):
    """Add emotion score columns to DataFrame."""
    df_copy = df.copy()
    
    num_rows = len(df_copy)
    for emotion, scores in emotion_scores.items():
        if len(scores) != num_rows:
            raise ValueError(
                f"Score count mismatch for '{emotion}': expected {num_rows}, got {len(scores)}"
            )
    
    for emotion in sorted(emotion_scores.keys()):
        col_name = f"emotion_{emotion}"
        df_copy[col_name] = emotion_scores[emotion]
    
    logger.info(f"✓ Added {len(emotion_scores)} emotion columns")
    return df_copy


def save_processed_csv(df, output_path):
    """Save DataFrame to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved CSV to: {output_path}")
        logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise RuntimeError(f"Failed to save CSV: {e}")


def validate_output_csv(output_path, expected_emotions=28):
    """Validate emotion scores in output CSV."""
    output_path = Path(output_path)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output CSV not found: {output_path}")
    
    try:
        df = pd.read_csv(output_path)
        issues = []
        stats = {}
        
        emotion_cols = [col for col in df.columns if col.startswith("emotion_")]
        stats['emotion_columns_found'] = len(emotion_cols)
        
        if len(emotion_cols) != expected_emotions:
            issues.append(f"Expected {expected_emotions} emotion columns, found {len(emotion_cols)}")
        
        for col in emotion_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val < 0 or max_val > 1:
                issues.append(
                    f"Column '{col}': values out of range [0, 1] "
                    f"(found [{min_val:.4f}, {max_val:.4f}])"
                )
        
        stats['rows'] = len(df)
        stats['columns'] = len(df.columns)
        stats['valid_score_ranges'] = len(issues) == 0
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
    
    except Exception as e:
        raise RuntimeError(f"Validation failed: {e}")


def load_csv(filepath):
    """Load CSV file with sentiment data."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✓ Loaded CSV: {filepath.name} ({len(df)} rows)")
        return df
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Failed to parse CSV: {e}")


def run_sentiment(company: str, input_csv_path=None, output_path=None, use_chunking=True):
    """
    Run sentiment analysis on a CSV file.
    
    Args:
        company: Company name for logging
        input_csv_path: Path to input CSV (expects 'bert_text' or similar column)
        output_path: Path to save output CSV with emotion scores
        use_chunking: Whether to chunk long texts
    
    Returns:
        Path to output CSV file
    """
    logger.info(f"SENTIMENT ANALYSIS - {company.upper()}")
    
    try:
        pipeline = initialize_pipeline()
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        id2label = pipeline["id2label"]
        
        if input_csv_path is None:
            raise ValueError("input_csv_path must be provided")
        
        df = load_csv(input_csv_path)
        
        if "bert_text" not in df.columns:
            if "text" in df.columns:
                texts = df["text"].tolist()
            elif "message" in df.columns:
                texts = df["message"].tolist()
            else:
                raise ValueError(f"No text column found. Available: {df.columns.tolist()}")
        else:
            texts = df["bert_text"].tolist()
        
        logger.info(f"Extracted {len(texts)} texts")
        
        emotion_scores = infer_emotions(
            texts,
            model,
            tokenizer,
            id2label,
            use_chunking=use_chunking
        )
        
        df_output = add_emotion_columns(df, emotion_scores)
        
        if output_path is None:
            raise ValueError("output_path must be provided")
        
        output_file = save_processed_csv(df_output, output_path)
        
        validation = validate_output_csv(output_file)
        if not validation['valid']:
            logger.warning(f"Validation issues: {validation['issues']}")
        else:
            logger.info("✓ Output validation passed")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
