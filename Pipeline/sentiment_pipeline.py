import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_csv(filepath):
    #returns Pandas DF
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✓ Loaded CSV: {filepath.name} ({len(df)} rows)")
        
        if "bert_text" not in df.columns:
            raise ValueError(f"'bert_text' column not found in CSV. Available columns: {df.columns.tolist()}")
        
        return df
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Failed to parse CSV: {e}")


def preprocess_text(text, max_length=512):

    #max_length (int): Maximum token length (RoBERTa limit is 514)
    #str: Preprocessed text (truncated if needed)

    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    return text


def chunk_text(text, chunk_size=400, chunk_overlap=50):
    """ 
    Args:
        text (str): Input text to chunk
        chunk_size (int): Approximate chunk size in tokens (default 400)
        chunk_overlap (int): Overlap between chunks in tokens (default 50)
    
    Returns:
        List[dict]: List of chunks with keys:
                   - 'content': chunk text
                   - 'start_char': character index in original text
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # RecursiveCharacterTextSplitter works with character counts (approximate tokens)
    # 1 token ≈ 4 characters, so scale chunk_size
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # Track chunk positions in original text
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
            # Fallback if exact position not found
            chunks_with_pos.append({
                'content': chunk,
                'start_char': -1
            })
    
    return chunks_with_pos


def chunk_texts(texts, chunk_size=400, chunk_overlap=50):
    """
    Chunk multiple texts and return flat list with source tracking.
    
    Returns:
        List[dict]: List of all chunks with keys:
                   - 'content': chunk text
                   - 'source_idx': index of original text
                   - 'chunk_idx': index within that text's chunks
    """
    all_chunks = []
    
    for source_idx, text in enumerate(texts):
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk['source_idx'] = source_idx
            chunk['chunk_idx'] = chunk_idx
            all_chunks.append(chunk)
    
    return all_chunks


def infer_emotions(texts, model, tokenizer, id2label, batch_size=8, max_length=512, use_chunking=True, chunk_size=400, chunk_overlap=50):
    
    # Args:
    #     texts (List[str]): List of texts to analyze
    #     model: ONNX Runtime InferenceSession
    #     tokenizer: HuggingFace tokenizer
    #     id2label (dict): Emotion label mapping {id: name}
    #     batch_size (int): Batch size for inference (default 8)
    #     max_length (int): Max tokenizer length (default 512)
    #     use_chunking (bool): Enable text chunking for long texts (default True)
    #     chunk_size (int): Chunk size in tokens (default 400)
    #     chunk_overlap (int): Chunk overlap in tokens (default 50)
    
    # Returns:
    #     dict: Mapping of emotion names to lists of scores
    #     If use_chunking: returns aggregated scores per original text
    #     If not: returns scores per text
    
    if not texts:
        raise ValueError("Texts list is empty")
    
    num_labels = len(id2label)
    
    try:
        # Chunk texts if enabled
        if use_chunking:
            logger.info(f"Chunking texts (chunk_size={chunk_size}, overlap={chunk_overlap})...")
            chunks_list = chunk_texts(texts, chunk_size, chunk_overlap)
            chunk_contents = [c['content'] for c in chunks_list]
            num_chunks = len(chunks_list)
            logger.info(f"✓ Created {num_chunks} chunks from {len(texts)} texts")
        else:
            chunk_contents = texts
            chunks_list = [{'source_idx': i, 'chunk_idx': 0, 'content': t} for i, t in enumerate(texts)]
            num_chunks = len(texts)
        
        # Initialize output for chunks
        chunk_emotion_scores = {id2label[i]: [] for i in range(num_labels)}
        chunk_lengths = []  # Track character length of each chunk for weighting
        
        logger.info(f"Running inference on {num_chunks} chunks (batch size: {batch_size})...")
        
        # Process in batches
        for batch_idx in range(0, len(chunk_contents), batch_size):
            batch_texts = chunk_contents[batch_idx:batch_idx + batch_size]
            batch_texts = [preprocess_text(t, max_length) for t in batch_texts]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )
            
            # Run ONNX inference
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            ort_outputs = model.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            # Convert logits to probabilities (softmax)
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Store scores and chunk lengths
            for label_id in range(num_labels):
                emotion_name = id2label[label_id]
                scores_for_emotion = probs[:, label_id].tolist()
                chunk_emotion_scores[emotion_name].extend(scores_for_emotion)
            
            for text in batch_texts:
                chunk_lengths.append(len(text))
            
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(chunk_contents) + batch_size - 1) // batch_size
            logger.info(f"  Processed batch {batch_num}/{total_batches}")
        
        logger.info(f"✓ Inference complete. Extracted {num_labels} emotion scores for {num_chunks} chunks")
        
        # Aggregate chunks to original texts if chunking was used
        if use_chunking:
            logger.info("Aggregating chunk scores using chunk-length weighted average...")
            emotion_scores = aggregate_chunk_scores(
                chunk_emotion_scores,
                chunks_list,
                chunk_lengths,
                len(texts),
                id2label
            )
            logger.info(f"✓ Aggregated to {len(texts)} original texts")
        else:
            emotion_scores = chunk_emotion_scores
        
        return emotion_scores
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Failed to run inference: {e}")


def aggregate_chunk_scores(chunk_scores, chunks_list, chunk_lengths, num_original_texts, id2label):
    """
    Args:
        chunk_scores (dict): Emotion scores for each chunk
        chunks_list (List[dict]): Metadata about chunks (source_idx, chunk_idx, etc.)
        chunk_lengths (List[int]): Character length of each chunk
        num_original_texts (int): Number of original texts
        id2label (dict): Emotion label mapping
    
    Returns:
        dict: Aggregated emotion scores per original text (weighted by chunk length)
    """
    num_labels = len(id2label)
    aggregated_scores = {id2label[i]: [] for i in range(num_labels)}
    
    # Group chunks by source text
    chunks_by_source = {}
    for chunk_idx, chunk_meta in enumerate(chunks_list):
        source_idx = chunk_meta['source_idx']
        if source_idx not in chunks_by_source:
            chunks_by_source[source_idx] = []
        chunks_by_source[source_idx].append(chunk_idx)
    
    # Aggregate each source text's chunks
    for source_idx in range(num_original_texts):
        chunk_indices = chunks_by_source.get(source_idx, [])
        
        if not chunk_indices:
            # No chunks for this text (shouldn't happen)
            for emotion_name in aggregated_scores.keys():
                aggregated_scores[emotion_name].append(0.0)
            continue
        
        # Get lengths of this text's chunks
        text_chunk_lengths = np.array([chunk_lengths[idx] for idx in chunk_indices])
        total_length = np.sum(text_chunk_lengths)
        
        # Calculate weights based on chunk length
        weights = text_chunk_lengths / total_length if total_length > 0 else np.ones_like(text_chunk_lengths) / len(text_chunk_lengths)
        
        # Weighted average for each emotion
        for emotion_name in aggregated_scores.keys():
            emotion_scores = np.array([chunk_scores[emotion_name][idx] for idx in chunk_indices])
            weighted_avg = np.sum(emotion_scores * weights)
            aggregated_scores[emotion_name].append(float(weighted_avg))
    
    return aggregated_scores


def add_emotion_columns(df, emotion_scores):

    df_copy = df.copy()
    
    # Validate scores length
    num_rows = len(df_copy)
    for emotion, scores in emotion_scores.items():
        if len(scores) != num_rows:
            raise ValueError(
                f"Score count mismatch for '{emotion}': "
                f"expected {num_rows}, got {len(scores)}"
            )
    
    # Add columns in sorted order of emotion names for consistency
    for emotion in sorted(emotion_scores.keys()):
        col_name = f"emotion_{emotion}"
        df_copy[col_name] = emotion_scores[emotion]
    
    logger.info(f"✓ Added {len(emotion_scores)} emotion columns to DataFrame")
    return df_copy


def save_processed_csv(df, output_folder, original_filename=None):
   
    output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if original_filename:
        base_name = Path(original_filename).stem
        output_filename = f"{base_name}_with_emotions.csv"
    else:
        output_filename = "output_with_emotions.csv"
    
    output_path = output_folder / output_filename
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved processed CSV to: {output_path}")
        logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise RuntimeError(f"Failed to save CSV: {e}")


def validate_output_csv(output_path, expected_emotions=28):
   
    output_path = Path(output_path)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output CSV not found: {output_path}")
    
    try:
        df = pd.read_csv(output_path)
        issues = []
        stats = {}
        
        # Check emotion columns exist
        emotion_cols = [col for col in df.columns if col.startswith("emotion_")]
        stats['emotion_columns_found'] = len(emotion_cols)
        
        if len(emotion_cols) != expected_emotions:
            issues.append( f"Expected {expected_emotions} emotion columns, found {len(emotion_cols)}")
        
        # Check score ranges [0, 1]
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
