"""
Main orchestration module for the sentiment analysis pipeline.
Provides high-level functions for end-to-end processing.
"""

import logging
from pathlib import Path
from typing import Optional

from .model_setup import initialize_pipeline
from .sentiment_pipeline import (
    load_csv,
    infer_emotions,
    add_emotion_columns,
    save_processed_csv,
    validate_output_csv,
)

logger = logging.getLogger(__name__)


def process_sentiment_file(
    input_csv_path,
    output_folder="Output",
    model_path=None,
    tokenizer_dir=None,
    config_path=None,
    use_chunking=True,
    chunk_size=400,
    chunk_overlap=50,
):
    """
    Process a CSV file through the sentiment analysis pipeline.
    
    Complete workflow:
    1. Load and initialize BERT model + tokenizer
    2. Read CSV file and validate 'bert_text' column
    3. Extract text data
    4. (Optional) Chunk texts using LangChain RecursiveCharacterTextSplitter
    5. Run emotion inference on texts/chunks
    6. Aggregate chunk scores (if chunking enabled)
    7. Add emotion columns to DataFrame
    8. Save enriched CSV to output folder
    9. Validate output
    
    Args:
        input_csv_path (str or Path): Path to input CSV file with 'bert_text' column
        output_folder (str or Path, optional):  Defaults to 'Output'
        model_path (str or Path, optional): Defaults to Model/model.onnx
        tokenizer_dir (str or Path, optional): Defaults to Model/
        config_path (str or Path, optional): Defaults to Model/config.json
        use_chunking (bool, optional) (default True)
        chunk_size (int, optional): Chunk size in tokens (default 400)
        chunk_overlap (int, optional): Overlap between chunks in tokens (default 50)
    
    Returns:
        dict: Processing results with keys:
              - 'success': bool indicating if processing completed successfully
              - 'output_path': Path to saved CSV file
              - 'input_rows': Number of rows processed
              - 'output_rows': Number of rows in output CSV
              - 'emotions_added': Number of emotion columns added
              - 'total_chunks': Number of chunks created (if chunking enabled)
              - 'validation': Validation results from validate_output_csv()
              
    Raises:
        FileNotFoundError: If input CSV or model files not found
        ValueError: If required columns missing
        RuntimeError: If processing fails at any stage
        
    Example:
        >>> result = process_sentiment_file('Input/reddit_posts_clean.csv')
        >>> if result['success']:
        ...     print(f"Saved to: {result['output_path']}")
        ...     print(f"Processed {result['input_rows']} rows")
        ...     print(f"Added {result['emotions_added']} emotion columns")
    """
    input_csv_path = Path(input_csv_path)
    output_folder = Path(output_folder)
    
    logger.info("SENTIMENT ANALYSIS PIPELINE - START")
    logger.info(f"Input CSV: {input_csv_path}")
    logger.info(f"Output Folder: {output_folder}")
    
    try:
        # Step 1: Initialize pipeline
        logger.info("\n[Step 1/5] Initializing model and tokenizer...")
        pipeline = initialize_pipeline(model_path, tokenizer_dir, config_path)
        model = pipeline["model"]
        tokenizer = pipeline["tokenizer"]
        id2label = pipeline["id2label"]
        
        # Step 2: Load CSV
        logger.info("\n[Step 2/5] Loading CSV file...")
        df = load_csv(input_csv_path)
        input_rows = len(df)
        
        logger.info("\n[Step 3/5] Extracting text data...")
        texts = df["bert_text"].tolist()
        logger.info(f"Extracted {len(texts)} text entries")
        
        logger.info("\n[Step 4/5] Running emotion inference...")
        if use_chunking:
            logger.info(f"Chunking enabled: chunk_size={chunk_size}, overlap={chunk_overlap}")
        emotion_scores = infer_emotions(
            texts,
            model,
            tokenizer,
            id2label,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Step 4: Add emotion columns
        logger.info("\n[Step 4/5] Adding emotion columns to DataFrame...")
        df_with_emotions = add_emotion_columns(df, emotion_scores)
        num_emotions = len(emotion_scores)
        
        # Step 5: Save output
        logger.info("\n[Step 5/5] Saving processed CSV...")
        output_path = save_processed_csv(
            df_with_emotions,
            output_folder,
            original_filename=input_csv_path.name,
        )
        
        # Validate output
        logger.info("\n[Validation] Checking output integrity...")
        validation_result = validate_output_csv(output_path, expected_emotions=num_emotions)
        
        if validation_result["valid"]:
            logger.info("✓ Output validation PASSED")
        else:
            logger.warning("⚠ Output validation found issues:")
            for issue in validation_result["issues"]:
                logger.warning(f"  - {issue}")
        
        logger.info("SENTIMENT ANALYSIS PIPELINE - COMPLETE")
        
        return {
            "success": True,
            "output_path": output_path,
            "input_rows": input_rows,
            "output_rows": len(df_with_emotions),
            "emotions_added": num_emotions,
            "chunking_enabled": use_chunking,
            "validation": validation_result,
        }
    
    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("SENTIMENT ANALYSIS PIPELINE - FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        logger.error("=" * 70)
        
        return {
            "success": False,
            "error": str(e),
            "output_path": None,
        }


def batch_process_sentiment_files(
    input_csv_paths,
    output_folder="Output",
    model_path=None,
    tokenizer_dir=None,
    config_path=None,
    use_chunking=True,
    chunk_size=400,
    chunk_overlap=50,
):
    """
    Process multiple CSV files sequentially through the pipeline.
    
    Args:
        input_csv_paths (List[str] or List[Path]): List of input CSV paths
        output_folder (str or Path, optional): Output directory
        model_path (str or Path, optional): Path to model.onnx
        tokenizer_dir (str or Path, optional): Directory with tokenizer files
        config_path (str or Path, optional): Path to config.json
        use_chunking (bool, optional): Enable text chunking (default True)
        chunk_size (int, optional): Chunk size in tokens (default 400)
        chunk_overlap (int, optional): Chunk overlap in tokens (default 50)
    
    Returns:
        dict: Summary of processing with keys:
              - 'total_files': Total files attempted
              - 'successful': Number of files processed successfully
              - 'failed': Number of files that failed
              - 'results': List of individual results from process_sentiment_file()
    """
    logger.info(f"Starting batch processing of {len(input_csv_paths)} files...")
    
    results = []
    for input_path in input_csv_paths:
        result = process_sentiment_file(
            input_path,
            output_folder=output_folder,
            model_path=model_path,
            tokenizer_dir=tokenizer_dir,
            config_path=config_path,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results.append(result)
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    logger.info(f"\nBatch processing summary:")
    logger.info(f"  Total: {len(results)} | Successful: {successful} | Failed: {failed}")
    
    return {
        "total_files": len(input_csv_paths),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    result = process_sentiment_file("Input/stocktwits_all_messages_clean_combined.csv")
    print(result)
