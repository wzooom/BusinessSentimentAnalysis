"""
Model setup module for loading RoBERTa ONNX model and tokenizer.
Handles model initialization and tokenizer configuration.
"""

import os
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as rt

logger = logging.getLogger(__name__)


def get_model_path():
    """Get the path to the Model directory."""
    base_path = Path(__file__).parent.parent
    return base_path / "Model"


def load_onnx_model(model_path=None):
   
    if model_path is None:
        model_path = get_model_path() / "model.onnx"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Use CPU for inference
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3  # WARNING level
        
        session = rt.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        logger.info(f"✓ ONNX model loaded successfully from {model_path}")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")


def load_tokenizer(tokenizer_dir=None):
    
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
        logger.info(f"✓ Tokenizer loaded successfully from {tokenizer_dir}")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")


def load_emotion_labels(config_path=None):

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
        
        # Convert string keys to integers
        id2label = {int(k): v for k, v in config["id2label"].items()}
        logger.info(f"✓ Loaded {len(id2label)} emotion labels from config")
        return id2label
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse config.json: {e}")


def initialize_pipeline(model_path=None, tokenizer_dir=None, config_path=None):
    """
    Returns:
        dict: Pipeline components with keys:
              - 'model': ONNX InferenceSession
              - 'tokenizer': Loaded tokenizer
              - 'id2label': Emotion label mapping (id -> name)
              - 'num_labels': Number of emotion classes (28)
    """
    try:
        logger.info("Initializing sentiment analysis pipeline...")
        
        # Load all components
        model = load_onnx_model(model_path)
        tokenizer = load_tokenizer(tokenizer_dir)
        id2label = load_emotion_labels(config_path)
        
        num_labels = len(id2label)
        logger.info(f"✓ Pipeline initialized successfully ({num_labels} emotion classes)")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label,
            "num_labels": num_labels,
        }
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
