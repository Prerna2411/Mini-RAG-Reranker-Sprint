"""
Configuration settings for the industrial RAG system.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "store"
RAW_DIR = DATA_DIR / "raw"

# Ensure directories exist
STORE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

# Database and index paths
CHUNKS_DB_PATH = STORE_DIR / "chunks.db"
FAISS_INDEX_PATH = STORE_DIR / "faiss_index.bin"
FAISS_META_PATH = STORE_DIR / "meta.json"
SOURCES_JSON_PATH = DATA_DIR / "sources.json"

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
BATCH_SIZE = 32


# Chunking parameters - following task requirements (200-400 words per chunk)
CHUNK_SIZE = 2000     # ~300-400 words (5-6 chars per word average)
CHUNK_OVERLAP = 250   # ~40-50 words overlap - good context preservation
MIN_CHUNK_SIZE = 150  # Minimum viable chunk

# Search parameters
DEFAULT_K = 5
MAX_K = 20
FAISS_TOP_K = 50  # Retrieve more for reranking

# Reranking parameters
RERANK_ALPHA = 0.7  # Weight for vector score in hybrid reranking
BM25_TOP_K = 30     # BM25 candidates to consider

# Abstention thresholds
BLENDED_THRESHOLD = 0.35
VECTOR_THRESHOLD = 0.3
BM25_THRESHOLD = 0.2

# Deterministic settings
SEED = 42
RANDOM_STATE = 42

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Industrial Safety RAG API"
API_VERSION = "1.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Text processing
MAX_TEXT_LENGTH = 10000
CLEAN_TEXT_REGEX = r'\s+'
REMOVE_SPECIAL_CHARS = False

# Performance settings
SQLITE_TIMEOUT = 30
FAISS_THREADS = 4

# Version info
VERSION = "1.0.0"
BUILD_DATE = "2024-01-01"

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "store_dir": str(STORE_DIR),
            "raw_dir": str(RAW_DIR),
            "chunks_db": str(CHUNKS_DB_PATH),
            "faiss_index": str(FAISS_INDEX_PATH),
            "faiss_meta": str(FAISS_META_PATH),
            "sources_json": str(SOURCES_JSON_PATH),
        },
        "model": {
            "name": MODEL_NAME,
            "dimension": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
        },
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE,
        },
        "search": {
            "default_k": DEFAULT_K,
            "max_k": MAX_K,
            "faiss_top_k": FAISS_TOP_K,
        },
        "reranking": {
            "alpha": RERANK_ALPHA,
            "bm25_top_k": BM25_TOP_K,
        },
        "thresholds": {
            "blended": BLENDED_THRESHOLD,
            "vector": VECTOR_THRESHOLD,
            "bm25": BM25_THRESHOLD,
        },
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "title": API_TITLE,
            "version": API_VERSION,
        },
        "system": {
            "seed": SEED,
            "random_state": RANDOM_STATE,
            "version": VERSION,
            "build_date": BUILD_DATE,
        }
    }
