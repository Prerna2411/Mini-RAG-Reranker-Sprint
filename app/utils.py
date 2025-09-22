"""
Utility functions for text processing, logging, and system operations.
"""
import re
import logging
import hashlib
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    SEED, RANDOM_STATE, MODEL_NAME, EMBEDDING_DIM, 
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    LOG_LEVEL, LOG_FORMAT, MAX_TEXT_LENGTH
)

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(RANDOM_STATE)

def setup_logging(name: str = "industrial_rag") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters if configured
    # text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with memory optimization - FIXED VERSION."""
    if not text or len(text) < MIN_CHUNK_SIZE:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100000  # Process 100KB at a time
    processed_chars = 0
    
    while start < text_length:
        # Determine end position for this iteration
        end = min(start + chunk_size, text_length)
        
        # Try to break at sentence boundaries
        if end < text_length:
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        # Extract chunk text
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end,
                'length': len(chunk_text)
            })
        
        # FIXED: Move start position with overlap, but ensure we make progress
        next_start = end - overlap
        
        # Ensure we make progress - start must advance
        if next_start <= start:
            next_start = start + chunk_size // 2  # Move at least half a chunk
        
        start = next_start
        
        # Break if we've reached the end
        if start >= text_length:
            break
        
        # Memory management: yield control periodically for large texts
        processed_chars += chunk_size
        if processed_chars >= batch_size:
            import gc
            gc.collect()  # Force garbage collection
            processed_chars = 0
    
    return chunks

def chunk_text_streaming(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Generator that yields chunks one at a time to save memory."""
    if not text or len(text) < MIN_CHUNK_SIZE:
        return
    
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to break at sentence boundaries
        if end < text_length:
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            yield {
                'text': chunk_text,
                'start': start,
                'end': end,
                'length': len(chunk_text)
            }
        
        # Move start position with overlap, but ensure we make progress
        start = end - overlap
        if start >= text_length:
            break
        
        # Safety check to prevent infinite loop
        if start <= 0:
            start = end

    
def generate_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Generate unique chunk ID."""
    return f"{doc_id}_{chunk_idx:04d}"

def load_sources(sources_path: Path) -> Dict[str, Dict[str, str]]:
    """Load sources.json file."""
    try:
        with open(sources_path, 'r', encoding='utf-8') as f:
            sources_list = json.load(f)
        
        sources_dict = {}
        for i, source in enumerate(sources_list):
            # Use index as key, or extract from title if possible
            key = f"doc_{i:02d}"
            sources_dict[key] = source
        
        return sources_dict

    except Exception as e:
        logging.error(f"Failed to load sources: {e}")
        return {}

def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def create_chunks_table(conn: sqlite3.Connection) -> None:
    """Create chunks table and FTS5 virtual table."""
    cursor = conn.cursor()
    
    # Create main chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            source_url TEXT,
            char_start INTEGER,
            char_end INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create FTS5 virtual table for full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            title,
            content='chunks',
            content_rowid='rowid'
        )
    """)
    
    # Create triggers to keep FTS5 in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, title) VALUES (new.rowid, new.text, new.title);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES('delete', old.rowid, old.text, old.title);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES('delete', old.rowid, old.text, old.title);
            INSERT INTO chunks_fts(rowid, text, title) VALUES (new.rowid, new.text, new.title);
        END
    """)
    
    conn.commit()

def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model."""
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info(f"Loaded embedding model: {MODEL_NAME}")
        return model
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise

def load_faiss_index(index_path: Path, meta_path: Path) -> Tuple[faiss.Index, Dict[str, Any]]:
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index(str(index_path))
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        logging.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, metadata
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        raise

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]

def blend_scores(vector_scores: List[float], bm25_scores: List[float], alpha: float = 0.7) -> List[float]:
    """Blend vector and BM25 scores."""
    if not vector_scores or not bm25_scores:
        return vector_scores or bm25_scores or []
    
    # Normalize both score lists
    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    # Blend scores
    blended = [alpha * v + (1 - alpha) * b for v, b in zip(norm_vector, norm_bm25)]
    return blended

def should_abstain(
    contexts: List[Dict[str, Any]], 
    blended_threshold: float = 0.35,
    vector_threshold: float = 0.3,
    bm25_threshold: float = 0.2
) -> Tuple[bool, Optional[str]]:
    """Determine if the system should abstain from answering."""
    if not contexts:
        return True, "No contexts retrieved"
    
    top_context = contexts[0]
    
    # Check blended score threshold
    if top_context.get('blended_score', 0) < blended_threshold:
        return True, f"Top blended score {top_context.get('blended_score', 0):.3f} below threshold {blended_threshold}"
    
    # Check if both vector and BM25 scores are low
    vector_score = top_context.get('vector_score', 0)
    bm25_score = top_context.get('bm25_score', 0)
    
    if vector_score < vector_threshold and bm25_score < bm25_threshold:
        return True, f"Both vector ({vector_score:.3f}) and BM25 ({bm25_score:.3f}) scores below thresholds"
    
    return False, None

def format_citation(context: Dict[str, Any]) -> str:
    """Format citation for a context."""
    title = context.get('title', 'Unknown')
    doc_id = context.get('doc_id', 'unknown')
    chunk_idx = context.get('chunk_idx', 0)
    url = context.get('url', '')
    
    citation = f"[{title}]"
    if url:
        citation += f" ({url})"
    
    return citation

import re
from typing import List, Dict, Any

def extract_answer_from_contexts(contexts: List[Dict[str, Any]], query: str = "", max_sentences: int = 2) -> str:
    """
    Extract a concise answer from given contexts.
    
    Args:
        contexts: List of context dictionaries containing 'text' keys.
        query: Optional user query to focus extraction on relevant sentences.
        max_sentences: Maximum number of sentences to include from each context.
    
    Returns:
        Concise, extractive answer string.
    """
    if not contexts:
        return ""
    
    answer_parts = []

    # Lowercase keywords from query for matching
    keywords = query.lower().split() if query else []

    for ctx in contexts:
        text = ctx.get("text", "")
        sentences = re.split(r'(?<=[.!?]) +', text)
        relevant_sentences = []

        for sent in sentences:
            # If query provided, pick sentences containing query keywords
            if keywords:
                if any(k in sent.lower() for k in keywords):
                    relevant_sentences.append(sent.strip())
            else:
                # Otherwise, take first few sentences as default
                relevant_sentences.append(sent.strip())

            if len(relevant_sentences) >= max_sentences:
                break

        if relevant_sentences:
            answer_parts.append(" ".join(relevant_sentences))

    return " ".join(answer_parts).strip()


def format_citation(context: Dict[str, Any]) -> str:
    """
    Simple citation formatter for a context.
    """
    title = context.get("title", "Unknown Source")
    url = context.get("url", "")
    return f"{title}{' (' + url + ')' if url else ''}"


def should_abstain(
    contexts: List[Dict[str, Any]],
    blended_threshold: float,
    vector_threshold: float,
    bm25_threshold: float
) -> tuple[bool, str]:
    """
    Decide whether to abstain from answering based on thresholds.
    """
    if not contexts:
        return True, "No contexts retrieved"

    top = contexts[0]
    blended = top.get("blended_score", 0)
    vector = top.get("vector_score", 0)
    bm25 = top.get("bm25_score", 0)

    if blended < blended_threshold:
        return True, f"Blended score below threshold ({blended} < {blended_threshold})"
    if vector < vector_threshold:
        return True, f"Vector score below threshold ({vector} < {vector_threshold})"
    if bm25 < bm25_threshold:
        return True, f"BM25 score below threshold ({bm25} < {bm25_threshold})"

    return False, ""


def extract_definition_content(text: str, query: str) -> str:
    """Extract definitional content from text."""
    # Extract the subject being asked about
    subject = extract_subject_from_query(query)
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    relevant_sentences = []
    
    # Look for sentences that define or explain the subject
    for sentence in sentences:
        if len(sentence) < 20:
            continue
            
        sentence_lower = sentence.lower()
        
        # Priority 1: Sentences that mention the subject and contain definitional language
        if subject and subject.lower() in sentence_lower:
            if any(pattern in sentence_lower for pattern in 
                  ['is a', 'is the', 'provides', 'standard', 'applies to', 'covers', 'methodology']):
                relevant_sentences.append(sentence)
        
        # Priority 2: Sentences with definitional patterns even without exact subject match
        elif any(pattern in sentence_lower for pattern in 
                ['standard', 'methodology', 'provides', 'applies to', 'system', 'safety']):
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        # Return the most relevant sentences
        return '. '.join(relevant_sentences[:3]) + '.'
    
    # Fallback: look for the most substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    if paragraphs:
        return paragraphs[0][:400] + ('...' if len(paragraphs[0]) > 400 else '')
    
    # Last resort: return first few substantial sentences
    substantial_sentences = [s for s in sentences if len(s) > 30][:3]
    return '. '.join(substantial_sentences) + '.' if substantial_sentences else text[:300] + '...'

def extract_relevant_sentences(text: str, query: str) -> str:
    """Extract relevant sentences based on keyword matching."""
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Score sentences by keyword overlap
    scored_sentences = []
    for sentence in sentences:
        if len(sentence) < 20:
            continue
            
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        
        if overlap > 0:
            # Boost score for longer sentences with good overlap
            score = overlap * (1 + len(sentence) / 1000)
            scored_sentences.append((score, sentence))
    
    if scored_sentences:
        # Sort by score and return top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:2]]
        return '. '.join(top_sentences) + '.'
    
    # Fallback: return first substantial sentences
    substantial = [s for s in sentences if len(s) > 30][:2]
    return '. '.join(substantial) + '.' if substantial else text[:200] + '...'

def extract_subject_from_query(query: str) -> str:
    """Extract the main subject from query (e.g., 'ISO 13849-1' from 'What is ISO 13849-1 about?')."""
    query_lower = query.lower()
    
    # Handle "what is X about" pattern
    if 'what is' in query_lower:
        parts = query_lower.split('what is')
        if len(parts) > 1:
            subject_part = parts[1].split('about')[0].strip()
            # Remove common words and get the main subject
            words = subject_part.replace('?', '').strip().split()
            # Look for technical terms, standards, etc.
            for word in words:
                if any(pattern in word for pattern in ['iso', 'iec', 'en', 'din']):
                    return word.upper()
            # Return the longest word as likely subject
            if words:
                return max(words, key=len)
    
    # Look for standard identifiers in the query
    words = query.split()
    for word in words:
        if any(pattern in word.lower() for pattern in ['iso', 'iec', 'en']):
            return word
    
    return ""

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),  # Resident Set Size
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),  # Virtual Memory Size
            "percent": round(process.memory_percent(), 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}

def log_memory_usage(logger: logging.Logger, context: str = ""):
    """Log current memory usage."""
    memory_info = get_memory_usage()
    if "error" not in memory_info:
        logger.info(f"Memory usage {context}: {memory_info['rss_mb']}MB RSS, "
                   f"{memory_info['percent']}% of system, "
                   f"{memory_info['available_mb']}MB available")
    else:
        logger.warning(f"Could not get memory info {context}: {memory_info['error']}")

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "version": "1.0.0",
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "seed": SEED,
        "random_state": RANDOM_STATE
    }
