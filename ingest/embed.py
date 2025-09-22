"""
Embedding generation and FAISS index creation.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import (
    MODEL_NAME, EMBEDDING_DIM, BATCH_SIZE, SEED, RANDOM_STATE,
    FAISS_INDEX_PATH, FAISS_META_PATH, CHUNKS_DB_PATH
)
from app.utils import setup_logging, get_db_connection

logger = setup_logging()

class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIM
        
        # Set random seeds for reproducibility
        np.random.seed(RANDOM_STATE)
        
        logger.info(f"Initialized embedding generator with {model_name}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

class FAISSIndexBuilder:
    """Build and manage FAISS index."""
    
    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.index = None
        self.chunk_ids = []
    
    def create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created FAISS IndexFlatIP with dimension {self.dimension}")
        return self.index
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """Add embeddings to the index."""
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunk IDs
        self.chunk_ids.extend(chunk_ids)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def save_index(self, index_path: Path, meta_path: Path, db_path: Path) -> None:
        """Save index and metadata."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata = {
            "model_name": MODEL_NAME,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "chunk_ids": self.chunk_ids,
            "db_path": str(db_path),
            "index_type": "IndexFlatIP",
            "seed": SEED,
            "random_state": RANDOM_STATE
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {meta_path}")

class EmbeddingService:
    """Main service for generating embeddings and building index."""
    
    def __init__(self):
        self.generator = EmbeddingGenerator()
        self.index_builder = FAISSIndexBuilder()
    
    def process_chunks(self, db_path: Path) -> Tuple[faiss.Index, List[str]]:
        """Process all chunks from database and create embeddings."""
        logger.info("Loading chunks from database")
        
        # Load chunks from database
        chunks_data = self._load_chunks_from_db(db_path)
        
        if not chunks_data:
            raise ValueError("No chunks found in database")
        
        texts = [chunk['text'] for chunk in chunks_data]
        chunk_ids = [chunk['chunk_id'] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = self.generator.generate_embeddings(texts)
        
        # Create and populate index
        self.index_builder.create_index()
        self.index_builder.add_embeddings(embeddings, chunk_ids)
        
        return self.index_builder.index, chunk_ids
    
    def _load_chunks_from_db(self, db_path: Path) -> List[Dict[str, Any]]:
        """Load chunks from SQLite database."""
        chunks = []
        
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, text FROM chunks
                ORDER BY doc_id, chunk_idx
            """)
            
            for row in cursor.fetchall():
                chunks.append({
                    'chunk_id': row['chunk_id'],
                    'text': row['text']
                })
        
        return chunks
    
    def build_and_save_index(
        self, 
        db_path: Path, 
        index_path: Path, 
        meta_path: Path
    ) -> None:
        """Build and save the complete index."""
        logger.info("Starting index building process")
        
        # Process chunks and create index
        index, chunk_ids = self.process_chunks(db_path)
        
        # Save index and metadata
        self.index_builder.save_index(index_path, meta_path, db_path)
        
        logger.info("Index building completed successfully")

def main():
    """Main function for building embeddings and index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS index")
    parser.add_argument("--db_path", type=Path, default=CHUNKS_DB_PATH, help="Path to chunks database")
    parser.add_argument("--index_path", type=Path, default=FAISS_INDEX_PATH, help="Path to save FAISS index")
    parser.add_argument("--meta_path", type=Path, default=FAISS_META_PATH, help="Path to save metadata")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Build index
    service = EmbeddingService()
    service.build_and_save_index(args.db_path, args.index_path, args.meta_path)
    
    print("Embedding generation and index building completed!")

if __name__ == "__main__":
    main()
