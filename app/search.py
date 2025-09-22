"""
Baseline vector search functionality using FAISS.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from .config import FAISS_TOP_K, EMBEDDING_DIM
from .utils import get_db_connection, load_embedding_model, load_faiss_index

logger = logging.getLogger(__name__)

class VectorSearch:
    """Baseline vector search using FAISS."""
    
    def __init__(self, model: SentenceTransformer, index: faiss.Index, metadata: Dict[str, Any]):
        self.model = model
        self.index = index
        self.metadata = metadata
        self.dimension = EMBEDDING_DIM
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        try:
            # Encode query
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(k, FAISS_TOP_K))
            
            # Get chunk details from database
            contexts = self._get_contexts_from_indices(indices[0], scores[0])
            
            return contexts[:k]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _get_contexts_from_indices(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
        """Get context details for the given indices."""
        contexts = []
        
        # Get chunk IDs from metadata
        chunk_ids = self.metadata.get('chunk_ids', [])
        with get_db_connection(Path(self.metadata['db_path'])) as conn:
        #with get_db_connection(self.metadata['db_path']) as conn:
            cursor = conn.cursor()
            
            for idx, score in zip(indices, scores):
                if idx == -1:  # Invalid index
                    continue
                
                if idx >= len(chunk_ids):
                    continue
                
                chunk_id = chunk_ids[idx]
                
                # Get chunk details
                cursor.execute("""
                    SELECT chunk_id, doc_id, title, chunk_idx, text, source_url, char_start, char_end
                    FROM chunks WHERE chunk_id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    contexts.append({
                        'chunk_id': row['chunk_id'],
                        'doc_id': row['doc_id'],
                        'title': row['title'],
                        'chunk_idx': row['chunk_idx'],
                        'text': row['text'],
                        'url': row['source_url'],
                        'char_start': row['char_start'],
                        'char_end': row['char_end'],
                        'vector_score': float(score)
                    })
        
        return contexts

class SearchService:
    """Main search service that coordinates vector search."""
    
    def __init__(self, db_path: str, index_path: str, meta_path: str):
        self.db_path = db_path
        self.index_path = index_path
        self.meta_path = meta_path
        
        # Load components
        self.model = load_embedding_model()
        self.index, self.metadata = load_faiss_index(index_path, meta_path)
        self.vector_search = VectorSearch(self.model, self.index, self.metadata)
        
        logger.info("Search service initialized")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search."""
        return self.vector_search.search(query, k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        return {
            "total_chunks": self.index.ntotal,
            "embedding_dim": self.dimension,
            "model_name": self.metadata.get('model_name', 'unknown'),
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            #"model_name": self.model.get_sentence_embedding_dimension(),
            "index_type": type(self.index).__name__
        }
