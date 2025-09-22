"""
Hybrid reranking functionality combining vector and BM25 scores.
"""
import logging
import sqlite3
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from .config import BM25_TOP_K, RERANK_ALPHA
from .utils import get_db_connection,blend_scores,normalize_scores

logger = logging.getLogger(__name__)

class HybridReranker:
    """Hybrid reranker combining vector and BM25 scores."""
    
    def __init__(self, db_path: str, alpha: float = RERANK_ALPHA):
        self.db_path = db_path
        self.alpha = alpha
    
    def rerank(self, vector_contexts: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Rerank contexts using hybrid scoring."""
        try:
            # Get BM25 candidates
            bm25_contexts = self._get_bm25_candidates(query, BM25_TOP_K)
            
            # Merge and deduplicate contexts
            merged_contexts = self._merge_contexts(vector_contexts, bm25_contexts)
            
            # Calculate blended scores
            reranked_contexts = self._calculate_blended_scores(merged_contexts, query)
            
            # Sort by blended score
            reranked_contexts.sort(key=lambda x: x.get('blended_score', 0), reverse=True)
            
            return reranked_contexts[:k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return vector_contexts[:k]  # Fallback to original order
    
    def _get_bm25_candidates(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Get BM25 candidates using SQLite FTS5."""
        contexts = []
    
        with get_db_connection(Path(self.db_path)) as conn:
            cursor = conn.cursor()
        
            # Escape special FTS5 characters and prepare query
            fts_query = self._prepare_fts_query(query)
        
            try:
            # Use FTS5 BM25 scoring
                cursor.execute("""
                SELECT 
                    c.chunk_id, c.doc_id, c.title, c.chunk_idx, c.text, c.source_url, 
                    c.char_start, c.char_end,
                    bm25(chunks_fts) as bm25_score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """, (fts_query, top_k))
            
                for row in cursor.fetchall():
                    contexts.append({
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'title': row['title'],
                    'chunk_idx': row['chunk_idx'],
                    'text': row['text'],
                    'url': row['source_url'],
                    'char_start': row['char_start'],
                    'char_end': row['char_end'],
                    'bm25_score': float(row['bm25_score'])
                    })
                
            except Exception as e:
                logger.warning(f"FTS5 search failed with query '{fts_query}': {e}")
            # Fallback to simple LIKE search
                contexts = self._fallback_text_search(query, top_k, conn)
    
        return contexts
    
    def _merge_contexts(self, vector_contexts: List[Dict[str, Any]], bm25_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate contexts from both sources."""
        merged = {}
        
        # Add vector contexts
        for ctx in vector_contexts:
            chunk_id = ctx['chunk_id']
            merged[chunk_id] = ctx.copy()
        
        # Add BM25 contexts, merging scores if already present
        for ctx in bm25_contexts:
            chunk_id = ctx['chunk_id']
            if chunk_id in merged:
                merged[chunk_id]['bm25_score'] = ctx['bm25_score']
            else:
                merged[chunk_id] = ctx.copy()
                merged[chunk_id]['vector_score'] = 0.0  # Default for BM25-only results
        
        return list(merged.values())
    
    def debug_fts_search(self, query: str):
        """Debug FTS5 search issues."""
        with get_db_connection(Path(self.db_path)) as conn:
            cursor = conn.cursor()
        
        # Check if FTS table exists and has data
            cursor.execute("SELECT count(*) FROM chunks_fts")
            fts_count = cursor.fetchone()[0]
            print(f"FTS5 table has {fts_count} rows")
        
        # Prepare the query
            fts_query = self._prepare_fts_query(query)
            print(f"Original query: '{query}'")
            print(f"FTS query: '{fts_query}'")
        
            try:
            # Try FTS search
                cursor.execute("SELECT rowid, text FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 3", (fts_query,))
                results = cursor.fetchall()
                print(f"FTS search returned {len(results)} results")
            
                for row in results:
                    print(f"Row {row[0]}: {row[1][:100]}...")
            except Exception as e:
                print(f"FTS search failed: {e}")
            # Try fallback
                fallback_results = self._fallback_text_search(query, 3, conn)
                print(f"Fallback search returned {len(fallback_results)} results")

    def _calculate_blended_scores(self, contexts: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Calculate blended scores for contexts."""
        if not contexts:
            return contexts
        
        # Extract scores
        vector_scores = [ctx.get('vector_score', 0.0) for ctx in contexts]
        bm25_scores = [ctx.get('bm25_score', 0.0) for ctx in contexts]
        
        # Normalize scores
        norm_vector_scores = normalize_scores(vector_scores)
        norm_bm25_scores = normalize_scores(bm25_scores)
        
        # Calculate blended scores
        blended_scores = blend_scores(norm_vector_scores, norm_bm25_scores, self.alpha)
        
        # Add scores to contexts
        for i, ctx in enumerate(contexts):
            ctx['vector_score'] = norm_vector_scores[i]
            ctx['bm25_score'] = norm_bm25_scores[i]
            ctx['blended_score'] = blended_scores[i]
        
        return contexts
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query for FTS5 by handling special characters."""
    # Remove or escape problematic characters
        query = query.replace('-', ' ')  # Replace hyphens with spaces
        query = query.replace('"', '""')  # Escape quotes
    
    # Split into words and create phrase search
        words = query.split()
        if len(words) == 1:
            return f'"{words[0]}"'
        else:
        # Use AND to search for all terms
            return ' AND '.join(f'"{word}"' for word in words if len(word) >= 1)

    def _fallback_text_search(self, query: str, top_k: int, conn) -> List[Dict[str, Any]]:
        """Fallback text search using LIKE when FTS5 fails."""
        contexts = []
        cursor = conn.cursor()
    
    # Simple LIKE search as fallback
        search_terms = query.split()
        like_conditions = ' OR '.join(['text LIKE ?'] * len(search_terms))
        like_params = [f'%{term}%' for term in search_terms]
    
        cursor.execute(f"""
        SELECT chunk_id, doc_id, title, chunk_idx, text, source_url, char_start, char_end,
               1.0 as bm25_score
        FROM chunks 
        WHERE {like_conditions}
        LIMIT ?
        """, like_params + [top_k])
    
        for row in cursor.fetchall():
            contexts.append({
            'chunk_id': row['chunk_id'],
            'doc_id': row['doc_id'],
            'title': row['title'],
            'chunk_idx': row['chunk_idx'],
            'text': row['text'],
            'url': row['source_url'],
            'char_start': row['char_start'],
            'char_end': row['char_end'],
            'bm25_score': 1.0  # Default score for fallback
            })
    
        return contexts

class RerankService:
    """Main reranking service."""
    
    def __init__(self, db_path: str, alpha: float = RERANK_ALPHA):
        self.db_path = db_path
        self.reranker = HybridReranker(db_path, alpha)
        logger.info(f"Rerank service initialized with alpha={alpha}")
    
    def rerank(self, vector_contexts: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Rerank contexts using hybrid approach."""
        return self.reranker.rerank(vector_contexts, query, k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking service statistics."""
        return {
            "alpha": self.alpha,
            "bm25_top_k": BM25_TOP_K,
            "reranker_type": "hybrid"
        }
