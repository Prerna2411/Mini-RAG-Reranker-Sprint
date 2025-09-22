"""
Answer generation and abstention logic.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from .config import BLENDED_THRESHOLD, VECTOR_THRESHOLD, BM25_THRESHOLD
from .utils import should_abstain, format_citation

logger = logging.getLogger(__name__)

def extract_relevant_sentences(text: str, query: str, max_sentences: int = 2) -> str:
    """
    Extract sentences from text that contain keywords from the query.
    Returns a concise, extractive answer.
    """
    keywords = query.lower().split()
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant = []

    for sent in sentences:
        if any(k in sent.lower() for k in keywords):
            relevant.append(sent.strip())
        if len(relevant) >= max_sentences:
            break

    return " ".join(relevant)


class AnswerGenerator:
    """Generate answers from retrieved contexts."""
    
    def __init__(
        self, 
        blended_threshold: float = BLENDED_THRESHOLD,
        vector_threshold: float = VECTOR_THRESHOLD,
        bm25_threshold: float = BM25_THRESHOLD
    ):
        self.blended_threshold = blended_threshold
        self.vector_threshold = vector_threshold
        self.bm25_threshold = bm25_threshold
    
    def generate_answer(
        self, 
        contexts: List[Dict[str, Any]], 
        query: str
    ) -> Tuple[Optional[str], bool, Optional[str], List[Dict[str, Any]]]:
        """
        Generate answer from contexts.

        Returns:
            answer: Generated answer or None if abstained
            abstained: Whether the system abstained
            abstention_reason: Reason for abstention if applicable
            processed_contexts: Contexts with additional metadata
        """
        if not contexts:
            return None, True, "No contexts retrieved", []

        # Check if we should abstain
        abstained, abstention_reason = should_abstain(
            contexts, 
            self.blended_threshold, 
            self.vector_threshold, 
            self.bm25_threshold
        )
        
        if abstained:
            return None, True, abstention_reason, contexts

        # Compose a query-focused answer
        answer = self._compose_answer(contexts, query)
        
        # Add citations to contexts
        processed_contexts = self._add_citations(contexts)

        return answer, False, None, processed_contexts
    
    def _compose_answer(self, contexts: List[Dict[str, Any]], query: str) -> str:
        """Compose a concise answer from the top contexts."""
        if not contexts:
            return ""

        # Use top context first
        top_context = contexts[0]
        answer = extract_relevant_sentences(top_context['text'], query)

        # Add relevant sentences from other contexts if needed
        for ctx in contexts[1:]:
            additional = extract_relevant_sentences(ctx['text'], query)
            if additional and additional not in answer:
                answer += f" Additionally, {additional}"

        # Append source information inline
        answer += f" [Source: {top_context['title']}]"
        return answer.strip()
    
    def _add_citations(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add citation information to contexts."""
        processed = []
        for i, ctx in enumerate(contexts):
            processed_ctx = ctx.copy()
            processed_ctx['citation'] = format_citation(ctx)
            processed_ctx['rank'] = i + 1
            processed.append(processed_ctx)
        return processed


class AnswerService:
    """Main answer generation service."""
    
    def __init__(
        self, 
        blended_threshold: float = BLENDED_THRESHOLD,
        vector_threshold: float = VECTOR_THRESHOLD,
        bm25_threshold: float = BM25_THRESHOLD
    ):
        self.generator = AnswerGenerator(blended_threshold, vector_threshold, bm25_threshold)
        self.thresholds = {
            'blended_min': blended_threshold,
            'vector_min': vector_threshold,
            'bm25_min': bm25_threshold
        }
        logger.info("Answer service initialized")
    
    def generate_answer(
        self, 
        contexts: List[Dict[str, Any]], 
        query: str
    ) -> Tuple[Optional[str], bool, Optional[str], List[Dict[str, Any]]]:
        """Generate answer from contexts."""
        return self.generator.generate_answer(contexts, query)
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return self.thresholds.copy()
    
    def update_thresholds(
        self, 
        blended_threshold: Optional[float] = None,
        vector_threshold: Optional[float] = None,
        bm25_threshold: Optional[float] = None
    ) -> None:
        """Update thresholds."""
        if blended_threshold is not None:
            self.thresholds['blended_min'] = blended_threshold
            self.generator.blended_threshold = blended_threshold
        
        if vector_threshold is not None:
            self.thresholds['vector_min'] = vector_threshold
            self.generator.vector_threshold = vector_threshold
        
        if bm25_threshold is not None:
            self.thresholds['bm25_min'] = bm25_threshold
            self.generator.bm25_threshold = bm25_threshold
        
        logger.info(f"Updated thresholds: {self.thresholds}")
