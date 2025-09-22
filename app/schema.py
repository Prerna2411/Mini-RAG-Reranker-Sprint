"""
Pydantic models for API request and response schemas.
"""
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class SearchMode(str, Enum):
    """Search mode options."""
    BASELINE = "baseline"
    RERANK = "rerank"

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    q: str = Field(..., description="Question to ask", min_length=1, max_length=1000)
    k: int = Field(default=5, description="Number of contexts to retrieve", ge=1, le=20)
    mode: SearchMode = Field(default=SearchMode.RERANK, description="Search mode")
    
    @validator('q')
    def validate_question(cls, v):
        """Validate and clean question."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class Context(BaseModel):
    """Individual context chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Source URL")
    text: str = Field(..., description="Chunk text content")
    vector_score: float = Field(..., description="Vector similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword score")
    blended_score: Optional[float] = Field(None, description="Final blended score")
    doc_id: str = Field(..., description="Document identifier")
    chunk_idx: int = Field(..., description="Chunk index within document")
    char_start: int = Field(..., description="Character start position")
    char_end: int = Field(..., description="Character end position")

class Thresholds(BaseModel):
    """Abstention thresholds."""
    blended_min: float = Field(..., description="Minimum blended score threshold")
    vector_min: float = Field(..., description="Minimum vector score threshold")
    bm25_min: float = Field(..., description="Minimum BM25 score threshold")

class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: Optional[str] = Field(None, description="Generated answer or null if abstained")
    contexts: List[Context] = Field(..., description="Retrieved context chunks")
    reranker_used: str = Field(..., description="Type of reranker used")
    abstained: bool = Field(..., description="Whether the system abstained from answering")
    abstention_reason: Optional[str] = Field(None, description="Reason for abstention")
    thresholds: Thresholds = Field(..., description="Applied thresholds")
    query: str = Field(..., description="Original query")
    k: int = Field(..., description="Number of contexts requested")
    mode: str = Field(..., description="Search mode used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the embedding model is loaded")
    index_loaded: bool = Field(..., description="Whether the FAISS index is loaded")
    db_connected: bool = Field(..., description="Whether the database is connected")
    total_chunks: int = Field(..., description="Total number of chunks in database")
    index_size: int = Field(..., description="Number of vectors in FAISS index")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
