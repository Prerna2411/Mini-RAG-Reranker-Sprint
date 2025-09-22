"""
FastAPI application with RAG endpoints.
"""
import time
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import (
    API_HOST, API_PORT, API_TITLE, API_VERSION, 
    CHUNKS_DB_PATH, FAISS_INDEX_PATH, FAISS_META_PATH
)
from .schema import (
    AskRequest, AskResponse, HealthResponse, ErrorResponse,
    SearchMode, Thresholds
)
from .search import SearchService
from .rerank import RerankService
from .answer import AnswerService
from .utils import setup_logging, get_system_info

# Setup logging
logger = setup_logging()

# Global services
search_service = None
rerank_service = None
answer_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global search_service, rerank_service, answer_service
    
    # Startup
    logger.info("Starting Industrial Safety RAG API...")
    
    try:
        # Initialize services
        search_service = SearchService(
            db_path=str(CHUNKS_DB_PATH),
            index_path=str(FAISS_INDEX_PATH),
            meta_path=str(FAISS_META_PATH)
        )
        
        rerank_service = RerankService(db_path=str(CHUNKS_DB_PATH))
        answer_service = AnswerService()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Industrial Safety RAG API...")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_search_service() -> SearchService:
    """Dependency to get search service."""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service not available")
    return search_service

def get_rerank_service() -> RerankService:
    """Dependency to get rerank service."""
    if rerank_service is None:
        raise HTTPException(status_code=503, detail="Rerank service not available")
    return rerank_service

def get_answer_service() -> AnswerService:
    """Dependency to get answer service."""
    if answer_service is None:
        raise HTTPException(status_code=503, detail="Answer service not available")
    return answer_service

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Industrial Safety RAG API",
        "version": API_VERSION,
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are loaded
        model_loaded = search_service is not None
        index_loaded = search_service is not None and search_service.index.ntotal > 0
        db_connected = search_service is not None
        
        # Get stats
        total_chunks = 0
        index_size = 0
        
        if search_service:
            stats = search_service.get_stats()
            total_chunks = stats.get('total_chunks', 0)
            index_size = stats.get('total_chunks', 0)
        
        return HealthResponse(
            status="healthy" if all([model_loaded, index_loaded, db_connected]) else "degraded",
            version=API_VERSION,
            model_loaded=model_loaded,
            index_loaded=index_loaded,
            db_connected=db_connected,
            total_chunks=total_chunks,
            index_size=index_size
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=API_VERSION,
            model_loaded=False,
            index_loaded=False,
            db_connected=False,
            total_chunks=0,
            index_size=0
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    search_svc: SearchService = Depends(get_search_service),
    rerank_svc: RerankService = Depends(get_rerank_service),
    answer_svc: AnswerService = Depends(get_answer_service)
):
    """Main question-answering endpoint."""
    start_time = time.time()
    
    try:
        # Perform search
        if request.mode == SearchMode.BASELINE:
            contexts = search_svc.search(request.q, request.k)
            reranker_used = "none"
        else:
            # Get more contexts for reranking
            vector_contexts = search_svc.search(request.q, min(request.k * 3, 20))
            contexts = rerank_svc.rerank(vector_contexts, request.q, request.k)
            reranker_used = "hybrid"
        
        # Generate answer
        answer, abstained, abstention_reason, processed_contexts = answer_svc.generate_answer(
            contexts, request.q
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Get thresholds
        thresholds = answer_svc.get_thresholds()
        
        return AskResponse(
            answer=answer,
            contexts=processed_contexts,
            reranker_used=reranker_used,
            abstained=abstained,
            abstention_reason=abstention_reason,
            thresholds=Thresholds(**thresholds),
            query=request.q,
            k=request.k,
            mode=request.mode.value,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(
    search_svc: SearchService = Depends(get_search_service),
    rerank_svc: RerankService = Depends(get_rerank_service),
    answer_svc: AnswerService = Depends(get_answer_service)
):
    """Get system statistics."""
    try:
        search_stats = search_svc.get_stats()
        rerank_stats = rerank_svc.get_stats()
        answer_stats = answer_svc.get_thresholds()
        system_info = get_system_info()
        
        return {
            "search": search_stats,
            "reranking": rerank_stats,
            "answer": answer_stats,
            "system": system_info
        }
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
