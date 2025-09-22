"""
Complete ingestion pipeline for processing PDFs and creating the RAG system.
"""
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any
import argparse
import traceback

from app.config import (
    RAW_DIR, CHUNKS_DB_PATH, FAISS_INDEX_PATH, FAISS_META_PATH, 
    SOURCES_JSON_PATH
)
from app.utils import setup_logging, log_memory_usage
from .chunk import DocumentProcessor
from .load import ChunkLoader
from .embed import EmbeddingService

logger = setup_logging()

poppler_path = r"E:\Poppler\poppler-25.07.0\Library\bin"
import os
if poppler_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + poppler_path


class IngestionPipeline:
    """Complete pipeline for ingesting documents and creating the RAG system."""
    
    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        sources_path: Path = SOURCES_JSON_PATH,
        db_path: Path = CHUNKS_DB_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH
    ):
        self.raw_dir = raw_dir
        self.sources_path = sources_path
        self.db_path = db_path
        self.index_path = index_path
        self.meta_path = meta_path
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with:")
        logger.info(f"  Raw dir: {self.raw_dir}")
        logger.info(f"  Sources: {self.sources_path}")
        logger.info(f"  DB path: {self.db_path}")
    
    def run(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Run the complete ingestion pipeline."""
        logger.info("Starting ingestion pipeline")
        
        try:
            # Check if already built
            if not force_rebuild and self._is_already_built():
                logger.info("System already built. Use --force to rebuild.")
                return self._get_build_stats()
            
            # Step 1: Process documents and create chunks
            logger.info("=" * 60)
            logger.info("STEP 1: Processing documents and creating chunks")
            logger.info("=" * 60)
            log_memory_usage(logger, "before processing")
            
            # Create processor
            logger.info("Creating DocumentProcessor instance...")
            try:
                processor = DocumentProcessor(self.sources_path, tesseract_path=None)
                logger.info("✅ DocumentProcessor created successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create DocumentProcessor: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Process all documents
            logger.info("Starting document processing...")
            try:
                chunks_data = processor.process_all_documents(self.raw_dir)
                logger.info(f"✅ Document processing completed. Chunks returned: {len(chunks_data) if chunks_data else 0}")
            except Exception as e:
                logger.error(f"❌ Document processing failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Clean up processor
            try:
                del processor
                gc.collect()
                logger.info("✅ Processor cleanup completed")
            except Exception as e:
                logger.warning(f"⚠️  Processor cleanup warning: {e}")
            
            log_memory_usage(logger, "after processing")
            
            # Validate chunks_data
            if not chunks_data:
                logger.error("❌ No chunks created from documents")
                raise ValueError("No chunks created from documents")
            
            if not isinstance(chunks_data, list):
                logger.error(f"❌ chunks_data is not a list, it's: {type(chunks_data)}")
                raise ValueError(f"Expected list of chunks, got {type(chunks_data)}")
            
            logger.info(f"✅ Chunks validation passed: {len(chunks_data)} chunks")
            
            # Convert to dictionaries
            logger.info("Converting chunks to dictionary format...")
            chunks = []
            try:
                for i, chunk in enumerate(chunks_data):
                    if i < 3:  # Log first 3 chunks for debugging
                        logger.info(f"Processing chunk {i}: {type(chunk)}")
                        logger.info(f"  Chunk attributes: {dir(chunk) if hasattr(chunk, '__dict__') else 'No attributes'}")
                    
                    chunk_dict = {
                        'chunk_id': chunk.chunk_id,
                        'doc_id': chunk.doc_id,
                        'title': chunk.title,
                        'chunk_idx': chunk.chunk_idx,
                        'text': chunk.text,
                        'source_url': chunk.source_url,
                        'char_start': chunk.char_start,
                        'char_end': chunk.char_end
                    }
                    chunks.append(chunk_dict)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Converted {i + 1}/{len(chunks_data)} chunks")
                
                logger.info(f"✅ Successfully converted {len(chunks)} chunks to dictionary format")
                
            except Exception as e:
                logger.error(f"❌ Failed to convert chunks to dictionaries: {e}")
                logger.error(traceback.format_exc())
                if chunks_data:
                    logger.error(f"Sample chunk type: {type(chunks_data[0])}")
                    logger.error(f"Sample chunk content: {chunks_data[0]}")
                raise
            
            # Clean up chunks_data
            del chunks_data
            gc.collect()
            
            # Step 2: Load chunks into database in batches
            logger.info("=" * 60)
            logger.info("STEP 2: Loading chunks into database")
            logger.info("=" * 60)
            log_memory_usage(logger, "before database loading")
            
            try:
                loader = ChunkLoader(self.db_path)
                logger.info(f"Loading {len(chunks)} chunks into database...")
                loader.load_chunks_batch(chunks, batch_size=1000)
                logger.info("✅ Database loading completed")
            except Exception as e:
                logger.error(f"❌ Database loading failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            log_memory_usage(logger, "after database loading")
            
            # Step 3: Generate embeddings and build FAISS index
            logger.info("=" * 60)
            logger.info("STEP 3: Generating embeddings and building FAISS index")
            logger.info("=" * 60)
            log_memory_usage(logger, "before embedding generation")
            
            try:
                embedding_service = EmbeddingService()
                logger.info("Building FAISS index...")
                embedding_service.build_and_save_index(
                    self.db_path, 
                    self.index_path, 
                    self.meta_path
                )
                logger.info("✅ FAISS index building completed")
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            log_memory_usage(logger, "after embedding generation")
            
            # Get final statistics
            logger.info("=" * 60)
            logger.info("GETTING FINAL STATISTICS")
            logger.info("=" * 60)
            try:
                stats = self._get_build_stats()
                logger.info("✅ Statistics retrieved successfully")
                logger.info("🎉 INGESTION PIPELINE COMPLETED SUCCESSFULLY 🎉")
            except Exception as e:
                logger.error(f"❌ Failed to get build stats: {e}")
                logger.error(traceback.format_exc())
                raise
            
            return stats
            
        except Exception as e:
            logger.error(f"💥 INGESTION PIPELINE FAILED: {e}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise
    
    def _is_already_built(self) -> bool:
        """Check if the system is already built."""
        built = (
            self.db_path.exists() and 
            self.index_path.exists() and 
            self.meta_path.exists()
        )
        logger.info(f"System already built check: {built}")
        if built:
            logger.info(f"  DB exists: {self.db_path.exists()}")
            logger.info(f"  Index exists: {self.index_path.exists()}")
            logger.info(f"  Meta exists: {self.meta_path.exists()}")
        return built
    
    def _get_build_stats(self) -> Dict[str, Any]:
        """Get statistics about the built system."""
        stats = {}
        
        try:
            # Database stats
            if self.db_path.exists():
                loader = ChunkLoader(self.db_path)
                db_stats = loader.get_stats()
                stats.update(db_stats)
                logger.info(f"Database stats: {db_stats}")
            
            # Index stats
            if self.index_path.exists() and self.meta_path.exists():
                import json
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                index_stats = {
                    'index_vectors': meta.get('total_vectors', 0),
                    'model_name': meta.get('model_name', 'unknown'),
                    'embedding_dimension': meta.get('dimension', 0)
                }
                stats.update(index_stats)
                logger.info(f"Index stats: {index_stats}")
                
        except Exception as e:
            logger.error(f"Error getting build stats: {e}")
            logger.error(traceback.format_exc())
        
        return stats

def main():
    """Main function for running the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Run complete ingestion pipeline")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="Directory containing PDF files")
    parser.add_argument("--sources", type=Path, default=SOURCES_JSON_PATH, help="Path to sources.json file")
    parser.add_argument("--db_path", type=Path, default=CHUNKS_DB_PATH, help="Path to database file")
    parser.add_argument("--index_path", type=Path, default=FAISS_INDEX_PATH, help="Path to FAISS index file")
    parser.add_argument("--meta_path", type=Path, default=FAISS_META_PATH, help="Path to metadata file")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if already exists")
    args = parser.parse_args()
    
    # Setup logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run pipeline
        pipeline = IngestionPipeline(
            raw_dir=args.raw_dir,
            sources_path=args.sources,
            db_path=args.db_path,
            index_path=args.index_path,
            meta_path=args.meta_path
        )
        
        stats = pipeline.run(force_rebuild=args.force)
        
        # Print results
        print("\n" + "="*50)
        print("🎉 INGESTION PIPELINE COMPLETED 🎉")
        print("="*50)
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        print(f"Total documents: {stats.get('total_documents', 0)}")
        print(f"Index vectors: {stats.get('index_vectors', 0)}")
        print(f"Model: {stats.get('model_name', 'unknown')}")
        print(f"Embedding dimension: {stats.get('embedding_dimension', 0)}")
        print(f"Average chunk length: {stats.get('average_chunk_length', 0)} characters")
        print("="*50)
        
    except Exception as e:
        print(f"\n💥 PIPELINE FAILED: {e}")
        print(f"Full error: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()