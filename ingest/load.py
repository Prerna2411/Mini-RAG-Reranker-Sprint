"""
Load chunks into SQLite database with FTS5 support.
"""
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any


from app.config import CHUNKS_DB_PATH
from app.utils import setup_logging, create_chunks_table

logger = setup_logging()

class ChunkLoader:
    """Load chunks into SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database and create tables."""
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            create_chunks_table(conn)
        logger.info(f"Initialized database at {self.db_path}")
    
    def load_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Load chunks into the database."""
        if not chunks:
            logger.warning("No chunks to load")
            return
        
        logger.info(f"Loading {len(chunks)} chunks into database")
        
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM chunks_fts")
            
            # Insert chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks (
                        chunk_id, doc_id, title, chunk_idx, text, 
                        source_url, char_start, char_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk['chunk_id'],
                    chunk['doc_id'],
                    chunk['title'],
                    chunk['chunk_idx'],
                    chunk['text'],
                    chunk['source_url'],
                    chunk['char_start'],
                    chunk['char_end']
                ))
            
            conn.commit()
            logger.info(f"Successfully loaded {len(chunks)} chunks")
    
    def load_chunks_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 1000) -> None:
        """Load chunks into the database in batches for memory efficiency."""
        if not chunks:
            logger.warning("No chunks to load")
            return
        
        logger.info(f"Loading {len(chunks)} chunks into database in batches of {batch_size}")
        
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM chunks_fts")
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Loading batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
                
                # Prepare batch data
                batch_data = []
                for chunk in batch:
                    batch_data.append((
                        chunk['chunk_id'],
                        chunk['doc_id'],
                        chunk['title'],
                        chunk['chunk_idx'],
                        chunk['text'],
                        chunk['source_url'],
                        chunk['char_start'],
                        chunk['char_end']
                    ))
                
                # Insert batch
                cursor.executemany("""
                    INSERT INTO chunks (
                        chunk_id, doc_id, title, chunk_idx, text, 
                        source_url, char_start, char_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                
                # Memory management
                import gc
                gc.collect()
            
            logger.info(f"Successfully loaded {len(chunks)} chunks in batches")
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
    
    def get_document_count(self) -> int:
        """Get total number of documents in database."""
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            cursor = conn.cursor()
            
            # Get chunk count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Get document count
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks")
            doc_count = cursor.fetchone()[0]
            
            # Get average chunk length
            cursor.execute("SELECT AVG(LENGTH(text)) FROM chunks")
            avg_length = cursor.fetchone()[0] or 0
            
            # Get total text length
            cursor.execute("SELECT SUM(LENGTH(text)) FROM chunks")
            total_length = cursor.fetchone()[0] or 0
            
            return {
                "total_chunks": chunk_count,
                "total_documents": doc_count,
                "average_chunk_length": round(avg_length, 2),
                "total_text_length": total_length
            }

def main():
    """Main function for loading chunks."""
    import argparse
    from ingest.chunk import DocumentProcessor
    
    parser = argparse.ArgumentParser(description="Load chunks into database")
    parser.add_argument("--raw_dir", type=Path, help="Directory containing PDF files")
    parser.add_argument("--sources", type=Path, help="Path to sources.json file")
    parser.add_argument("--db_path", type=Path, default=CHUNKS_DB_PATH, help="Path to database file")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Process documents if raw_dir is provided
    if args.raw_dir:
        sources_path = args.sources or (Path(__file__).parent.parent / "data" / "sources.json")
        processor = DocumentProcessor(sources_path)
        chunks_data = processor.process_all_documents(args.raw_dir)
        
        # Convert chunks to dictionaries
        chunks = []
        for chunk in chunks_data:
            chunks.append({
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'title': chunk.title,
                'chunk_idx': chunk.chunk_idx,
                'text': chunk.text,
                'source_url': chunk.source_url,
                'char_start': chunk.char_start,
                'char_end': chunk.char_end
            })
    else:
        # Load from existing data
        chunks = []
        logger.warning("No raw_dir provided, loading empty database")
    
    # Load chunks into database
    loader = ChunkLoader(args.db_path)
    loader.load_chunks(chunks)
    
    # Print statistics
    stats = loader.get_stats()
    print(f"Database statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Average chunk length: {stats['average_chunk_length']} characters")
    print(f"  Total text length: {stats['total_text_length']} characters")

if __name__ == "__main__":
    main()
