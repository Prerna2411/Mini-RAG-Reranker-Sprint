"""
PDF parsing and text chunking functionality with OCR support.
"""
import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import fitz  # PyMuPDF
from dataclasses import dataclass
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, RAW_DIR
from app.utils import clean_text, chunk_text, chunk_text_streaming, generate_chunk_id, load_sources

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk."""
    chunk_id: str
    doc_id: str
    title: str
    chunk_idx: int
    text: str
    source_url: str
    char_start: int
    char_end: int

class PDFParser:
    """PDF parsing utilities with OCR support."""
    
    def __init__(self, tesseract_path: str = None):
        """Initialize with optional Tesseract path."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try common Windows paths
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
                    os.getenv('USERNAME', '')
                )
            ]
            for path in common_paths:
                if Path(path).exists():
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    @staticmethod
    def extract_text_pypdf2(pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_pymupdf(pdf_path: Path) -> str:
        """Extract text using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_ocr(self, pdf_path: Path) -> str:
        import time
        """Extract text using OCR from scanned PDFs with memory optimization."""
        try:
            logger.info(f"Using OCR to extract text from {pdf_path}")
            
            # Convert PDF to images with lower DPI to save memory
            images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=None)
            
            text = ""
            total_pages = len(images)
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{total_pages}")
                
                # Convert PIL image to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text using OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += page_text + "\n"
                
                # Memory management: clear image from memory
                del image
                
                # Force garbage collection every 10 pages
                if (i + 1) % 10 == 0:
                    import gc
                    gc.collect()
                    logger.info(f"Memory cleanup after page {i+1}")
            
            logger.info(f"OCR extracted {len(text)} characters from {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using the best available method."""
        # Try PyMuPDF first (usually better)
        text = self.extract_text_pymupdf(pdf_path)
        
        # If no text or very little text, try PyPDF2
        if not text or len(text.strip()) < 100:
            text = self.extract_text_pypdf2(pdf_path)
        
        # If still no text or very little text, use OCR
        if not text or len(text.strip()) < 100:
            logger.info(f"Text extraction failed, trying OCR for {pdf_path}")
            text = self.extract_text_ocr(pdf_path)
        
        return clean_text(text)




class DocumentProcessor:
    """Process documents and create chunks."""
    
    def __init__(self, sources_path: Path, tesseract_path: str = None):
        self.sources = load_sources(sources_path)
        self.parser = PDFParser(tesseract_path)
        
        # Set up Poppler path if not already in PATH
        poppler_path = r"E:\Poppler\poppler-25.07.0\Library\bin"
        import os
        if poppler_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + poppler_path
        
        logger.info("DocumentProcessor initialized successfully")
    
    def process_document(self, pdf_path: Path) -> List[Chunk]:
        """Process a single PDF document with detailed debugging."""
        doc_id = pdf_path.stem
        
        # Get document info from sources
        doc_info = self.sources.get(doc_id, {})
        title = doc_info.get('title', doc_id)
        url = doc_info.get('url', '')
        
        logger.info(f"Processing document: {title}")
        logger.info(f"Document ID: {doc_id}")
        
        try:
            # Extract text using OCR (since your logs show OCR is working)
            logger.info("Starting text extraction...")
            text = self.parser.extract_text_ocr(pdf_path)  # Use OCR directly since it's working
            logger.info(f"Text extraction completed. Length: {len(text) if text else 0}")
            
            if not text:
                logger.warning(f"No text extracted from {pdf_path}")
                return []
            
            if len(text.strip()) < MIN_CHUNK_SIZE:
                logger.warning(f"Text too short ({len(text.strip())} chars) from {pdf_path}, min required: {MIN_CHUNK_SIZE}")
                return []
            
            logger.info(f"Text validation passed. Starting chunking...")
            
            # Import chunking functions
            try:
                from app.utils import chunk_text, generate_chunk_id
                logger.info("Successfully imported chunking functions")
            except ImportError as e:
                logger.error(f"Failed to import chunking functions: {e}")
                # Fallback chunking if imports fail
                return self._fallback_chunking(text, doc_id, title, url)
            
            # Chunk the text
            try:
                text_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                logger.info(f"Chunking completed. Created {len(text_chunks)} text chunks")
            except Exception as e:
                logger.error(f"Chunking failed: {e}")
                return self._fallback_chunking(text, doc_id, title, url)
            
            # Create Chunk objects
            chunks = []
            logger.info("Creating Chunk objects...")
            
            for i, chunk_data in enumerate(text_chunks):
                try:
                    chunk_id = generate_chunk_id(doc_id, i)
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        chunk_idx=i,
                        text=chunk_data['text'],
                        source_url=url,
                        char_start=chunk_data['start'],
                        char_end=chunk_data['end']
                    )
                    chunks.append(chunk)
                    
                    if i < 3:  # Log first 3 chunks for debugging
                        logger.info(f"Created chunk {i}: ID={chunk_id}, text_length={len(chunk_data['text'])}")
                    
                except Exception as e:
                    logger.error(f"Failed to create chunk {i}: {e}")
                    continue
            
            logger.info(f"Successfully created {len(chunks)} Chunk objects for {title}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in process_document: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _fallback_chunking(self, text: str, doc_id: str, title: str, url: str) -> List[Chunk]:
        """Fallback chunking method if imports fail."""
        logger.info("Using fallback chunking method")
        
        chunks = []
        chunk_size = 1000  # Default chunk size
        overlap = 200      # Default overlap
        
        # Simple chunking
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
                
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                chunk_idx=len(chunks),
                text=chunk_text,
                source_url=url,
                char_start=i,
                char_end=i + len(chunk_text)
            )
            chunks.append(chunk)
        
        logger.info(f"Fallback chunking created {len(chunks)} chunks")
        return chunks
    
    def process_all_documents(self, raw_dir: Path) -> List[Chunk]:
        """Process all PDF documents in the raw directory with progress tracking."""
        all_chunks = []
        
        pdf_files = list(raw_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {raw_dir}")
            return all_chunks
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing document {i}/{len(pdf_files)}: {pdf_path.name}")
                
                # Process document
                chunks = self.process_document(pdf_path)
                logger.info(f"process_document returned {len(chunks) if chunks else 0} chunks")
                
                if chunks:  # Only extend if we got chunks
                    # Log first chunk details for debugging
                    if len(chunks) > 0:
                        first_chunk = chunks[0]
                        logger.info(f"First chunk type: {type(first_chunk)}")
                        logger.info(f"First chunk has attributes: {hasattr(first_chunk, 'chunk_id')}")
                        if hasattr(first_chunk, 'chunk_id'):
                            logger.info(f"First chunk ID: {first_chunk.chunk_id}")
                    
                    all_chunks.extend(chunks)
                    logger.info(f"Extended all_chunks, new total: {len(all_chunks)}")
                else:
                    logger.warning(f"No chunks returned from {pdf_path.name}")
                
                # Memory management after each document
                if chunks:
                    del chunks
                import gc
                gc.collect()
                logger.info(f"Memory cleanup done after document {pdf_path.name}")
                
                logger.info(f"Completed {pdf_path.name}: Total chunks so far: {len(all_chunks)}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"Final: Total chunks created: {len(all_chunks)}")
        
        # Validate the chunks before returning
        if all_chunks:
            logger.info(f"Sample chunk validation:")
            sample_chunk = all_chunks[0]
            logger.info(f"  Type: {type(sample_chunk)}")
            logger.info(f"  Has chunk_id: {hasattr(sample_chunk, 'chunk_id')}")
            logger.info(f"  Has text: {hasattr(sample_chunk, 'text')}")
            
            # Check if all chunks are valid
            valid_chunks = 0
            for chunk in all_chunks[:5]:  # Check first 5
                if hasattr(chunk, 'chunk_id') and hasattr(chunk, 'text'):
                    valid_chunks += 1
            logger.info(f"Valid chunks in sample: {valid_chunks}/5")
        
        return all_chunks
def main():
    """Main function for chunking documents."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Chunk PDF documents")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="Directory containing PDF files")
    parser.add_argument("--sources", type=Path, help="Path to sources.json file")
    parser.add_argument("--tesseract_path", type=str, help="Path to Tesseract executable")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Process documents
    sources_path = args.sources or (Path(__file__).parent.parent / "data" / "sources.json")
    processor = DocumentProcessor(sources_path, args.tesseract_path)
    chunks = processor.process_all_documents(args.raw_dir)
    
    print(f"Processed {len(chunks)} chunks from {len(set(chunk.doc_id for chunk in chunks))} documents")

if __name__ == "__main__":
    main()


    