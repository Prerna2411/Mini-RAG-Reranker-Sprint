
import sys
sys.path.append('.')
from pathlib import Path
from ingest.chunk import DocumentProcessor
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

print('=== Testing DocumentProcessor ===')

# Test sources loading
print('1. Testing sources loading...')
try:
    processor = DocumentProcessor(Path('data/sources.json'))
    print(f'Sources loaded: {len(processor.sources)} entries')
    print('First few keys:', list(processor.sources.keys())[:3])
except Exception as e:
    print(f'Error loading sources: {e}')
    import traceback
    traceback.print_exc()

# Test document processing
print('\n2. Testing document processing...')
pdf_path = Path('data/raw/06 - Todd Dickey - IRSC 2022 (Introduction to Industrial Robot Safety ISO 10218 Parts 1 and 2).pdf')
try:
    chunks = processor.process_document(pdf_path)
    print(f'Success! Created {len(chunks)} chunks')
    if chunks:
        print(f'First chunk text: {chunks[0].text[:200]}...')
except Exception as e:
    print(f'Error processing document: {e}')
    import traceback
    traceback.print_exc()
