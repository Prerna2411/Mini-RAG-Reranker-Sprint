# 🚀 Python Developer Technical Assessment: Mini RAG + Reranker Sprint

## Overview

This repository contains a small **question-answering (Q&A) system*** over a tiny document set focused on industrial & machine safety. The project implements a **basic similarity search** first and then improves the results with a **reranker**. The system is designed to provide short, extractive answers with citations to the source documents.

---

## Folder Structure

```
TASK(RAG)/
│
├─ app/
│   ├─ __pycache__/
│   ├─ __init__.py
│   ├─ answer.py          # Handles final answer extraction with citations
│   ├─ api.py             # FastAPI endpoint /ask
│   ├─ config.py          # Configuration for embeddings, reranker, etc.
│   ├─ k.py               # Utilities for top-k selection
│   ├─ rerank.py          # Implements hybrid/learned reranker
│   ├─ schema.py          # Pydantic schemas for API requests/responses
│   ├─ search.py          # Baseline similarity search
│   └─ utils.py           # General helper functions
│
├─ ingest/
│   └─ chunk.py            ##to make chunkd
    ├─ embed.py           # to make embeddings
│   ├─ load.py            
│   ├─ pipeline.py         # Pipeline for making chunks and embeddings
│  
│   
│
├─ eval/
│   ├─ question_baseline.json
│   ├─ results_baseline.json
│   └─ results_rerank.json
│
├─ data/
│   ├─ raw/    # pdfs
│   └─ sources.json       # Metadata for PDF sources

---

## ⚙️ Setup Instructions

1. **Clone the repo:**
```
git clone <https://github.com/Prerna2411/Mini-RAG-Reranker-Sprint>
cd Mini-RAG-Reranker
```
2.Create and activate a virtual environment:
python -m venv venv
Linux/Mac
source venv/bin/activate
Windows
venv\Scripts\activate

3.Install dependencies:
pip install -r requirements.txt

4.Download PDF dataset and place in data/raw

Use the provided industrial-safety-pdfs.zip.

5.Ingest & Chunk PDFs-:
python ingest/pipeline.py
This splits PDFs into paragraph-sized chunks and stores them in SQLite.
Uses a free local model (e.g., all-MiniLM-L6-v2) + FAISS for buliding indexes.

How to Run the API

Start FastAPI:
uvicorn app.api:app --reload

Request Body:

{
  "q": "What are the requirements for emergency stop devices?",
  "k": 5,
  "mode": "rerank"
}

Response:

{
  "answer": "The emergency stop device must ... [chunk citation]",
  "contexts": [
    {"text": "...", "score": 0.87, "link": "source.pdf#chunk1"},
    ...
  ],
  "reranker_used": "hybrid"
}

Approach & Workflow

Chunk documents → split PDFs into paragraph-sized pieces.

Embed & index → compute embeddings for chunks using a local model + FAISS/Chroma.

Baseline search → retrieve top-k chunks using cosine similarity.

Reranker → hybrid approach: blend vector similarity + keyword score (BM25/SQLite FTS).

Answer extraction → extract a concise answer from top chunks with citation.

Threshold for abstain → if confidence < threshold, the model abstains.

Evaluation → run 8 questions, save results in eval/results_baseline.json and eval/results_rerank.json.


Example curl Requests

Easy Question

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" \
-d '{"q": "What is a safety device?", "k": 5, "mode": "baseline"}'


Tricky Question

curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" \
-d '{"q": "What are the differences between Type A and Type B safety devices?", "k": 5, "mode": "rerank"}'

Learnings

Implementing a hybrid reranker improved retrieval quality significantly over baseline cosine similarity.

Chunking PDFs into meaningful pieces is crucial for extractive Q&A.

Abstaining when confidence is low ensures the system doesn’t produce misleading answers.

CPU-only pipelines with free local embeddings are feasible for small-scale document QA tasks.

Constraints

No paid APIs; only CPU-based local embeddings.

Extractive answers with citations.

Simple confidence threshold for abstaining.

Outputs are repeatable by setting seeds.

References

sources.json contains metadata for all PDFs.

Model: all-MiniLM-L6-v2 for embeddings.

FAISS/Chroma for vector search.


---


