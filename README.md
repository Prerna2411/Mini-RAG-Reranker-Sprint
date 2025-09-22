# 🚀 Python Developer Technical Assessment: Mini RAG + Reranker Sprint

This project implements a **mini Retrieval-Augmented Generation (RAG) pipeline** with a **reranker** to answer domain-specific questions from provided documents.  
It retrieves relevant context passages, reranks them, and generates extractive answers with citations.  

The goal is to test **information retrieval, reranking, and answer generation** in a reproducible, CPU-only environment.

---

## 📁 Repository Structure

Mini-RAG-Reranker/
│
├── ingest/ # Code for ingesting documents and chunking
├── embeddings/ # Code to generate embeddings and index them
├── search/ # Baseline search functionality
├── reranker/ # Reranking pipeline
├── api/ # FastAPI app for question answering
├── evaluate.py # Script to evaluate answers on 8-question file
├── sources.json # Source documents
├── questions.json # 8-question test file
├── requirements.txt # Python dependencies
└── README.md # This file



---

## ⚙️ Setup Instructions

1. **Clone the repo:**
```
git clone <https://github.com/Prerna2411/Mini-RAG-Reranker-Sprint>
cd Mini-RAG-Reranker
```
2.Create and activate a virtual environment:
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

3.Install dependencies:
pip install -r requirements.txt

Generate embeddings and indexes (run once):
