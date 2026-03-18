# Intelligent Document Query System using Retrieval-Augmented Generation (RAG)

This project is a document question-answering system built with a Retrieval-Augmented Generation pipeline. Instead of asking a language model to answer from memory, the system first retrieves the most relevant parts of the source documents and then answers from that context.

The application combines document ingestion, chunking, semantic embeddings, FAISS-based retrieval, prompt construction, and answer generation behind a FastAPI service. The result is a workflow that is easier to trust because answers are tied back to retrieved source content.

## What I Wanted To Build

I wanted this one to feel like a realistic internal knowledge assistant for policy search, operations playbooks, or documentation support. The interesting part is not just calling an LLM, but building the retrieval path that makes the response grounded.

The project demonstrates:

- document ingestion for PDF, TXT, and Markdown files
- chunking with metadata preservation
- semantic embeddings with Sentence Transformers
- FAISS vector indexing and similarity search
- grounded prompt construction for RAG
- FastAPI endpoint design for ingestion and question answering
- optional OpenAI generation with a local fallback mode
- automated tests for chunking, retrieval, and API behavior

## Architecture

```text
Documents -> Text Extraction -> Chunking -> Embeddings -> FAISS Index
                                                         |
User Query -> Query Embedding -> Similarity Search -> Top-K Chunks
                                                         |
                                             Prompt Construction
                                                         |
                                         LLM / Extractive Fallback
                                                         |
                                                   Final Answer
```

## Project Structure

```text
intelligent-document-query-system/
|-- app/
|   |-- api/
|   |-- core/
|   |-- models/
|   `-- services/
|-- data/
|   |-- documents/
|   `-- index/
|-- scripts/
|-- tests/
|-- .env.example
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## End-to-End Flow

1. Documents are loaded from `data/documents/`.
2. The ingestion pipeline extracts text and keeps source metadata such as filename and page number.
3. The chunking layer splits content into overlapping chunks for better retrieval quality.
4. The embedding layer converts chunks into vectors using Sentence Transformers.
5. FAISS stores those vectors for efficient similarity search.
6. When the user asks a question, the query is embedded and matched against the FAISS index.
7. Top-ranked chunks are inserted into a grounded prompt.
8. The answer generator uses OpenAI if `OPENAI_API_KEY` is configured, otherwise it returns an extractive grounded answer locally.

## Setup

```powershell
cd "C:\Users\Maneesha Vuggam\Documents\New project\intelligent-document-query-system"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Optional:

- Set `OPENAI_API_KEY` in `.env` if you want LLM-generated answers.
- Leave it blank if you want to demo the system in local extractive mode.

## Run Ingestion

```powershell
python -m scripts.ingest_documents
```

This creates:

- `data/index/rag_index.faiss`
- `data/index/chunk_metadata.json`

## Run The API

```powershell
uvicorn app.main:app --reload
```

Open the interactive docs at:

- `http://127.0.0.1:8000/docs`

## Core Endpoints

- `GET /health`
- `GET /api/v1/documents`
- `POST /api/v1/documents/ingest`
- `POST /api/v1/ask`

## Example API Usage

Ingest the sample documents:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/documents/ingest" -Method Post -ContentType "application/json" -Body "{}"
```

Ask a question:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/ask" -Method Post -ContentType "application/json" -Body '{"query":"When must travel reimbursement requests be submitted?","include_context":true}'
```

## Example Questions

- `When must travel reimbursement requests be submitted?`
- `What are the rules for Severity 1 incidents?`
- `How often must stakeholder updates be posted during a major outage?`
- `How much PTO do full-time employees accrue?`

## Testing

```powershell
pytest tests -q
```

## Tradeoffs And Limitations

- FAISS is a good local choice for semantic search, but it is not the same as operating a distributed vector service in production.
- The fallback answer mode keeps the app usable without an API key, but it is much simpler than a true generative response.
- The sample documents are intentionally small and structured for local development.

## What I Learned

This project reinforced that most of the quality in a RAG system comes from the retrieval side. If chunking, metadata, and search quality are weak, the generator has very little chance of producing a reliable answer.

## How To Explain This In An Interview

You can explain the architecture like this:

> This system uses Retrieval-Augmented Generation, meaning it does not answer directly from model memory. It first converts document chunks into embeddings, stores them in FAISS, retrieves the most relevant chunks for a query, and then generates a response grounded in that retrieved context. This improves factual accuracy, makes answers traceable to source documents, and reduces hallucinations compared with a standalone LLM call.

## Manual Build Guide

If you want to build this project from scratch manually, use this order:

1. Create a FastAPI app and define your API contracts.
2. Add a document loader for PDF and text files.
3. Implement chunking with overlap and metadata tracking.
4. Generate embeddings with a sentence-transformer model.
5. Build and persist a FAISS index.
6. Create a query pipeline that embeds the question and retrieves top-k chunks.
7. Build a prompt with the retrieved chunks.
8. Add an answer generator using an LLM or a local fallback.
9. Add tests for chunking, retrieval, and API responses.
10. Document the setup, architecture, and usage professionally.

## Resume Bullet

Built a FastAPI-based Retrieval-Augmented Generation system for querying enterprise documents, combining PDF ingestion, chunking, Sentence Transformers embeddings, FAISS semantic retrieval, and grounded answer generation with source-aware responses.
