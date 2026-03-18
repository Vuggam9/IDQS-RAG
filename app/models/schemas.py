from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_chunks: int
    indexed_documents: int


class IngestRequest(BaseModel):
    document_dir: str | None = Field(
        default=None,
        description="Optional path to a directory containing PDF, TXT, or MD files.",
    )
    chunk_size: int | None = Field(default=None, ge=50, le=1000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=300)


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    chunks_indexed: int
    pages_detected: int


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    index_path: str
    metadata_path: str
    documents: list[DocumentSummary]


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    page_number: int | None = None
    score: float
    text_preview: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=10)
    include_prompt: bool = False
    include_context: bool = False


class AskResponse(BaseModel):
    query: str
    answer: str
    generation_mode: str
    citations: list[Citation]
    prompt: str | None = None
    retrieved_context: list[str] | None = None


class IndexedDocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
