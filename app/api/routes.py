from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexedDocumentsResponse,
    IngestRequest,
    IngestResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    return HealthResponse(**pipeline.health())


@router.get("/api/v1/documents", response_model=IndexedDocumentsResponse)
def list_documents(request: Request) -> IndexedDocumentsResponse:
    pipeline = request.app.state.pipeline
    return IndexedDocumentsResponse(documents=pipeline.list_documents())


@router.post("/api/v1/documents/ingest", response_model=IngestResponse)
def ingest_documents(request: Request, payload: IngestRequest) -> IngestResponse:
    pipeline = request.app.state.pipeline
    try:
        response = pipeline.ingest(
            document_dir=payload.document_dir,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(**response)


@router.post("/api/v1/ask", response_model=AskResponse)
def ask_question(request: Request, payload: AskRequest) -> AskResponse:
    pipeline = request.app.state.pipeline
    try:
        response = pipeline.ask(query=payload.query, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not payload.include_prompt:
        response["prompt"] = None
    if not payload.include_context:
        response["retrieved_context"] = None
    return AskResponse(**response)
