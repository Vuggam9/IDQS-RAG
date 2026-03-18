from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.models.domain import SourceDocument

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents_from_directory(directory: Path) -> list[SourceDocument]:
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    documents: list[SourceDocument] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        documents.extend(load_document(path))

    if not documents:
        raise ValueError(f"No supported documents found in {directory}")

    return documents


def load_document(path: Path) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    return _load_text_document(path)


def _load_text_document(path: Path) -> list[SourceDocument]:
    text = path.read_text(encoding="utf-8")
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [
        SourceDocument(
            doc_id=path.stem.lower().replace(" ", "-"),
            source=str(path),
            text=normalized,
            page_number=None,
        )
    ]


def _load_pdf(path: Path) -> list[SourceDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required to ingest PDF files.") from exc

    reader = PdfReader(str(path))
    pages: list[SourceDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if not text:
            continue
        pages.append(
            SourceDocument(
                doc_id=path.stem.lower().replace(" ", "-"),
                source=str(path),
                text=text,
                page_number=page_number,
            )
        )
    return pages


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def count_pages(documents: Iterable[SourceDocument]) -> int:
    pages = {doc.page_number for doc in documents if doc.page_number is not None}
    return len(pages) if pages else 1
