from __future__ import annotations

from collections import defaultdict

from app.models.domain import ChunkRecord, SourceDocument


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[ChunkRecord] = []
    chunk_counts: dict[str, int] = defaultdict(int)

    for document in documents:
        words = document.text.split()
        if not words:
            continue

        start = 0
        step = chunk_size - chunk_overlap
        while start < len(words):
            stop = min(start + chunk_size, len(words))
            text = " ".join(words[start:stop]).strip()
            if not text:
                break
            chunk_counts[document.doc_id] += 1
            chunk_number = chunk_counts[document.doc_id]
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_number:03d}",
                    doc_id=document.doc_id,
                    source=document.source,
                    page_number=document.page_number,
                    text=text,
                    word_count=len(text.split()),
                )
            )
            if stop == len(words):
                break
            start += step

    return chunks
