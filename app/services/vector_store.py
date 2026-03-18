from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.models.domain import ChunkRecord, SearchResult


class VectorStore:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[ChunkRecord] = []

    def build(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks or not embeddings:
            raise ValueError("Chunks and embeddings are required to build the index.")

        vectors = np.array(embeddings, dtype="float32")
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        self.chunks = chunks

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Vector index has not been built.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks = [ChunkRecord(**item) for item in data]
        return True

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if self.index is None:
            loaded = self.load()
            if not loaded:
                raise ValueError("No index is available. Run ingestion first.")

        query_vector = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results: list[SearchResult] = []
        for score, index_value in zip(scores[0], indices[0]):
            if index_value == -1:
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[index_value],
                    score=float(score),
                )
            )
        return results

    def document_summaries(self) -> list[dict]:
        per_document: dict[str, dict] = {}
        for chunk in self.chunks:
            summary = per_document.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "chunks_indexed": 0,
                    "pages": set(),
                },
            )
            summary["chunks_indexed"] += 1
            if chunk.page_number is not None:
                summary["pages"].add(chunk.page_number)

        return [
            {
                "doc_id": item["doc_id"],
                "source": item["source"],
                "chunks_indexed": item["chunks_indexed"],
                "pages_detected": len(item["pages"]) if item["pages"] else 1,
            }
            for item in per_document.values()
        ]

    @property
    def indexed_chunks(self) -> int:
        return len(self.chunks)
