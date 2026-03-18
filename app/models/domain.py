from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SourceDocument:
    doc_id: str
    source: str
    text: str
    page_number: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page_number: int | None = None
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    chunk: ChunkRecord
    score: float
