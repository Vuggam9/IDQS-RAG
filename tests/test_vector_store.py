from pathlib import Path

from app.models.domain import ChunkRecord
from app.services.vector_store import VectorStore


def test_vector_store_build_save_load_and_search(tmp_path: Path) -> None:
    store = VectorStore(
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    chunks = [
        ChunkRecord(
            chunk_id="doc-a-chunk-001",
            doc_id="doc-a",
            source="a.txt",
            text="incident escalation procedure",
        ),
        ChunkRecord(
            chunk_id="doc-b-chunk-001",
            doc_id="doc-b",
            source="b.txt",
            text="employee reimbursement policy",
        ),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.build(chunks, embeddings)
    store.save()

    reloaded = VectorStore(
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    assert reloaded.load() is True

    results = reloaded.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc-a"
