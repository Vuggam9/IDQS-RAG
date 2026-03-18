from app.models.domain import SourceDocument
from app.services.chunking import chunk_documents


def test_chunk_documents_creates_overlap() -> None:
    document = SourceDocument(
        doc_id="policy",
        source="policy.txt",
        text=" ".join(f"word{i}" for i in range(1, 31)),
    )

    chunks = chunk_documents([document], chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 4
    assert chunks[0].chunk_id == "policy-chunk-001"
    assert chunks[1].text.split()[:2] == ["word9", "word10"]
