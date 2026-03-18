from fastapi.testclient import TestClient

from app.main import app


class StubPipeline:
    def health(self) -> dict:
        return {
            "status": "ok",
            "index_ready": True,
            "indexed_chunks": 6,
            "indexed_documents": 2,
        }

    def list_documents(self) -> list[dict]:
        return [
            {
                "doc_id": "sample-doc",
                "source": "sample.txt",
                "chunks_indexed": 3,
                "pages_detected": 1,
            }
        ]

    def ingest(self, document_dir=None, chunk_size=None, chunk_overlap=None) -> dict:
        return {
            "indexed_documents": 1,
            "indexed_chunks": 3,
            "index_path": "index.faiss",
            "metadata_path": "metadata.json",
            "documents": self.list_documents(),
        }

    def ask(self, query: str, top_k: int | None = None) -> dict:
        return {
            "query": query,
            "answer": "Sample grounded answer.",
            "generation_mode": "extractive-fallback",
            "citations": [
                {
                    "doc_id": "sample-doc",
                    "chunk_id": "sample-doc-chunk-001",
                    "source": "sample.txt",
                    "page_number": None,
                    "score": 0.91,
                    "text_preview": "This is a sample preview.",
                }
            ],
            "prompt": "Prompt text",
            "retrieved_context": ["This is a sample preview."],
        }


def test_health_endpoint() -> None:
    app.state.pipeline = StubPipeline()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["index_ready"] is True


def test_ask_endpoint_hides_optional_fields_by_default() -> None:
    app.state.pipeline = StubPipeline()
    client = TestClient(app)

    response = client.post("/api/v1/ask", json={"query": "What is the policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Sample grounded answer."
    assert body["prompt"] is None
    assert body["retrieved_context"] is None
