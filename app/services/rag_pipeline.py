from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.models.schemas import DocumentSummary
from app.services.answer_generator import AnswerGenerator
from app.services.chunking import chunk_documents
from app.services.document_loader import count_pages, load_documents_from_directory
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_service = EmbeddingService(settings.embedding_model_name)
        self.vector_store = VectorStore(
            index_path=settings.faiss_index_path,
            metadata_path=settings.metadata_path,
        )
        self.answer_generator = AnswerGenerator(
            model_name=settings.llm_model_name,
            openai_api_key=settings.openai_api_key,
        )
        self.vector_store.load()

    def ingest(
        self,
        document_dir: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        directory = Path(document_dir) if document_dir else self.settings.docs_directory
        documents = load_documents_from_directory(directory)
        chunks = chunk_documents(
            documents=documents,
            chunk_size=(
                chunk_size if chunk_size is not None else self.settings.default_chunk_size
            ),
            chunk_overlap=(
                chunk_overlap
                if chunk_overlap is not None
                else self.settings.default_chunk_overlap
            ),
        )
        embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.build(chunks, embeddings)
        self.vector_store.save()

        grouped: dict[str, list] = {}
        for document in documents:
            grouped.setdefault(document.doc_id, []).append(document)

        summaries = [
            DocumentSummary(
                doc_id=doc_id,
                source=items[0].source,
                chunks_indexed=len([chunk for chunk in chunks if chunk.doc_id == doc_id]),
                pages_detected=count_pages(items),
            )
            for doc_id, items in grouped.items()
        ]

        return {
            "indexed_documents": len(summaries),
            "indexed_chunks": len(chunks),
            "index_path": str(self.settings.faiss_index_path),
            "metadata_path": str(self.settings.metadata_path),
            "documents": summaries,
        }

    def ask(self, query: str, top_k: int | None = None) -> dict:
        query_embedding = self.embedding_service.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k if top_k is not None else self.settings.default_top_k,
        )
        answer, generation_mode, prompt = self.answer_generator.generate(query, results)
        citations = [
            {
                "doc_id": result.chunk.doc_id,
                "chunk_id": result.chunk.chunk_id,
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "score": round(result.score, 4),
                "text_preview": result.chunk.text[:220],
            }
            for result in results
        ]
        return {
            "query": query,
            "answer": answer,
            "generation_mode": generation_mode,
            "citations": citations,
            "prompt": prompt,
            "retrieved_context": [result.chunk.text for result in results],
        }

    def health(self) -> dict:
        if self.vector_store.index is None:
            self.vector_store.load()

        summaries = self.vector_store.document_summaries()
        return {
            "status": "ok",
            "index_ready": self.vector_store.index is not None,
            "indexed_chunks": self.vector_store.indexed_chunks,
            "indexed_documents": len(summaries),
        }

    def list_documents(self) -> list[dict]:
        if self.vector_store.index is None:
            self.vector_store.load()
        return self.vector_store.document_summaries()
