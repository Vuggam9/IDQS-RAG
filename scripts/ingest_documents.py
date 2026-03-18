from app.core.config import get_settings
from app.services.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline(get_settings())
    result = pipeline.ingest()
    print("Indexed documents:", result["indexed_documents"])
    print("Indexed chunks:", result["indexed_chunks"])
    for item in result["documents"]:
        print(f"- {item.doc_id}: {item.chunks_indexed} chunks")


if __name__ == "__main__":
    main()
