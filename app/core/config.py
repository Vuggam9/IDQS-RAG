from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = "Intelligent Document Query System"
    app_version: str = "1.0.0"
    docs_directory: Path = BASE_DIR / "data" / "documents"
    index_directory: Path = BASE_DIR / "data" / "index"
    faiss_index_path: Path = BASE_DIR / "data" / "index" / "rag_index.faiss"
    metadata_path: Path = BASE_DIR / "data" / "index" / "chunk_metadata.json"
    default_chunk_size: int = 180
    default_chunk_overlap: int = 40
    default_top_k: int = 4
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    docs_directory = Path(os.getenv("DOCS_DIRECTORY", BASE_DIR / "data" / "documents"))
    index_directory = Path(os.getenv("INDEX_DIRECTORY", BASE_DIR / "data" / "index"))
    return Settings(
        docs_directory=docs_directory,
        index_directory=index_directory,
        faiss_index_path=Path(os.getenv("FAISS_INDEX_PATH", index_directory / "rag_index.faiss")),
        metadata_path=Path(os.getenv("METADATA_PATH", index_directory / "chunk_metadata.json")),
        default_chunk_size=int(os.getenv("CHUNK_SIZE", "180")),
        default_chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
        default_top_k=int(os.getenv("TOP_K", "4")),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
