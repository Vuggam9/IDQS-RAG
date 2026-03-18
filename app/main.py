from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.services.rag_pipeline import RAGPipeline

configure_logging()

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A production-style RAG API for grounded question answering over PDFs and text documents.",
)
app.state.pipeline = RAGPipeline(settings)
app.include_router(router)
