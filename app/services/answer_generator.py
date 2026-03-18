from __future__ import annotations

import logging
import re

from app.models.domain import SearchResult

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, model_name: str, openai_api_key: str | None) -> None:
        self.model_name = model_name
        self.openai_api_key = openai_api_key

    def generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[str, str, str]:
        prompt = self.build_prompt(query, results)

        if self.openai_api_key:
            try:
                return self._generate_with_openai(prompt), "openai", prompt
            except Exception as exc:  # pragma: no cover
                logger.warning("Falling back to local answer generation: %s", exc)

        return self._generate_extractively(query, results), "extractive-fallback", prompt

    def build_prompt(self, query: str, results: list[SearchResult]) -> str:
        context_blocks = []
        for index, result in enumerate(results, start=1):
            citation = f"{result.chunk.doc_id}/{result.chunk.chunk_id}"
            page_suffix = (
                f" | page {result.chunk.page_number}" if result.chunk.page_number else ""
            )
            context_blocks.append(
                f"[Context {index} | {citation}{page_suffix}]\n{result.chunk.text}"
            )

        context = "\n\n".join(context_blocks)
        return (
            "You are a document-grounded assistant. Answer only using the provided context. "
            "If the answer is not supported by the context, say that the documents do not contain enough evidence.\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Answer with a concise paragraph followed by short supporting citations."
        )

    def _generate_with_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        response = client.responses.create(model=self.model_name, input=prompt)
        if getattr(response, "output_text", None):
            return response.output_text.strip()
        return str(response).strip()

    def _generate_extractively(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find any indexed document content relevant to the question."

        query_terms = set(re.findall(r"\w+", query.lower()))
        ranked_sentences: list[tuple[int, str, str]] = []

        for result in results:
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            for sentence in sentences:
                terms = set(re.findall(r"\w+", sentence.lower()))
                score = len(query_terms & terms)
                if score == 0:
                    continue
                ranked_sentences.append((score, sentence.strip(), result.chunk.chunk_id))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = ranked_sentences[:3]
        if not selected:
            selected = [
                (1, result.chunk.text[:220].strip(), result.chunk.chunk_id)
                for result in results[:2]
            ]

        body = " ".join(sentence for _, sentence, _ in selected).strip()
        citations = ", ".join(f"[{chunk_id}]" for _, _, chunk_id in selected)
        return f"{body} {citations}".strip()
