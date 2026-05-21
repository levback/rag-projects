"""High-level inference engine that wires together retrieval and generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from src.core.base_llm import BaseLLM, Message
from src.prompts.templates import PromptTemplate, RAG_QA
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Everything needed for a single inference call."""

    query: str
    system_prompt: str = ""
    use_rag: bool = True
    top_k: int = 5
    extra_context: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Structured output from the inference engine."""

    answer: str
    sources: list[str] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)
    retrieved_chunks: list[str] = field(default_factory=list)


class InferenceEngine:
    """Orchestrates RAG retrieval + LLM generation for a given request.

    When ``use_rag=True``, the engine:
    1. Retrieves relevant context from the vector store.
    2. Formats a RAG prompt using the configured template.
    3. Calls the LLM and returns a structured :class:`InferenceResult`.

    When ``use_rag=False``, it performs a direct LLM completion.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever | None = None,
        rag_template: PromptTemplate | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._rag_template = rag_template or RAG_QA

    # ── Sync ─────────────────────────────────────────────────────────────────

    def run(self, request: InferenceRequest) -> InferenceResult:
        """Execute the request and return a structured result."""
        logger.debug("InferenceEngine.run — use_rag=%s, query=%.60s", request.use_rag, request.query)

        retrieved_chunks: list[str] = []
        sources: list[str] = []

        if request.use_rag and self._retriever is not None:
            results = self._retriever.retrieve(request.query, top_k=request.top_k)
            retrieved_chunks = [r.document.text for r in results]
            sources = [r.document.metadata.get("source", "") for r in results]

        prompt_text = self._build_prompt(request, retrieved_chunks)
        messages = self._build_messages(request.system_prompt, prompt_text)

        response = self._llm.complete(messages)

        return InferenceResult(
            answer=response.content,
            sources=list(dict.fromkeys(sources)),  # deduplicate while preserving order
            model=response.model,
            usage=response.usage,
            retrieved_chunks=retrieved_chunks,
        )

    def stream(self, request: InferenceRequest) -> Iterator[str]:
        """Stream the answer token-by-token (no structured result)."""
        retrieved_chunks: list[str] = []

        if request.use_rag and self._retriever is not None:
            results = self._retriever.retrieve(request.query, top_k=request.top_k)
            retrieved_chunks = [r.document.text for r in results]

        prompt_text = self._build_prompt(request, retrieved_chunks)
        messages = self._build_messages(request.system_prompt, prompt_text)
        yield from self._llm.stream(messages)

    # ── Async ─────────────────────────────────────────────────────────────────

    async def arun(self, request: InferenceRequest) -> InferenceResult:
        """Async variant of :meth:`run`."""
        logger.debug("InferenceEngine.arun — use_rag=%s", request.use_rag)

        retrieved_chunks: list[str] = []
        sources: list[str] = []

        if request.use_rag and self._retriever is not None:
            results = await self._retriever.aretrieve(request.query, top_k=request.top_k)
            retrieved_chunks = [r.document.text for r in results]
            sources = [r.document.metadata.get("source", "") for r in results]

        prompt_text = self._build_prompt(request, retrieved_chunks)
        messages = self._build_messages(request.system_prompt, prompt_text)
        response = await self._llm.acomplete(messages)

        return InferenceResult(
            answer=response.content,
            sources=list(dict.fromkeys(sources)),
            model=response.model,
            usage=response.usage,
            retrieved_chunks=retrieved_chunks,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(self, request: InferenceRequest, chunks: list[str]) -> str:
        if not chunks and not request.extra_context:
            return request.query

        context_parts = list(chunks)
        if request.extra_context:
            context_parts.insert(0, request.extra_context)

        context = "\n\n---\n\n".join(context_parts)
        return self._rag_template.format(context=context, question=request.query)

    @staticmethod
    def _build_messages(system_prompt: str, prompt_text: str) -> list[Message]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt_text))
        return messages
