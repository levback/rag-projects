"""Agentic RAG — intent-routing pipeline that decides whether to retrieve or respond directly.

Project #5: An agent controller routes the query to either:
- ``search`` path: vector store retrieval → context injection → generation
- ``direct`` path: generation without retrieval (for conversational turns)

Routing can be keyword-based (local, free) or LLM-based (smarter, needs a model).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that strongly imply a factual lookup is needed
_SEARCH_KEYWORDS: frozenset[str] = frozenset(
    [
        "what", "who", "when", "where", "which", "why", "how",
        "define", "explain", "describe", "list", "find", "search",
        "tell me about", "give me", "show me", "summarize",
    ]
)


class IntentType(Enum):
    SEARCH = "search"
    DIRECT = "direct"


@dataclass
class AgenticRAGConfig:
    """Configuration for :class:`AgenticRAGPipeline`."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """HuggingFace sentence-transformers model for embeddings."""

    generation_model: str = "google/flan-t5-base"
    """Local model for answer generation.
    Bedrock alternative: ``meta.llama3-8b-instruct-v1:0``."""

    router_model: str | None = None
    """If set, use this model via the injected LLM to route intent.
    If None, use keyword-based routing (free, offline)."""

    top_k: int = 3
    """Number of chunks to retrieve."""

    max_new_tokens: int = 256
    """Maximum tokens in generated answer."""

    vector_store: str = "chroma"
    """Vector store backend: ``"chroma"`` or ``"faiss"``."""

    collection_name: str = "agentic_rag"
    """ChromaDB collection name."""

    persist_dir: str = "data/vectordb/agentic_rag"
    """Persistence directory."""


@dataclass
class AgenticRAGResult:
    """Result of an :class:`AgenticRAGPipeline` query."""

    query: str
    answer: str
    intent: IntentType = IntentType.DIRECT
    retrieved_chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class AgenticRAGPipeline:
    """RAG pipeline with intent-aware routing.

    The routing decision is made before retrieval:
    - **keyword router** (default): checks the query for interrogative/search words
    - **LLM router** (opt-in): asks the injected LLM to classify the intent

    Args:
        config: :class:`AgenticRAGConfig` instance.
        llm: Optional LLM. Used for LLM-based routing when *config.router_model*
             is set, and for answer generation.
             Bedrock example: ``ModelFactory.create_llm("bedrock", "amazon.titan-text-express-v1")``

    Example::

        pipeline = AgenticRAGPipeline()
        pipeline.index(["The Eiffel Tower is in Paris, France."])
        result = pipeline.query("Where is the Eiffel Tower?")
        print(result.intent, result.answer)
    """

    def __init__(
        self,
        config: AgenticRAGConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        self._config = config or AgenticRAGConfig()
        self._llm = llm
        self._embedder: Any = None
        self._vector_store: Any = None
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._local_generator: Any = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(
        self,
        texts: list[str],
        sources: list[str] | None = None,
    ) -> int:
        """Chunk and embed *texts* into the vector store."""
        sources = sources or ["doc"] * len(texts)
        all_chunks: list[str] = []
        all_sources: list[str] = []
        for text, src in zip(texts, sources):
            chunks = self._chunk(text)
            all_chunks.extend(chunks)
            all_sources.extend([src] * len(chunks))

        self._ensure_store()
        self._vector_store.add_texts(
            all_chunks,
            metadatas=[{"source": s} for s in all_sources],
        )
        self._chunks = all_chunks
        self._sources = all_sources
        logger.info("AgenticRAG indexed %d chunks", len(all_chunks))
        return len(all_chunks)

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> AgenticRAGResult:
        """Route *question* and generate an answer.

        Args:
            question: The user's query.

        Returns:
            :class:`AgenticRAGResult` with the answer and routing decision.
        """
        intent = self._route(question)
        logger.debug("Intent for %r: %s", question, intent.value)

        if intent == IntentType.SEARCH and self._vector_store is not None:
            return self._search_and_generate(question)
        return self._generate_direct(question, intent)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _route(self, query: str) -> IntentType:
        """Classify the intent of *query*."""
        if self._config.router_model and self._llm is not None:
            return self._llm_route(query)
        return self._keyword_route(query)

    def _keyword_route(self, query: str) -> IntentType:
        query_lower = query.lower()
        for kw in _SEARCH_KEYWORDS:
            if kw in query_lower:
                return IntentType.SEARCH
        return IntentType.DIRECT

    def _llm_route(self, query: str) -> IntentType:
        prompt = (
            "Classify the following query as either 'search' (needs external knowledge lookup) "
            "or 'direct' (can be answered conversationally without lookup).\n"
            "Reply with a single word: search or direct.\n\n"
            f"Query: {query}\n"
            "Classification:"
        )
        try:
            raw = self._llm.complete(prompt).strip().lower()
            if "direct" in raw:
                return IntentType.DIRECT
        except Exception as exc:
            logger.warning("LLM routing failed, falling back to keyword: %s", exc)
        return IntentType.SEARCH

    def _search_and_generate(self, question: str) -> AgenticRAGResult:
        docs = self._vector_store.similarity_search(question, k=self._config.top_k)
        chunks = [d.page_content for d in docs]
        sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))

        context = "\n\n".join(chunks)
        prompt = (
            "Answer the question using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = self._generate(prompt)
        return AgenticRAGResult(
            query=question,
            answer=answer,
            intent=IntentType.SEARCH,
            retrieved_chunks=chunks,
            sources=sources,
        )

    def _generate_direct(self, question: str, intent: IntentType) -> AgenticRAGResult:
        prompt = f"Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"
        answer = self._generate(prompt)
        return AgenticRAGResult(query=question, answer=answer, intent=intent)

    def _generate(self, prompt: str) -> str:
        if self._llm is not None:
            return self._llm.complete(prompt).strip()
        if self._local_generator is None:
            from transformers import pipeline  # lazy import

            self._local_generator = pipeline(
                "text2text-generation",
                model=self._config.generation_model,
                max_new_tokens=self._config.max_new_tokens,
            )
        return self._local_generator(prompt)[0]["generated_text"].strip()

    def _ensure_store(self) -> None:
        if self._vector_store is not None:
            return
        from langchain_community.vectorstores import Chroma  # lazy import
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy import

        embeddings = HuggingFaceEmbeddings(model_name=self._config.embedding_model)
        self._vector_store = Chroma(
            collection_name=self._config.collection_name,
            embedding_function=embeddings,
            persist_directory=self._config.persist_dir,
        )

    def _chunk(self, text: str, size: int = 500, overlap: int = 50) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks
