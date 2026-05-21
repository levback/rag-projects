"""Basic RAG pipeline — FAISS + sentence-transformers + configurable LLM.

Project #1: Simplest end-to-end RAG with no external API key required.
Default local model: ``flan-t5-small`` (generation) + ``all-MiniLM-L6-v2`` (embeddings).
Both can be replaced by injecting a Bedrock/OpenAI :class:`~src.core.base_llm.BaseLLM`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BasicRAGConfig:
    """Configuration for :class:`BasicRAGPipeline`."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """HuggingFace sentence-transformers model for embeddings."""

    generation_model: str = "google/flan-t5-small"
    """HuggingFace text2text-generation model for answer generation.
    Bedrock alternative: ``amazon.titan-text-lite-v1``."""

    chunk_size: int = 500
    """Maximum characters per text chunk."""

    chunk_overlap: int = 50
    """Character overlap between consecutive chunks."""

    top_k: int = 3
    """Number of chunks retrieved per query."""

    max_new_tokens: int = 256
    """Maximum tokens the generation model may produce."""

    index_dir: str = "data/vectordb/basic_rag"
    """Directory for persisting the FAISS index."""


@dataclass
class BasicRAGResult:
    """Answer from a :class:`BasicRAGPipeline` query."""

    query: str
    answer: str
    retrieved_chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class _LocalFlanT5Generator:
    """Thin wrapper around HuggingFace ``flan-t5-small`` for text generation.

    Implements a minimal ``complete(prompt) -> str`` interface compatible
    with :class:`~src.core.base_llm.BaseLLM`.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 256) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline  # lazy import

        self._pipeline = pipeline(
            "text2text-generation",
            model=self._model_name,
            max_new_tokens=self._max_new_tokens,
        )
        logger.info("Loaded local generator: %s", self._model_name)

    def complete(self, prompt: str) -> str:
        """Generate a response for *prompt*."""
        self._load()
        results = self._pipeline(prompt)
        return results[0]["generated_text"]


class BasicRAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    Workflow:
    1. ``index(texts)`` — chunk, embed, store in FAISS
    2. ``query(q)``     — embed query → retrieve chunks → generate answer

    Args:
        config: :class:`BasicRAGConfig` instance.
        llm: Optional external LLM (Bedrock, OpenAI, Anthropic). When provided,
             it replaces the local ``flan-t5-small`` generator.
             Must have a ``complete(prompt: str) -> str`` method.

    Bedrock substitution example::

        from src.core.model_factory import ModelFactory
        llm = ModelFactory.create_llm("bedrock", "amazon.titan-text-express-v1")
        pipeline = BasicRAGPipeline(llm=llm)
    """

    def __init__(
        self,
        config: BasicRAGConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        self._config = config or BasicRAGConfig()
        self._llm = llm
        self._embedder: Any = None
        self._index: Any = None
        self._chunks: list[str] = []
        self._sources: list[str] = []

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(
        self,
        texts: list[str],
        sources: list[str] | None = None,
    ) -> int:
        """Chunk and embed *texts* into the FAISS index.

        Args:
            texts: Raw text strings to index.
            sources: Optional provenance label per text (same length as *texts*).

        Returns:
            Total number of chunks indexed.
        """
        import faiss  # lazy import
        import numpy as np  # lazy import
        from sentence_transformers import SentenceTransformer  # lazy import

        sources = sources or ["unknown"] * len(texts)
        all_chunks: list[str] = []
        all_sources: list[str] = []
        for text, src in zip(texts, sources):
            chunks = self._chunk(text)
            all_chunks.extend(chunks)
            all_sources.extend([src] * len(chunks))

        # Embed
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._config.embedding_model)
            logger.info("Loaded embedding model: %s", self._config.embedding_model)

        vectors = self._embedder.encode(all_chunks, show_progress_bar=False)
        vectors = vectors.astype(np.float32)

        # Build FAISS index
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        self._index = index
        self._chunks = all_chunks
        self._sources = all_sources
        logger.info("Indexed %d chunks from %d documents", len(all_chunks), len(texts))
        return len(all_chunks)

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> BasicRAGResult:
        """Answer *question* using the indexed documents.

        Args:
            question: Natural language query.

        Returns:
            :class:`BasicRAGResult` with the generated answer and retrieved chunks.
        """
        if self._index is None or not self._chunks:
            raise RuntimeError("No documents indexed. Call index() first.")

        import numpy as np  # lazy import

        q_vec = self._embedder.encode([question], show_progress_bar=False).astype(np.float32)
        k = min(self._config.top_k, len(self._chunks))
        distances, indices = self._index.search(q_vec, k)

        retrieved_chunks: list[str] = []
        retrieved_sources: list[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                retrieved_chunks.append(self._chunks[idx])
                retrieved_sources.append(self._sources[idx])

        context = "\n\n".join(retrieved_chunks)
        prompt = (
            f"Answer the question using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        generator = self._llm or _LocalFlanT5Generator(
            self._config.generation_model, self._config.max_new_tokens
        )
        answer = generator.complete(prompt)

        return BasicRAGResult(
            query=question,
            answer=answer.strip(),
            retrieved_chunks=retrieved_chunks,
            sources=list(dict.fromkeys(retrieved_sources)),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> list[str]:
        """Split *text* into overlapping character-level chunks."""
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks
