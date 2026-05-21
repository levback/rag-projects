"""Document search / retrieval layer."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.rag.embedder import Embedder
from src.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.0
    metadata_filter: dict | None = None


class Retriever:
    """Converts a natural-language query into a vector and searches the store.

    Optionally filters results by a minimum similarity score.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Embed *query* and return the most relevant documents.

        Args:
            query: Natural-language search string.
            top_k: Override the default ``top_k`` from config.

        Returns:
            List of :class:`~src.rag.vector_store.SearchResult` ordered by score.
        """
        if not query.strip():
            return []

        k = top_k or self._config.top_k
        logger.debug("Retrieving top-%d docs for query: %.60s", k, query)

        query_embedding = self._embedder.embed(query)
        results = self._store.search(
            query_embedding=query_embedding,
            top_k=k,
            where=self._config.metadata_filter,
        )

        if self._config.similarity_threshold > 0.0:
            results = [r for r in results if r.score >= self._config.similarity_threshold]
            logger.debug("After threshold filter: %d results remain", len(results))

        return results

    def retrieve_texts(self, query: str, top_k: int | None = None) -> list[str]:
        """Convenience wrapper that returns only the document texts."""
        return [r.document.text for r in self.retrieve(query, top_k=top_k)]

    async def aretrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Async variant of :meth:`retrieve`."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, top_k)
