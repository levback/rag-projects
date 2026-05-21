"""Embedding creation utilities."""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class Embedder:
    """Creates dense vector embeddings from text using a configurable backend.

    Supported backends:
    - ``"openai"``   – OpenAI text-embedding models
    - ``"huggingface"`` – Sentence-Transformers / HF models (local)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._batch_size = batch_size
        self._api_key = api_key
        self._client = None  # lazy init

    # ── Backend init ─────────────────────────────────────────────────────────

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore[import]

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _get_hf_model(self):
        if self._client is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for HuggingFace embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._client = SentenceTransformer(self._model)
        return self._client

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single *text*."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for a list of *texts* with batching."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            logger.debug("Embedding batch %d-%d via %s", i, i + len(batch), self._provider)
            embeddings = self._embed_batch_impl(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        if self._provider == "openai":
            client = self._get_openai_client()
            response = client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]

        if self._provider == "huggingface":
            model = self._get_hf_model()
            return model.encode(texts, convert_to_numpy=False).tolist()

        raise ValueError(f"Unknown embedding provider: {self._provider!r}")

    async def aembed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Async variant — runs the sync method in a thread executor."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)
