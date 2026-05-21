"""Vector database wrapper (provider-agnostic)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A text chunk with optional metadata and a unique id."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A retrieved document with its similarity score."""

    document: Document
    score: float


class VectorStore:
    """Thin wrapper around a vector database backend.

    Supported backends: ``"chroma"``, ``"faiss"``.
    """

    def __init__(
        self,
        provider: str = "chroma",
        collection_name: str = "default",
        persist_directory: str = "data/vectordb",
        distance_metric: str = "cosine",
    ) -> None:
        self._provider = provider
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._distance_metric = distance_metric
        self._store = None  # lazy init

    # ── Backend init ─────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is not None:
            return self._store

        if self._provider == "chroma":
            self._store = self._init_chroma()
        elif self._provider == "faiss":
            self._store = self._init_faiss()
        else:
            raise ValueError(f"Unknown vector store provider: {self._provider!r}")

        return self._store

    def _init_chroma(self):
        try:
            import chromadb  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            ) from exc

        client = chromadb.PersistentClient(path=self._persist_directory)
        collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )
        logger.info("ChromaDB collection '%s' ready at %s", self._collection_name, self._persist_directory)
        return collection

    def _init_faiss(self):
        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc
        # FAISS index is created on first upsert when we know the dimension
        return {"faiss": faiss, "index": None, "docs": {}}

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(self, documents: list[Document]) -> None:
        """Insert or update *documents* in the store."""
        store = self._get_store()
        logger.debug("Upserting %d documents", len(documents))

        if self._provider == "chroma":
            store.upsert(
                ids=[d.id for d in documents],
                documents=[d.text for d in documents],
                embeddings=[d.embedding for d in documents if d.embedding],
                metadatas=[d.metadata for d in documents],
            )
        elif self._provider == "faiss":
            self._faiss_upsert(store, documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar documents to *query_embedding*."""
        store = self._get_store()
        logger.debug("Searching top_k=%d", top_k)

        if self._provider == "chroma":
            return self._chroma_search(store, query_embedding, top_k, where)
        if self._provider == "faiss":
            return self._faiss_search(store, query_embedding, top_k)

        raise ValueError(f"Unknown provider: {self._provider!r}")

    def delete(self, ids: list[str]) -> None:
        """Remove documents by *ids*."""
        store = self._get_store()
        if self._provider == "chroma":
            store.delete(ids=ids)
        logger.debug("Deleted %d documents", len(ids))

    def count(self) -> int:
        """Return total number of documents stored."""
        store = self._get_store()
        if self._provider == "chroma":
            return store.count()
        if self._provider == "faiss":
            return len(store["docs"])
        return 0

    # ── Chroma helpers ────────────────────────────────────────────────────────

    def _chroma_search(self, store, query_embedding, top_k, where) -> list[SearchResult]:
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = store.query(**kwargs)
        output: list[SearchResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                id=doc_id,
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] or {},
            )
            score = 1.0 - results["distances"][0][i]  # convert distance → similarity
            output.append(SearchResult(document=doc, score=score))
        return output

    # ── FAISS helpers ─────────────────────────────────────────────────────────

    def _faiss_upsert(self, store, documents: list[Document]) -> None:
        import numpy as np

        faiss = store["faiss"]
        embeddings = [d.embedding for d in documents if d.embedding]
        if not embeddings:
            return

        dim = len(embeddings[0])
        if store["index"] is None:
            store["index"] = faiss.IndexFlatIP(dim)

        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        store["index"].add(vecs)
        for doc in documents:
            store["docs"][doc.id] = doc

    def _faiss_search(self, store, query_embedding, top_k) -> list[SearchResult]:
        import numpy as np

        if store["index"] is None or store["index"].ntotal == 0:
            return []

        faiss = store["faiss"]
        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        distances, indices = store["index"].search(query, top_k)

        doc_list = list(store["docs"].values())
        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(doc_list):
                continue
            results.append(SearchResult(document=doc_list[idx], score=float(dist)))
        return results
