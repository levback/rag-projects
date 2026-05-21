"""Document indexer: processes raw files and stores them in the vector db."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

from src.processing.chunking import TextChunker
from src.rag.embedder import Embedder
from src.rag.vector_store import Document, VectorStore

logger = logging.getLogger(__name__)

# Default text loader: reads plain text files
_DEFAULT_LOADER: Callable[[Path], str] = lambda p: p.read_text(encoding="utf-8")


def _doc_id(source: str, chunk_index: int) -> str:
    """Stable deterministic id based on source path and chunk position."""
    raw = f"{source}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Indexer:
    """Ingests documents from disk, chunks them, embeds and stores them.

    Example::

        indexer = Indexer(embedder=emb, vector_store=vs)
        indexer.index_directory("docs/")
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunker: TextChunker | None = None,
        file_loader: Callable[[Path], str] | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._chunker = chunker or TextChunker()
        self._loader = file_loader or _DEFAULT_LOADER

    # ── Public API ────────────────────────────────────────────────────────────

    def index_text(self, text: str, source: str = "inline") -> int:
        """Chunk *text*, embed and upsert. Returns the number of chunks stored."""
        chunks = self._chunker.split(text)
        documents = self._chunks_to_documents(chunks, source)
        self._embed_and_upsert(documents)
        logger.info("Indexed %d chunks from '%s'", len(documents), source)
        return len(documents)

    def index_file(self, file_path: str | Path) -> int:
        """Load, chunk, embed and store a single file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = self._loader(path)
        return self.index_text(text, source=str(path))

    def index_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.txt",
        recursive: bool = True,
    ) -> int:
        """Recursively index all matching files in *directory*.

        Returns:
            Total number of chunks indexed.
        """
        base = Path(directory)
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = list(base.glob(glob) if recursive else base.glob(glob.lstrip("**/")))
        total = 0
        for f in files:
            try:
                total += self.index_file(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", f, exc)

        logger.info("Indexed %d total chunks from %d files in '%s'", total, len(files), directory)
        return total

    # ── Private ───────────────────────────────────────────────────────────────

    def _chunks_to_documents(self, chunks: list[str], source: str) -> list[Document]:
        return [
            Document(
                id=_doc_id(source, i),
                text=chunk,
                metadata={"source": source, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _embed_and_upsert(self, documents: list[Document]) -> None:
        texts = [d.text for d in documents]
        embeddings = self._embedder.embed_batch(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        self._store.upsert(documents)
