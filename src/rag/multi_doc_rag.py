"""Multi-document RAG — index many PDFs / text files into a shared vector store.

Project #4: Load multiple documents, chunk all content, store in ChromaDB,
answer queries spanning all sources simultaneously.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MultiDocConfig:
    """Configuration for :class:`MultiDocumentRAG`."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """HuggingFace sentence-transformers model for embeddings."""

    vector_store: str = "chroma"
    """Vector store backend: ``"chroma"`` or ``"faiss"``."""

    collection_name: str = "multi_doc_collection"
    """ChromaDB collection name."""

    chunk_size: int = 1000
    """Maximum characters per chunk."""

    chunk_overlap: int = 200
    """Overlap between consecutive chunks (characters)."""

    top_k: int = 5
    """Number of chunks returned per query."""

    persist_dir: str = "data/vectordb/multi_doc"
    """Persistence directory for the vector store."""


@dataclass
class MultiDocResult:
    """Result of a multi-document query."""

    query: str
    answer: str
    retrieved_chunks: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)


class MultiDocumentRAG:
    """RAG pipeline that spans multiple documents.

    Supports PDFs, plain text, and markdown. The vector store is
    shared across all loaded documents, enabling cross-document retrieval.

    Args:
        config: :class:`MultiDocConfig` instance.
        llm: Optional LLM for generation. Must implement ``complete(prompt) -> str``.
             Defaults to ``google/flan-t5-base`` locally.
             Bedrock alternative: ``anthropic.claude-3-haiku-20240307-v1:0``.

    Example::

        rag = MultiDocumentRAG()
        rag.load_directory("data/papers/")
        result = rag.query("What are the main findings?")
    """

    def __init__(
        self,
        config: MultiDocConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        self._config = config or MultiDocConfig()
        self._llm = llm
        self._embedder: Any = None
        self._vector_store: Any = None
        self._document_registry: list[str] = []  # source paths

    # ── Document loading ──────────────────────────────────────────────────────

    def load_document(self, source: str) -> int:
        """Load and index a single document.

        Args:
            source: File path (PDF / txt / md) or URL.

        Returns:
            Number of chunks indexed from this document.
        """
        from src.loaders.document_loader import DocumentLoader

        loader = DocumentLoader()
        doc = loader.load(source)
        chunks = self._chunk(doc.content)
        self._ensure_store()
        for i, chunk in enumerate(chunks):
            self._vector_store.add_texts(
                [chunk],
                metadatas=[{"source": source, "chunk_index": i}],
            )
        self._document_registry.append(source)
        logger.info("Loaded %s → %d chunks", source, len(chunks))
        return len(chunks)

    def load_documents(self, sources: list[str]) -> int:
        """Load and index multiple documents. Returns total chunk count."""
        total = 0
        for src in sources:
            try:
                total += self.load_document(src)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", src, exc)
        return total

    def load_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
    ) -> int:
        """Load all matching files from *directory*."""
        from src.loaders.document_loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load_directory(directory, extensions)
        total = 0
        for doc in docs:
            chunks = self._chunk(doc.content)
            self._ensure_store()
            for i, chunk in enumerate(chunks):
                self._vector_store.add_texts(
                    [chunk],
                    metadatas=[{"source": doc.source, "chunk_index": i}],
                )
            self._document_registry.append(doc.source)
            total += len(chunks)
            logger.info("Loaded %s → %d chunks", doc.source, len(chunks))
        return total

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> MultiDocResult:
        """Answer *question* using all indexed documents.

        Args:
            question: Natural language query.

        Returns:
            :class:`MultiDocResult` with answer and provenance.
        """
        if self._vector_store is None:
            raise RuntimeError("No documents loaded. Call load_document() or load_directory() first.")

        docs = self._vector_store.similarity_search(question, k=self._config.top_k)
        retrieved_chunks = [d.page_content for d in docs]
        sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))

        context = "\n\n".join(retrieved_chunks)
        prompt = (
            f"Answer the following question using only the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        generator = self._llm or _LocalFlanT5Base()
        answer = generator.complete(prompt)

        return MultiDocResult(
            query=question,
            answer=answer.strip(),
            retrieved_chunks=retrieved_chunks,
            source_documents=sources,
        )

    @property
    def document_count(self) -> int:
        return len(self._document_registry)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_store(self) -> None:
        """Initialise the vector store and embeddings if not yet done."""
        if self._vector_store is not None:
            return
        from langchain_community.vectorstores import Chroma  # lazy import
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy import

        embeddings = HuggingFaceEmbeddings(model_name=self._config.embedding_model)
        if self._config.vector_store == "chroma":
            self._vector_store = Chroma(
                collection_name=self._config.collection_name,
                embedding_function=embeddings,
                persist_directory=self._config.persist_dir,
            )
        else:
            from langchain_community.vectorstores import FAISS  # lazy import

            self._vector_store = FAISS.from_texts(["_init_"], embeddings)
        logger.info("Initialised %s vector store", self._config.vector_store)

    def _chunk(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy import

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
            return splitter.split_text(text)
        except ImportError:
            # Fallback: simple character-level chunking
            size, overlap = self._config.chunk_size, self._config.chunk_overlap
            chunks, start = [], 0
            while start < len(text):
                end = min(start + size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += size - overlap
            return chunks


class _LocalFlanT5Base:
    """Local flan-t5-base generator for MultiDocumentRAG fallback."""

    def __init__(self, max_new_tokens: int = 256) -> None:
        self._max_new_tokens = max_new_tokens
        self._pipeline: Any = None

    def complete(self, prompt: str) -> str:
        if self._pipeline is None:
            from transformers import pipeline  # lazy import

            self._pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=self._max_new_tokens,
            )
            logger.info("Loaded local generator: google/flan-t5-base")
        return self._pipeline(prompt)[0]["generated_text"]
