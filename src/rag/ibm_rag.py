"""IBM-style production RAG — robust document indexing with retry logic and monitoring.

Project #2: Production-grade patterns from IBM's Docling+Granite tutorial.
Key features: retry on LLM failures, structured metadata, Chroma persistence,
health checks, and explicit Bedrock integration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class IBMRAGConfig:
    """Configuration for :class:`IBMProductionRAG`."""

    embedding_model: str = "ibm-granite/granite-embedding-30m-english"
    """IBM Granite embedding model (or ``all-MiniLM-L6-v2`` as fallback)."""

    generation_model: str = "google/flan-t5-base"
    """Local generation model.
    Bedrock alternatives:
    - ``ibm-granite/granite-3.2-2b-instruct`` (via Amazon Bedrock Marketplace)
    - ``anthropic.claude-3-haiku-20240307-v1:0``
    - ``amazon.titan-text-express-v1``."""

    chunk_size: int = 800
    """Characters per text chunk."""

    chunk_overlap: int = 150
    """Overlap between consecutive chunks."""

    top_k: int = 4
    """Number of chunks to retrieve per query."""

    max_retries: int = 3
    """Maximum LLM call retries on transient failure."""

    retry_delay: float = 1.0
    """Seconds to wait between retries (exponential backoff base)."""

    collection_name: str = "ibm_rag_production"
    """ChromaDB collection name."""

    persist_dir: str = "data/vectordb/ibm_rag"
    """Vector store persistence directory."""

    include_tables: bool = True
    """Index table content (requires Docling)."""

    include_images: bool = False
    """Index image captions (requires vision LLM + Docling)."""


@dataclass
class IBMRAGResult:
    """Result of an :class:`IBMProductionRAG` query."""

    query: str
    answer: str
    retrieved_passages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retries_used: int = 0


class IBMProductionRAG:
    """Production-grade RAG pipeline with retry logic, structured logging,
    and explicit Bedrock-first design.

    Supports two document ingestion modes:
    1. **Standard** (``load_text``): plain text or PDF via PDFExtractor
    2. **Docling** (``load_pdf_docling``): full Docling parse (text + tables + optional images)

    Args:
        config: :class:`IBMRAGConfig` instance.
        llm: Optional LLM. Bedrock is strongly recommended for production.
             Example::

                 llm = ModelFactory.create_llm("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
                 rag = IBMProductionRAG(llm=llm)
    """

    def __init__(
        self,
        config: IBMRAGConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        self._config = config or IBMRAGConfig()
        self._llm = llm
        self._vector_store: Any = None
        self._embedder: Any = None
        self._local_gen: Any = None
        self._indexed_sources: list[str] = []
        self._total_chunks: int = 0

    # ── Health ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if at least one document has been indexed."""
        return self._total_chunks > 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "indexed_sources": len(self._indexed_sources),
            "total_chunks": self._total_chunks,
            "vector_store": self._config.collection_name,
        }

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def load_text(self, text: str, source: str = "unknown") -> int:
        """Index raw text. Returns number of chunks added."""
        chunks = self._chunk(text)
        self._ensure_store()
        self._vector_store.add_texts(
            chunks,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        )
        self._indexed_sources.append(source)
        self._total_chunks += len(chunks)
        logger.info("[IBMProductionRAG] Indexed %d chunks from %s", len(chunks), source)
        return len(chunks)

    def load_pdf(self, path: str | Path) -> int:
        """Extract text from *path* (standard PDFExtractor) and index it."""
        from src.processing.pdf_extractor import PDFExtractor

        path = Path(path)
        extractor = PDFExtractor()
        text = extractor.extract(path)
        return self.load_text(text, source=str(path))

    def load_pdf_docling(self, path: str | Path) -> dict[str, int]:
        """Parse *path* with Docling for text + tables (+ images if configured).

        Returns counts dict ``{"text": N, "tables": M, "images": K}``.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption  # lazy
            from docling.datamodel.base_models import InputFormat  # lazy
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # lazy
        except ImportError as exc:
            logger.warning("Docling not available, falling back to standard PDF loading: %s", exc)
            self.load_pdf(path)
            return {"text": self._total_chunks, "tables": 0, "images": 0}

        path = Path(path)
        source = str(path)
        pipeline_opts = PdfPipelineOptions(
            do_ocr=False,
            generate_picture_images=self._config.include_images,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        doc = converter.convert(source=source).document

        counts = {"text": 0, "tables": 0, "images": 0}
        texts_to_index: list[str] = []
        metas: list[dict[str, Any]] = []

        for chunk_text in self._docling_chunks(doc):
            texts_to_index.append(chunk_text)
            metas.append({"source": source, "modality": "text"})
            counts["text"] += 1

        if self._config.include_tables:
            for table in doc.tables:
                try:
                    md = table.export_to_markdown()
                    texts_to_index.append(md)
                    metas.append({"source": source, "modality": "table"})
                    counts["tables"] += 1
                except Exception as exc:
                    logger.debug("Table export failed: %s", exc)

        self._ensure_store()
        if texts_to_index:
            self._vector_store.add_texts(texts_to_index, metadatas=metas)
            self._indexed_sources.append(source)
            self._total_chunks += len(texts_to_index)
        logger.info(
            "[IBMProductionRAG] Docling load: %d text, %d tables from %s",
            counts["text"], counts["tables"], path.name,
        )
        return counts

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> IBMRAGResult:
        """Answer *question* with retry logic and latency tracking.

        Args:
            question: Natural language query.

        Returns:
            :class:`IBMRAGResult` with answer, sources, and performance metrics.
        """
        if not self.is_ready:
            return IBMRAGResult(
                query=question,
                answer="No documents indexed. Please load documents before querying.",
            )

        t0 = time.monotonic()
        docs = self._vector_store.similarity_search(question, k=self._config.top_k)
        passages = [d.page_content for d in docs]
        sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))

        context = "\n\n".join(passages)
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        answer, retries = self._generate_with_retry(prompt)
        latency = (time.monotonic() - t0) * 1000

        return IBMRAGResult(
            query=question,
            answer=answer.strip(),
            retrieved_passages=passages,
            sources=sources,
            latency_ms=round(latency, 1),
            retries_used=retries,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_with_retry(self, prompt: str) -> tuple[str, int]:
        """Call the LLM with exponential-backoff retries. Returns (answer, retries_used)."""
        retries = 0
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                answer = self._generate(prompt)
                return answer, retries
            except Exception as exc:
                last_exc = exc
                retries += 1
                delay = self._config.retry_delay * (2 ** attempt)
                logger.warning(
                    "[IBMProductionRAG] Attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1, exc, delay,
                )
                time.sleep(delay)

        logger.error("[IBMProductionRAG] All retries exhausted: %s", last_exc)
        return "Failed to generate an answer after multiple retries.", retries

    def _generate(self, prompt: str) -> str:
        if self._llm is not None:
            return self._llm.complete(prompt)
        if self._local_gen is None:
            from transformers import pipeline  # lazy

            self._local_gen = pipeline(
                "text2text-generation",
                model=self._config.generation_model,
                max_new_tokens=256,
            )
        return self._local_gen(prompt)[0]["generated_text"]

    def _ensure_store(self) -> None:
        if self._vector_store is not None:
            return
        from langchain_community.vectorstores import Chroma  # lazy
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self._config.embedding_model)
        except Exception:
            # Fallback to a reliable model if IBM Granite embedding fails
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self._vector_store = Chroma(
            collection_name=self._config.collection_name,
            embedding_function=embeddings,
            persist_directory=self._config.persist_dir,
        )

    def _chunk(self, text: str) -> list[str]:
        size, overlap = self._config.chunk_size, self._config.chunk_overlap
        chunks, start = [], 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks

    def _docling_chunks(self, doc: Any) -> list[str]:
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # lazy
            from docling_core.types.doc.document import TableItem  # lazy

            return [
                chunk.text
                for chunk in HybridChunker().chunk(doc)
                if chunk.text.strip()
                and not (len(chunk.meta.doc_items) == 1 and isinstance(chunk.meta.doc_items[0], TableItem))
            ]
        except Exception:
            full = doc.export_to_text() if hasattr(doc, "export_to_text") else ""
            return self._chunk(full)
