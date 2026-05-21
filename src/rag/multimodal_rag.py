"""Multimodal RAG — text + tables + image descriptions from PDFs.

Project #8: Uses Docling to parse PDFs into text, tables (markdown), and
image descriptions (via a vision LLM). All modalities are embedded and
stored together in a shared vector store for unified retrieval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MultimodalRAGConfig:
    """Configuration for :class:`MultimodalRAGPipeline`."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """HuggingFace sentence-transformers model for embeddings."""

    vision_model: str | None = None
    """Vision LLM for image captioning.
    Set to a Bedrock multimodal model, e.g. ``amazon.nova-pro-v1:0``
    (or ``amazon.titan-multimodal-embeddings-g1-v1``).
    If None, images are skipped."""

    generation_model: str = "google/flan-t5-base"
    """Local model for answer generation.
    Bedrock alternative: ``anthropic.claude-3-sonnet-20240229-v1:0``."""

    top_k: int = 5
    """Number of chunks to retrieve per query."""

    chunk_size: int = 800
    """Characters per text chunk."""

    chunk_overlap: int = 100
    """Overlap between text chunks."""

    persist_dir: str = "data/vectordb/multimodal_rag"
    """Directory for persisting the vector store."""

    image_prompt: str = "If the image contains text or data, describe it in detail."
    """Prompt used by the vision model to caption images."""


@dataclass
class MultimodalRAGResult:
    """Result of a :class:`MultimodalRAGPipeline` query."""

    query: str
    answer: str
    retrieved_passages: list[str] = field(default_factory=list)
    modalities_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class _VisionLLM:
    """Thin wrapper for calling a vision LLM that accepts base64-encoded images.

    Currently supported backends:
    - Bedrock (``amazon.nova-pro-v1:0``) — inject via *llm*
    - HuggingFace BLIP / LLaVA (local) — automatic fallback

    Args:
        llm: BaseLLM instance that supports image input (multimodal Bedrock models).
        model_name: HF model fallback (``Salesforce/blip-image-captioning-base``).
    """

    def __init__(
        self,
        llm: Any | None = None,
        model_name: str = "Salesforce/blip-image-captioning-base",
        prompt: str = "Describe this image.",
    ) -> None:
        self._llm = llm
        self._model_name = model_name
        self._prompt = prompt
        self._hf_pipeline: Any = None

    def caption(self, image: Any) -> str:
        """Generate a text description of *image* (PIL.Image).

        Args:
            image: A ``PIL.Image.Image`` object.

        Returns:
            Text caption/description of the image.
        """
        if self._llm is not None:
            return self._caption_via_llm(image)
        return self._caption_via_hf(image)

    def _caption_via_hf(self, image: Any) -> str:
        import PIL.Image  # lazy import

        if self._hf_pipeline is None:
            from transformers import pipeline  # lazy import

            self._hf_pipeline = pipeline(
                "image-to-text",
                model=self._model_name,
            )
            logger.info("Loaded vision model: %s", self._model_name)
        results = self._hf_pipeline(image)
        if results:
            return results[0].get("generated_text", "")
        return ""

    def _caption_via_llm(self, image: Any) -> str:
        """Caption via an injected multimodal LLM (e.g. Bedrock Nova Pro).

        Encodes the image as base64 and calls the LLM's chat method with
        an image content block. Falls back to HF on error.
        """
        import base64
        import io

        try:
            import PIL.Image  # lazy import
            import PIL.ImageOps

            image = PIL.ImageOps.exif_transpose(image) or image
            image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Try Bedrock-style multimodal call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": base64.b64decode(b64)},
                            }
                        },
                        {"text": self._prompt},
                    ],
                }
            ]
            return self._llm.complete(self._prompt, messages=messages)
        except Exception as exc:
            logger.warning("Vision LLM captioning failed, falling back to HF: %s", exc)
            return self._caption_via_hf(image)


class MultimodalRAGPipeline:
    """RAG pipeline that ingests text, tables, and images from PDF documents.

    Workflow:
    1. ``load_pdf(path)``     — parse with Docling → extract text + tables + images
    2. ``query(question)``    — embed query → retrieve across all modalities → generate

    Args:
        config: :class:`MultimodalRAGConfig` instance.
        llm: Optional text generation LLM. Bedrock recommended for production.
        vision_llm: Optional vision LLM for image captioning.
                    If None and ``config.vision_model`` is None, images are skipped.

    Note:
        Requires ``docling`` (``pip install docling``) and ``pillow``.
        These are heavy dependencies — install them separately.

    Bedrock example::

        from src.core.model_factory import ModelFactory
        llm = ModelFactory.create_llm("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        pipeline = MultimodalRAGPipeline(llm=llm)
        pipeline.load_pdf("report.pdf")
        result = pipeline.query("What were the total revenues?")
    """

    def __init__(
        self,
        config: MultimodalRAGConfig | None = None,
        llm: Any | None = None,
        vision_llm: Any | None = None,
    ) -> None:
        self._config = config or MultimodalRAGConfig()
        self._llm = llm
        self._vision_llm = _VisionLLM(
            llm=vision_llm,
            prompt=self._config.image_prompt,
        )
        self._passages: list[dict[str, str]] = []  # {"text": ..., "modality": ..., "source": ...}
        self._embedder: Any = None
        self._faiss_index: Any = None
        self._faiss_texts: list[str] = []
        self._faiss_meta: list[dict[str, str]] = []
        self._local_gen: Any = None

    # ── Document loading ──────────────────────────────────────────────────────

    def load_pdf(self, path: str | Path) -> dict[str, int]:
        """Parse *path* with Docling and index text, tables, and images.

        Args:
            path: Path to a PDF file.

        Returns:
            Dict with counts: ``{"text": N, "tables": M, "images": K}``.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption  # lazy
            from docling.datamodel.base_models import InputFormat  # lazy
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # lazy
        except ImportError as exc:
            raise ImportError(
                "docling is not installed. Install it with: pip install docling"
            ) from exc

        path = Path(path)
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            generate_picture_images=True,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        docling_doc = converter.convert(source=str(path)).document
        source = str(path)

        counts = {"text": 0, "tables": 0, "images": 0}
        texts_to_index: list[str] = []
        metas: list[dict[str, str]] = []

        # Text chunks
        for chunk_text in self._docling_text_chunks(docling_doc):
            texts_to_index.append(chunk_text)
            metas.append({"modality": "text", "source": source})
            counts["text"] += 1

        # Tables → markdown
        for table in docling_doc.tables:
            try:
                md = table.export_to_markdown()
                texts_to_index.append(md)
                metas.append({"modality": "table", "source": source})
                counts["tables"] += 1
            except Exception as exc:
                logger.debug("Table export failed: %s", exc)

        # Images → captions
        if self._config.vision_model is not None or self._vision_llm._llm is not None:
            for picture in docling_doc.pictures:
                try:
                    import PIL.Image  # lazy

                    image = picture.get_image(docling_doc)
                    if image:
                        caption = self._vision_llm.caption(image)
                        if caption:
                            texts_to_index.append(caption)
                            metas.append({"modality": "image", "source": source})
                            counts["images"] += 1
                except Exception as exc:
                    logger.debug("Image captioning failed: %s", exc)
        else:
            logger.debug("Skipping images (no vision model configured)")

        self._add_to_index(texts_to_index, metas)
        logger.info(
            "Loaded %s: %d text, %d table, %d image passages",
            path.name, counts["text"], counts["tables"], counts["images"],
        )
        return counts

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> MultimodalRAGResult:
        """Answer *question* using indexed multimodal passages."""
        if not self._faiss_texts:
            raise RuntimeError("No documents indexed. Call load_pdf() first.")

        top_passages, top_metas = self._retrieve(question)
        modalities = list(dict.fromkeys(m["modality"] for m in top_metas))
        sources = list(dict.fromkeys(m["source"] for m in top_metas))

        context = "\n\n".join(top_passages)
        prompt = (
            "Answer the question using only the context below (which may include text, tables, and image descriptions).\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = self._generate(prompt)

        return MultimodalRAGResult(
            query=question,
            answer=answer.strip(),
            retrieved_passages=top_passages,
            modalities_used=modalities,
            sources=sources,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _docling_text_chunks(self, doc: Any) -> list[str]:
        """Chunk text content from a Docling document."""
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # lazy
            from docling_core.types.doc.document import TableItem  # lazy

            chunks = []
            for chunk in HybridChunker().chunk(doc):
                items = chunk.meta.doc_items
                if len(items) == 1 and isinstance(items[0], TableItem):
                    continue
                if chunk.text.strip():
                    chunks.append(chunk.text)
            return chunks
        except Exception:
            # Fallback: export full document text and split
            full_text = doc.export_to_text() if hasattr(doc, "export_to_text") else ""
            return self._simple_split(full_text)

    def _simple_split(self, text: str) -> list[str]:
        size, overlap = self._config.chunk_size, self._config.chunk_overlap
        chunks, start = [], 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks

    def _add_to_index(self, texts: list[str], metas: list[dict[str, str]]) -> None:
        """Embed *texts* and add to in-memory FAISS index."""
        if not texts:
            return
        import faiss  # lazy
        import numpy as np  # lazy
        from sentence_transformers import SentenceTransformer  # lazy

        if self._embedder is None:
            self._embedder = SentenceTransformer(self._config.embedding_model)

        vecs = self._embedder.encode(texts, show_progress_bar=False).astype(np.float32)
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatL2(vecs.shape[1])

        self._faiss_index.add(vecs)
        self._faiss_texts.extend(texts)
        self._faiss_meta.extend(metas)

    def _retrieve(self, query: str) -> tuple[list[str], list[dict[str, str]]]:
        import numpy as np  # lazy

        q_vec = self._embedder.encode([query], show_progress_bar=False).astype(np.float32)
        k = min(self._config.top_k, len(self._faiss_texts))
        _, indices = self._faiss_index.search(q_vec, k)
        passages = [self._faiss_texts[i] for i in indices[0] if 0 <= i < len(self._faiss_texts)]
        metas = [self._faiss_meta[i] for i in indices[0] if 0 <= i < len(self._faiss_meta)]
        return passages, metas

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
