"""End-to-end integration tests for the full RAG pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.base_llm import LLMConfig, LLMResponse
from src.inference.inference_engine import InferenceEngine, InferenceRequest
from src.processing.chunking import TextChunker
from src.processing.preprocessing import TextPreprocessor
from src.rag.embedder import Embedder
from src.rag.indexer import Indexer
from src.rag.retriever import Retriever, RetrievalConfig
from src.rag.vector_store import Document, VectorStore


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed.return_value = [0.1] * 10
    embedder.embed_batch.return_value = [[0.1] * 10, [0.2] * 10]
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []
    store.count.return_value = 0
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        content="The answer is 42.", model="gpt-4o"
    )
    llm.config = LLMConfig(model="gpt-4o")
    return llm


# ─── Preprocessing → Chunking pipeline ───────────────────────────────────────

class TestPreprocessChunkPipeline:
    def test_full_text_pipeline(self):
        raw = "  Hello <b>World</b>!! This is a test.  Extra   spaces.  "
        preprocessor = TextPreprocessor(remove_html=True)
        clean = preprocessor.process(raw)
        chunker = TextChunker()
        chunks = chunker.split(clean)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        assert "<b>" not in chunks[0]


# ─── Indexing pipeline ────────────────────────────────────────────────────────

class TestIndexingPipeline:
    def test_index_text(self, mock_embedder, mock_vector_store):
        indexer = Indexer(embedder=mock_embedder, vector_store=mock_vector_store)
        count = indexer.index_text("Sentence one. Sentence two. Sentence three.", source="test")
        assert count >= 1
        mock_vector_store.upsert.assert_called_once()

    def test_index_file_not_found_raises(self, mock_embedder, mock_vector_store):
        indexer = Indexer(embedder=mock_embedder, vector_store=mock_vector_store)
        with pytest.raises(FileNotFoundError):
            indexer.index_file("/nonexistent/path/file.txt")


# ─── Retrieval pipeline ───────────────────────────────────────────────────────

class TestRetrievalPipeline:
    def test_retrieve_returns_results(self, mock_embedder, mock_vector_store):
        from src.rag.vector_store import SearchResult

        doc = Document(id="d1", text="The answer is 42.", metadata={"source": "test.txt"})
        mock_vector_store.search.return_value = [SearchResult(document=doc, score=0.9)]

        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=RetrievalConfig(top_k=3),
        )
        results = retriever.retrieve("What is the answer?")
        assert len(results) == 1
        assert results[0].document.text == "The answer is 42."
        assert results[0].score == 0.9

    def test_similarity_threshold_filters(self, mock_embedder, mock_vector_store):
        from src.rag.vector_store import SearchResult

        doc_high = Document(id="d1", text="High relevance", metadata={})
        doc_low = Document(id="d2", text="Low relevance", metadata={})
        mock_vector_store.search.return_value = [
            SearchResult(document=doc_high, score=0.9),
            SearchResult(document=doc_low, score=0.3),
        ]

        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=RetrievalConfig(similarity_threshold=0.5),
        )
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert results[0].document.text == "High relevance"


# ─── Full RAG inference pipeline ─────────────────────────────────────────────

class TestInferencePipeline:
    def test_rag_inference_end_to_end(self, mock_embedder, mock_vector_store, mock_llm):
        from src.rag.vector_store import SearchResult

        doc = Document(id="d1", text="42 is the answer.", metadata={"source": "book.txt"})
        mock_vector_store.search.return_value = [SearchResult(document=doc, score=0.95)]

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        engine = InferenceEngine(llm=mock_llm, retriever=retriever)

        request = InferenceRequest(query="What is the answer?", use_rag=True)
        result = engine.run(request)

        assert result.answer == "The answer is 42."
        assert "book.txt" in result.sources
        assert len(result.retrieved_chunks) == 1

    def test_inference_without_rag(self, mock_llm):
        engine = InferenceEngine(llm=mock_llm)
        request = InferenceRequest(query="Tell me a joke.", use_rag=False)
        result = engine.run(request)
        assert result.answer == "The answer is 42."
        assert result.retrieved_chunks == []
