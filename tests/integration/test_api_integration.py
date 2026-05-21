"""Integration tests for external API connections.

These tests are skipped automatically when the corresponding environment
variables are not set, so they are safe to run in CI without live credentials.
"""
from __future__ import annotations

import os

import pytest


# ─── Markers / skip helpers ───────────────────────────────────────────────────

def _requires_env(*vars_: str):
    """Skip the test if any of the given environment variables are missing."""
    missing = [v for v in vars_ if not os.environ.get(v)]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"Environment variable(s) not set: {', '.join(missing)}",
    )


# ─── OpenAI API ───────────────────────────────────────────────────────────────

@pytest.mark.integration
@_requires_env("OPENAI_API_KEY")
class TestOpenAIIntegration:
    def test_simple_completion(self):
        from src.core.model_factory import create_llm

        llm = create_llm("openai", model="gpt-4o-mini", max_tokens=50, temperature=0.0)
        answer = llm.chat("Reply with the single word: pong")
        assert "pong" in answer.lower()

    def test_embedding_creation(self):
        from src.rag.embedder import Embedder

        emb = Embedder(provider="openai", model="text-embedding-3-small")
        vector = emb.embed("Hello world")
        assert len(vector) == 1536
        assert all(isinstance(x, float) for x in vector)


# ─── Anthropic API ────────────────────────────────────────────────────────────

@pytest.mark.integration
@_requires_env("ANTHROPIC_API_KEY")
class TestAnthropicIntegration:
    def test_simple_completion(self):
        from src.core.model_factory import create_llm

        llm = create_llm(
            "anthropic",
            model="claude-3-haiku-20240307",
            max_tokens=50,
            temperature=0.0,
        )
        answer = llm.chat("Reply with the single word: pong")
        assert "pong" in answer.lower()


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestChromaDBIntegration:
    def test_upsert_and_search(self, tmp_path):
        pytest.importorskip("chromadb")

        from src.rag.vector_store import Document, VectorStore

        store = VectorStore(
            provider="chroma",
            collection_name="test_col",
            persist_directory=str(tmp_path),
        )
        doc = Document(id="doc1", text="ChromaDB test", embedding=[0.1] * 10)
        store.upsert([doc])
        assert store.count() == 1

        results = store.search(query_embedding=[0.1] * 10, top_k=1)
        assert len(results) == 1
        assert results[0].document.id == "doc1"

        store.delete(["doc1"])
        assert store.count() == 0
