"""Unit tests for RAG components: VectorStore, Embedder, Indexer, Retriever."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.vector_store import Document, SearchResult, VectorStore
from src.rag.embedder import Embedder
from src.rag.indexer import Indexer, _doc_id
from src.rag.retriever import Retriever, RetrievalConfig


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStoreChroma:
    def _make_store_with_mock(self):
        mock_collection = MagicMock()
        with patch("chromadb.PersistentClient") as mock_client_cls:
            mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection
            store = VectorStore(provider="chroma")
            store._get_store()  # trigger init
        return store, mock_collection

    def test_upsert_calls_collection(self):
        store, mock_col = self._make_store_with_mock()
        docs = [Document(id="d1", text="Hello", embedding=[0.1, 0.2])]
        store.upsert(docs)
        mock_col.upsert.assert_called_once()

    def test_search_returns_results(self):
        store, mock_col = self._make_store_with_mock()
        mock_col.query.return_value = {
            "ids": [["d1"]],
            "documents": [["Hello"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.1]],
        }
        results = store.search([0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.id == "d1"
        assert results[0].score == pytest.approx(0.9)  # 1.0 - 0.1

    def test_search_with_where_filter(self):
        store, mock_col = self._make_store_with_mock()
        mock_col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        store.search([0.1], top_k=3, where={"source": "file.txt"})
        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs.get("where") == {"source": "file.txt"}

    def test_count_returns_int(self):
        store, mock_col = self._make_store_with_mock()
        mock_col.count.return_value = 42
        assert store.count() == 42

    def test_delete_calls_collection(self):
        store, mock_col = self._make_store_with_mock()
        store.delete(["d1", "d2"])
        mock_col.delete.assert_called_once_with(ids=["d1", "d2"])

    def test_unknown_provider_raises(self):
        store = VectorStore(provider="unknown")
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            store._get_store()

    def test_missing_chromadb_raises(self):
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "chromadb":
                raise ImportError("no chromadb")
            return real_import(name, *a, **kw)

        store = VectorStore(provider="chroma")
        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="chromadb is required"):
                store._init_chroma()


class TestVectorStoreFaiss:
    def _make_faiss_store(self):
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        mock_faiss.IndexFlatIP.return_value.ntotal = 0

        with patch("src.rag.vector_store.VectorStore._init_faiss") as mock_init:
            store = VectorStore(provider="faiss")
            store._store = {
                "faiss": mock_faiss,
                "index": None,
                "docs": {},
            }
        return store, mock_faiss

    def test_faiss_search_empty_index_returns_empty(self):
        store, _ = self._make_faiss_store()
        # index is None → empty result
        results = store._faiss_search(store._store, [0.1, 0.2], top_k=5)
        assert results == []

    def test_count_faiss(self):
        store, _ = self._make_faiss_store()
        store._store["docs"] = {"d1": MagicMock(), "d2": MagicMock()}
        assert store.count() == 2

    def test_faiss_upsert_no_embeddings_skips(self):
        store, _ = self._make_faiss_store()
        docs = [Document(id="d1", text="hello")]  # no embedding
        store._faiss_upsert(store._store, docs)
        assert store._store["index"] is None  # nothing added

    def test_missing_faiss_raises(self):
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **kw):
            if name == "faiss":
                raise ImportError("no faiss")
            return real_import(name, *a, **kw)

        store = VectorStore(provider="faiss")
        with patch("builtins.__import__", side_effect=_block):
            with pytest.raises(ImportError, match="faiss-cpu is required"):
                store._init_faiss()

    def test_get_store_calls_init_faiss(self):
        """Covers line 57: FAISS path in _get_store()."""
        import sys as _sys
        mock_faiss = MagicMock()
        with patch.dict(_sys.modules, {"faiss": mock_faiss}):
            store = VectorStore(provider="faiss")
            result = store._get_store()
        assert result["faiss"] is mock_faiss
        assert result["index"] is None

    def test_faiss_upsert_with_embeddings(self):
        """Covers _faiss_upsert lines 172-180 with actual document embeddings."""
        import sys as _sys
        import numpy as np
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        store = VectorStore(provider="faiss")
        faiss_store = {"faiss": mock_faiss, "index": None, "docs": {}}
        docs = [Document(id="d1", text="hello", embedding=[0.1, 0.2, 0.3])]
        store._faiss_upsert(faiss_store, docs)

        mock_faiss.IndexFlatIP.assert_called_once_with(3)
        assert faiss_store["docs"]["d1"] is docs[0]

    def test_faiss_search_with_results(self):
        """Covers _faiss_search lines 188-199 with a non-empty index."""
        import numpy as np
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 1
        doc1 = Document(id="d1", text="hello", embedding=[0.1, 0.2])
        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))

        faiss_store = {"faiss": mock_faiss, "index": mock_index, "docs": {"d1": doc1}}
        store = VectorStore(provider="faiss")
        results = store._faiss_search(faiss_store, [0.1, 0.2], top_k=1)

        assert len(results) == 1
        assert results[0].document.id == "d1"
        assert results[0].score == pytest.approx(0.9)

    def test_count_unknown_provider_returns_zero(self):
        """Covers line 137: count() returns 0 for unexpected provider."""
        store = VectorStore(provider="chroma")
        store._provider = "other"
        store._store = MagicMock()  # pre-set so _get_store() returns immediately
        assert store.count() == 0

    def test_search_unknown_provider_raises(self):
        """Covers line 121: search raises for unknown provider."""
        store = VectorStore(provider="chroma")
        store._provider = "other"
        store._store = MagicMock()
        with pytest.raises(ValueError, match="Unknown provider"):
            store.search([0.1], top_k=1)


# ── Embedder ──────────────────────────────────────────────────────────────────

class TestEmbedderOpenAI:
    def _make_embedder(self):
        mock_openai_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.return_value.data = [mock_item]

        with patch("openai.OpenAI", return_value=mock_openai_client):
            embedder = Embedder(provider="openai", api_key="sk-test")
            embedder._client = mock_openai_client

        return embedder, mock_openai_client

    def test_embed_returns_vector(self):
        embedder, _ = self._make_embedder()
        result = embedder.embed("Hello world")
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_empty_returns_empty(self):
        embedder, _ = self._make_embedder()
        assert embedder.embed_batch([]) == []

    def test_embed_batch_calls_api(self):
        embedder, mock_client = self._make_embedder()
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.3, 0.4]
        mock_client.embeddings.create.return_value.data = [mock_item1, mock_item2]

        results = embedder.embed_batch(["text1", "text2"])
        assert len(results) == 2

    def test_embed_batch_batching(self):
        """Verify large inputs are split into batches."""
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1]
        mock_client.embeddings.create.return_value.data = [mock_item]

        embedder = Embedder(provider="openai", batch_size=2, api_key="sk-test")
        embedder._client = mock_client

        # 5 texts with batch_size=2 → 3 API calls
        embedder.embed_batch(["t"] * 5)
        assert mock_client.embeddings.create.call_count == 3

    def test_unknown_provider_raises(self):
        embedder = Embedder(provider="unknown")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            embedder._embed_batch_impl(["text"])

    def test_aembed_batch_async(self):
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.5]
        mock_client.embeddings.create.return_value.data = [mock_item]

        embedder = Embedder(provider="openai", api_key="sk-test")
        embedder._client = mock_client

        result = asyncio.run(embedder.aembed_batch(["hello"]))
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_openai_client_lazy_init(self):
        """Covers embedder lines 35, 37: _get_openai_client() creates client on first call."""
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            embedder = Embedder(provider="openai", api_key="sk-test")
            client = embedder._get_openai_client()
        assert client is mock_client
        mock_cls.assert_called_once_with(api_key="sk-test")

    def test_hf_embed_batch(self):
        """Covers embedder lines 41-50, 79-80: _get_hf_model() and HF embed path."""
        import sys as _sys
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2, 0.3]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(_sys.modules, {"sentence_transformers": mock_st}):
            embedder = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
            result = embedder.embed_batch(["hello world"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_st.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_hf_missing_sentence_transformers_raises(self):
        """Covers ImportError path in _get_hf_model()."""
        import sys as _sys
        with patch.dict(_sys.modules, {"sentence_transformers": None}):
            embedder = Embedder(provider="huggingface")
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                embedder._get_hf_model()


# ── Indexer ───────────────────────────────────────────────────────────────────

class TestIndexer:
    def _make_indexer(self):
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_batch.return_value = [[0.1, 0.2]]

        mock_store = MagicMock(spec=VectorStore)

        indexer = Indexer(embedder=mock_embedder, vector_store=mock_store)
        return indexer, mock_embedder, mock_store

    def test_index_text_returns_chunk_count(self):
        indexer, _, mock_store = self._make_indexer()
        count = indexer.index_text("This is a test sentence. And another one.", source="test")
        assert count >= 1
        mock_store.upsert.assert_called_once()

    def test_index_file_not_found_raises(self):
        indexer, _, _ = self._make_indexer()
        with pytest.raises(FileNotFoundError):
            indexer.index_file("/nonexistent/file.txt")

    def test_index_file_reads_and_indexes(self, tmp_path):
        indexer, mock_embedder, mock_store = self._make_indexer()
        f = tmp_path / "doc.txt"
        f.write_text("Some text content for indexing.", encoding="utf-8")
        count = indexer.index_file(f)
        assert count >= 1

    def test_index_directory_indexes_all_files(self, tmp_path):
        indexer, mock_embedder, mock_store = self._make_indexer()
        (tmp_path / "a.txt").write_text("File A content.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B content.", encoding="utf-8")
        total = indexer.index_directory(tmp_path, glob="*.txt")
        assert total >= 2

    def test_index_directory_not_found_raises(self):
        indexer, _, _ = self._make_indexer()
        with pytest.raises(NotADirectoryError):
            indexer.index_directory("/nonexistent/dir")

    def test_index_directory_skips_failed_files(self, tmp_path):
        """Files that fail to load are warned and skipped."""
        indexer, _, mock_store = self._make_indexer()
        (tmp_path / "ok.txt").write_text("Good content.", encoding="utf-8")

        def _bad_loader(path):
            if path.name == "bad.txt":
                raise RuntimeError("bad file")
            return path.read_text(encoding="utf-8")

        indexer._loader = _bad_loader
        (tmp_path / "bad.txt").write_text("bad", encoding="utf-8")
        total = indexer.index_directory(tmp_path, glob="*.txt")
        assert total >= 1  # ok.txt was indexed

    def test_doc_id_is_deterministic(self):
        id1 = _doc_id("file.txt", 0)
        id2 = _doc_id("file.txt", 0)
        id3 = _doc_id("file.txt", 1)
        assert id1 == id2
        assert id1 != id3


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:
    def _make_retriever(self):
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        mock_store = MagicMock(spec=VectorStore)
        doc = Document(id="d1", text="Relevant chunk.", metadata={"source": "file.txt"})
        mock_store.search.return_value = [SearchResult(document=doc, score=0.9)]

        retriever = Retriever(embedder=mock_embedder, vector_store=mock_store)
        return retriever, mock_embedder, mock_store

    def test_retrieve_returns_search_results(self):
        retriever, _, _ = self._make_retriever()
        results = retriever.retrieve("test query", top_k=3)
        assert len(results) == 1
        assert results[0].document.text == "Relevant chunk."

    def test_retrieve_empty_query_returns_empty(self):
        retriever, _, _ = self._make_retriever()
        results = retriever.retrieve("", top_k=5)
        assert results == []

    def test_retrieve_calls_embedder_and_store(self):
        retriever, mock_embedder, mock_store = self._make_retriever()
        retriever.retrieve("test query", top_k=2)
        mock_embedder.embed.assert_called_once_with("test query")
        mock_store.search.assert_called_once()

    def test_retrieve_with_metadata_filter(self):
        from src.rag.retriever import RetrievalConfig
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_store = MagicMock(spec=VectorStore)
        doc = Document(id="d1", text="Filtered chunk.", metadata={"source": "file.txt"})
        mock_store.search.return_value = [SearchResult(document=doc, score=0.9)]
        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=mock_store,
            config=RetrievalConfig(metadata_filter={"source": "file.txt"}),
        )
        retriever.retrieve("query", top_k=5)
        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs.get("where") == {"source": "file.txt"}
