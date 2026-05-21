"""Tests for src/knowledge_graph/graph_store.py and triple_extractor.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Triple tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTriple:
    def test_triple_immutable(self):
        from src.knowledge_graph.graph_store import Triple
        t = Triple(head="A", relation="knows", tail="B")
        with pytest.raises((AttributeError, TypeError)):
            t.head = "C"  # type: ignore[misc]

    def test_triple_str(self):
        from src.knowledge_graph.graph_store import Triple
        t = Triple("A", "knows", "B")
        assert "(A)" in str(t)
        assert "knows" in str(t)
        assert "(B)" in str(t)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeGraphStore tests
# ─────────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraphStore:
    def _make_store(self):
        pytest.importorskip("networkx")
        from src.knowledge_graph.graph_store import KnowledgeGraphStore
        return KnowledgeGraphStore(name="test_graph")

    def _make_triple(self, head="A", relation="rel", tail="B"):
        from src.knowledge_graph.graph_store import Triple
        return Triple(head=head, relation=relation, tail=tail)

    def test_add_triple_increments_edges(self):
        store = self._make_store()
        store.add_triple(self._make_triple())
        assert store.edge_count == 1
        assert store.node_count == 2

    def test_add_triples_bulk(self):
        store = self._make_store()
        triples = [
            self._make_triple("A", "knows", "B"),
            self._make_triple("B", "likes", "C"),
            self._make_triple("C", "works_at", "D"),
        ]
        store.add_triples(triples)
        assert store.edge_count == 3
        assert store.node_count == 4

    def test_search_exact_entity(self):
        store = self._make_store()
        store.add_triple(self._make_triple("Einstein", "developed", "General Relativity"))
        result = store.search("Einstein")
        assert result.query_entity == "Einstein"
        assert result.total_triples > 0
        assert "Einstein" in result.context

    def test_search_case_insensitive_prefix(self):
        store = self._make_store()
        store.add_triple(self._make_triple("Albert Einstein", "born_in", "Ulm"))
        result = store.search("albert")  # lowercase prefix
        assert result.total_triples > 0

    def test_search_nonexistent_entity(self):
        store = self._make_store()
        result = store.search("XYZ_NONEXISTENT")
        assert result.total_triples == 0
        assert "No information" in result.context

    def test_dfs_max_depth_respected(self):
        store = self._make_store()
        # Chain: A → B → C → D (depth 3)
        store.add_triples([
            self._make_triple("A", "rel", "B"),
            self._make_triple("B", "rel", "C"),
            self._make_triple("C", "rel", "D"),
        ])
        result_d1 = store.search("A", max_depth=1)
        result_d2 = store.search("A", max_depth=2)
        # depth 2 should find more triples
        assert result_d2.total_triples >= result_d1.total_triples

    def test_get_context_multi_token(self):
        store = self._make_store()
        store.add_triple(self._make_triple("Python", "is", "programming language"))
        context = store.get_context("Tell me about Python programming")
        assert "Python" in context or "No relevant" in context

    def test_get_context_no_match(self):
        store = self._make_store()
        context = store.get_context("some unknown xyz query")
        assert "No relevant graph context found" in context

    def test_save_load_roundtrip(self, tmp_path):
        store = self._make_store()
        store.add_triple(self._make_triple("Paris", "capital_of", "France"))
        path = tmp_path / "graph.json"
        store.save(path)

        loaded = type(store).load(path)
        assert loaded.name == "test_graph"
        assert loaded.edge_count == 1
        assert len(loaded.triples) == 1

    def test_load_missing_file_raises(self):
        from src.knowledge_graph.graph_store import KnowledgeGraphStore
        with pytest.raises(FileNotFoundError):
            KnowledgeGraphStore.load("/nonexistent/graph.json")

    def test_triples_property_returns_copy(self):
        store = self._make_store()
        store.add_triple(self._make_triple())
        triples = store.triples
        triples.clear()  # mutate the returned list
        assert len(store.triples) == 1  # original unchanged

    def test_context_deduplicates(self):
        store = self._make_store()
        # Two identical triples should not produce duplicate lines
        store.add_triple(self._make_triple("A", "rel", "B"))
        store.add_triple(self._make_triple("A", "rel", "B"))
        result = store.search("A")
        lines = result.context.strip().split("\n")
        assert len(lines) == len(set(lines))


# ─────────────────────────────────────────────────────────────────────────────
# TripleExtractor tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTripleExtractor:
    def _make_mock_llm(self, response: str):
        llm = MagicMock()
        llm.complete.return_value = response
        return llm

    def test_extract_parses_valid_json(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        response = '[{"head": "Einstein", "relation": "developed", "tail": "Relativity"}]'
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm, chunk_size=2000)
        triples = extractor.extract("Albert Einstein developed the theory of general relativity.")
        assert len(triples) == 1
        assert triples[0].head == "Einstein"
        assert triples[0].relation == "developed"
        assert triples[0].tail == "Relativity"

    def test_extract_handles_invalid_json(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        llm = self._make_mock_llm("not valid json at all")
        extractor = TripleExtractor(llm=llm)
        triples = extractor.extract("some text")
        assert triples == []

    def test_extract_deduplicates_triples(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        response = json.dumps([
            {"head": "A", "relation": "rel", "tail": "B"},
            {"head": "A", "relation": "rel", "tail": "B"},  # duplicate
        ])
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm, deduplicate=True)
        triples = extractor.extract("text")
        assert len(triples) == 1

    def test_extract_no_dedup(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        response = json.dumps([
            {"head": "A", "relation": "rel", "tail": "B"},
            {"head": "A", "relation": "rel", "tail": "B"},
        ])
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm, deduplicate=False)
        triples = extractor.extract("text")
        assert len(triples) == 2

    def test_extract_skips_missing_fields(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        response = json.dumps([
            {"head": "A", "relation": "rel"},  # missing tail
            {"head": "", "relation": "rel", "tail": "B"},  # empty head
            {"head": "C", "relation": "knows", "tail": "D"},  # valid
        ])
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm)
        triples = extractor.extract("text")
        assert len(triples) == 1
        assert triples[0].head == "C"

    def test_llm_failure_returns_empty(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM error")
        extractor = TripleExtractor(llm=llm)
        triples = extractor.extract("text")
        assert triples == []

    def test_long_text_chunked(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        response = json.dumps([{"head": "A", "relation": "rel", "tail": "B"}])
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm, chunk_size=50, deduplicate=False)
        long_text = "A relates to B. " * 20  # 320 chars → multiple chunks
        triples = extractor.extract(long_text)
        # Each chunk produces 1 triple → multiple calls
        assert llm.complete.call_count > 1
        # All triples returned
        assert len(triples) == llm.complete.call_count

    def test_json_embedded_in_markdown(self):
        from src.knowledge_graph.triple_extractor import TripleExtractor

        # LLM wraps JSON in markdown
        response = "Here are the triples:\n```json\n[{\"head\":\"X\",\"relation\":\"is\",\"tail\":\"Y\"}]\n```"
        llm = self._make_mock_llm(response)
        extractor = TripleExtractor(llm=llm)
        triples = extractor.extract("X is Y.")
        assert len(triples) == 1
