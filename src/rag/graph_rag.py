"""GraphRAG — knowledge graph retrieval with NetworkX multi-hop traversal.

Project #3: Extract (head, relation, tail) triples via LLM, store in a
NetworkX DiGraph, answer queries using DFS multi-hop context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGConfig:
    """Configuration for :class:`GraphRAGPipeline`."""

    max_hops: int = 2
    """Maximum DFS depth for graph traversal."""

    chunk_size: int = 2000
    """Characters per chunk for triple extraction."""

    generation_model: str = "google/flan-t5-base"
    """Local HF model for triple extraction and answer generation.
    Bedrock alternative: ``anthropic.claude-3-haiku-20240307-v1:0``."""

    max_new_tokens: int = 512
    """Maximum tokens in generated answer."""


@dataclass
class GraphRAGResult:
    """Result of a :class:`GraphRAGPipeline` query."""

    query: str
    answer: str
    graph_context: str = ""
    triples_used: int = 0
    sources: list[str] = field(default_factory=list)


class GraphRAGPipeline:
    """RAG pipeline backed by a NetworkX knowledge graph.

    Workflow:
    1. ``build_graph(texts)`` — extract triples and populate the graph
    2. ``query(q)``           — retrieve multi-hop context → generate answer

    Args:
        config: :class:`GraphRAGConfig` instance.
        llm: LLM for triple extraction AND answer generation.
             Must implement ``complete(prompt: str) -> str``.
             Bedrock/OpenAI/Anthropic can be injected here for better extraction.
        graph: Optional pre-built :class:`~src.knowledge_graph.graph_store.KnowledgeGraphStore`.

    Example::

        from src.core.model_factory import ModelFactory
        llm = ModelFactory.create_llm("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        pipeline = GraphRAGPipeline(llm=llm)
        pipeline.build_graph(texts=["Albert Einstein developed General Relativity in 1915."])
        result = pipeline.query("What did Einstein develop?")
    """

    def __init__(
        self,
        config: GraphRAGConfig | None = None,
        llm: Any | None = None,
        graph: "KnowledgeGraphStore | None" = None,  # noqa: F821
    ) -> None:
        self._config = config or GraphRAGConfig()
        self._llm = llm or _LocalFlanT5()
        from src.knowledge_graph.graph_store import KnowledgeGraphStore

        self._graph = graph or KnowledgeGraphStore()
        self._sources: list[str] = []

    # ── Graph building ────────────────────────────────────────────────────────

    def build_graph(
        self,
        texts: list[str],
        sources: list[str] | None = None,
    ) -> int:
        """Extract triples from *texts* and add them to the knowledge graph.

        Args:
            texts: Raw text documents.
            sources: Optional provenance labels (same length as *texts*).

        Returns:
            Total number of triples added to the graph.
        """
        from src.knowledge_graph.triple_extractor import TripleExtractor

        sources = sources or ["doc_{}".format(i) for i in range(len(texts))]
        extractor = TripleExtractor(
            llm=self._llm,
            chunk_size=self._config.chunk_size,
        )
        total = 0
        for text, src in zip(texts, sources):
            triples = extractor.extract(text)
            self._graph.add_triples(triples)
            self._sources.append(src)
            total += len(triples)
            logger.info("Source %s: %d triples extracted", src, len(triples))

        logger.info(
            "Graph built: %d nodes, %d edges, %d triples total",
            self._graph.node_count,
            self._graph.edge_count,
            total,
        )
        return total

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> GraphRAGResult:
        """Answer *question* using the knowledge graph.

        Args:
            question: Natural language question.

        Returns:
            :class:`GraphRAGResult` with generated answer and graph context.
        """
        graph_context = self._graph.get_context(question, max_depth=self._config.max_hops)
        triples_used = graph_context.count("\n") + 1 if graph_context.strip() else 0

        prompt = (
            "Use the knowledge graph facts below to answer the question.\n\n"
            "Knowledge Graph:\n"
            f"{graph_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        answer = self._llm.complete(prompt)

        return GraphRAGResult(
            query=question,
            answer=answer.strip(),
            graph_context=graph_context,
            triples_used=triples_used,
            sources=list(self._sources),
        )

    @property
    def graph(self) -> "KnowledgeGraphStore":
        return self._graph


class _LocalFlanT5:
    """Local flan-t5-base used for GraphRAG extraction and generation."""

    def __init__(self, max_new_tokens: int = 512) -> None:
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
            logger.info("Loaded local generator for GraphRAG: google/flan-t5-base")
        result = self._pipeline(prompt)
        return result[0]["generated_text"]
