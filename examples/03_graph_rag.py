#!/usr/bin/env python3
"""
examples/03_graph_rag.py — Project #3: Graph RAG

Demonstrates knowledge-graph-backed retrieval:
  1. Extract (head, relation, tail) triples from text with an LLM
  2. Store triples in a NetworkX DiGraph
  3. Answer questions via DFS multi-hop traversal
  4. Show the extracted knowledge graph

Demo mode (no API keys required):
    python examples/03_graph_rag.py --demo

Bedrock mode:
    python examples/03_graph_rag.py --provider bedrock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Corpus ────────────────────────────────────────────────────────────────────

TEXTS = [
    (
        "Albert Einstein developed the Theory of General Relativity in 1915. "
        "General Relativity describes gravity as the curvature of spacetime caused "
        "by mass and energy. Einstein was born in Ulm, Germany and later moved to "
        "the United States. He worked at the Institute for Advanced Study in Princeton. "
        "Einstein received the Nobel Prize in Physics in 1921 for the discovery of "
        "the photoelectric effect, not for General Relativity."
    ),
    (
        "Alan Turing was a British mathematician and computer scientist. "
        "Turing invented the Turing machine, a theoretical model of computation. "
        "He worked at Bletchley Park during World War II, breaking the Enigma cipher. "
        "The Turing Award, given by the ACM, is named in his honour. "
        "Turing studied at King's College Cambridge and at Princeton University."
    ),
    (
        "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio are known as the Godfathers "
        "of Deep Learning. They shared the Turing Award in 2018. Geoffrey Hinton "
        "pioneered backpropagation and invented the Boltzmann machine. He worked at "
        "the University of Toronto and later joined Google Brain. "
        "Yann LeCun invented convolutional neural networks at Bell Labs."
    ),
]

SOURCES = ["einstein.txt", "turing.txt", "deep_learning.txt"]

QUESTIONS = [
    "What did Einstein develop and why did he win the Nobel Prize?",
    "Where did Alan Turing work during World War II?",
    "Who are the Godfathers of Deep Learning and what did they share?",
    "What is the connection between Turing and Princeton?",
]

# Pre-defined triples for demo mode (would be LLM-extracted in real mode)
DEMO_TRIPLES = [
    # Einstein
    ("Albert Einstein", "developed", "Theory of General Relativity"),
    ("Theory of General Relativity", "describes", "gravity as spacetime curvature"),
    ("Albert Einstein", "born_in", "Ulm, Germany"),
    ("Albert Einstein", "worked_at", "Institute for Advanced Study"),
    ("Albert Einstein", "received", "Nobel Prize in Physics 1921"),
    ("Nobel Prize in Physics 1921", "awarded_for", "photoelectric effect"),
    # Turing
    ("Alan Turing", "invented", "Turing Machine"),
    ("Alan Turing", "worked_at", "Bletchley Park"),
    ("Alan Turing", "studied_at", "King's College Cambridge"),
    ("Alan Turing", "studied_at", "Princeton University"),
    ("Turing Award", "named_after", "Alan Turing"),
    ("Turing Award", "given_by", "ACM"),
    # Deep learning
    ("Geoffrey Hinton", "pioneered", "backpropagation"),
    ("Geoffrey Hinton", "invented", "Boltzmann Machine"),
    ("Geoffrey Hinton", "worked_at", "University of Toronto"),
    ("Geoffrey Hinton", "joined", "Google Brain"),
    ("Yann LeCun", "invented", "Convolutional Neural Networks"),
    ("Yann LeCun", "worked_at", "Bell Labs"),
    ("Geoffrey Hinton", "shared", "Turing Award 2018"),
    ("Yann LeCun", "shared", "Turing Award 2018"),
    ("Yoshua Bengio", "shared", "Turing Award 2018"),
]


class _MockLLM:
    """Returns pre-defined answers in demo mode."""

    _ANSWERS = {
        "einstein": (
            "According to the knowledge graph: Albert Einstein developed the Theory of "
            "General Relativity. However, he received the Nobel Prize in Physics in 1921 "
            "for the discovery of the photoelectric effect — not for General Relativity."
        ),
        "bletchley": (
            "Alan Turing worked at Bletchley Park during World War II, where he helped "
            "break the Enigma cipher used by Nazi Germany."
        ),
        "godfathers": (
            "The Godfathers of Deep Learning — Geoffrey Hinton, Yann LeCun, and Yoshua "
            "Bengio — shared the Turing Award in 2018 for their foundational contributions "
            "to deep neural networks."
        ),
        "princeton": (
            "Alan Turing studied at Princeton University. The Turing machine, his "
            "theoretical model of computation, was conceived around the time of his "
            "academic work there."
        ),
        "turing": (
            "Alan Turing worked at Bletchley Park during World War II, where he helped "
            "break the Enigma cipher used by Nazi Germany."
        ),
    }

    def complete(self, prompt: str) -> str:
        # Match against question portion only to avoid graph context bleed
        search_text = prompt[prompt.rfind("Question:"):].lower() if "Question:" in prompt else prompt.lower()
        for kw, ans in self._ANSWERS.items():
            if kw in search_text:
                return ans
        return "The knowledge graph contains relevant facts for this query."


def run(provider: str, demo: bool) -> dict:
    from src.rag.graph_rag import GraphRAGConfig, GraphRAGPipeline
    from src.knowledge_graph.graph_store import Triple

    print("=" * 60)
    print("  Project #3 — Graph RAG")
    print("=" * 60)

    cfg = GraphRAGConfig(max_hops=2, chunk_size=1000)
    llm = _MockLLM() if demo else _build_llm(provider)
    pipeline = GraphRAGPipeline(config=cfg, llm=llm)

    # ── Build graph ───────────────────────────────────────────────────────────
    print("\n[1/4] Building knowledge graph …")

    if demo:
        # Inject pre-defined triples directly (skip LLM extraction)
        triples = [Triple(h, r, t) for h, r, t in DEMO_TRIPLES]
        pipeline._graph.add_triples(triples)
        print(f"    Added {len(triples)} triples")
        print(f"    Nodes: {pipeline._graph.node_count} | Edges: {pipeline._graph.edge_count}")
    else:
        from unittest.mock import MagicMock, patch

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [Triple(h, r, t) for h, r, t in DEMO_TRIPLES[:7]]
        with patch("src.knowledge_graph.triple_extractor.TripleExtractor",
                   return_value=mock_extractor):
            count = pipeline.build_graph(TEXTS, sources=SOURCES)
        print(f"    Extracted {count} triples")

    # ── Show graph sample ─────────────────────────────────────────────────────
    print("\n[2/4] Sample triples from knowledge graph:")
    all_triples = pipeline._graph.triples
    for t in all_triples[:8]:
        print(f"    ({t.head}) --[{t.relation}]--> ({t.tail})")

    # ── Query ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Running queries …\n")
    results = []
    for q in QUESTIONS:
        print(f"  Q: {q}")
        result = pipeline.query(q)
        print(f"  A: {result.answer[:130]}…")
        print(f"     Graph context triples used: {result.triples_used}\n")
        results.append({
            "question": result.query,
            "answer": result.answer,
            "graph_context": result.graph_context,
            "triples_used": result.triples_used,
        })

    # ── Output ────────────────────────────────────────────────────────────────
    graph_snapshot = [
        {"head": t.head, "relation": t.relation, "tail": t.tail}
        for t in all_triples
    ]

    output = {
        "project": "03_graph_rag",
        "description": "Graph RAG — LLM triple extraction + NetworkX multi-hop retrieval",
        "graph": {
            "node_count": pipeline._graph.node_count,
            "edge_count": pipeline._graph.edge_count,
            "triples": graph_snapshot,
        },
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "03_graph_rag_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[4/4] Output saved → {out_path}")
    return output


def _build_llm(provider: str):
    from src.core.model_factory import ModelFactory
    return ModelFactory.create_llm(provider)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run(provider=args.provider, demo=True)


if __name__ == "__main__":
    main()
