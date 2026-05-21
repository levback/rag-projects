#!/usr/bin/env python3
"""
examples/05_agentic_rag.py — Project #5: Agentic RAG

Demonstrates intent-aware routing:
  - SEARCH intent → vector retrieval + generation
  - DIRECT intent → LLM answers directly without retrieval
  - Shows routing decisions for each query

Demo mode (no API keys required):
    python examples/05_agentic_rag.py --demo

Bedrock mode:
    python examples/05_agentic_rag.py --provider bedrock
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

CORPUS = [
    "Python was created by Guido van Rossum and first released in 1991. "
    "It emphasises code readability and uses significant indentation.",

    "Python 3.12 introduced significant performance improvements through the "
    "Faster CPython project, achieving a 25% speedup over Python 3.11 on standard "
    "benchmarks. It also added f-string improvements and a new type parameter syntax.",

    "The Python Package Index (PyPI) hosts over 500,000 packages as of 2024. "
    "pip is the standard package installer. Virtual environments (venv) isolate "
    "project dependencies to prevent version conflicts.",

    "Popular Python frameworks include Django and Flask for web development, "
    "FastAPI for high-performance APIs, and Celery for distributed task queues. "
    "Django follows the batteries-included philosophy with an ORM, admin panel, "
    "and authentication built-in.",

    "NumPy provides n-dimensional array operations with C-speed. Pandas builds "
    "on NumPy with DataFrame-based tabular data manipulation. Scikit-learn "
    "provides machine learning algorithms with a consistent API.",
]

# Mix of SEARCH (factual/retrieval) and DIRECT (conversational/no-context) queries
QUERIES = [
    ("What performance improvement did Python 3.12 achieve?",           "search"),
    ("How many packages are on PyPI?",                                   "search"),
    ("What is the difference between Django and FastAPI?",              "search"),
    ("Thank you, that was very helpful!",                               "direct"),
    ("Can you summarise what we discussed so far?",                     "direct"),
    ("What NumPy operations does pandas build on?",                     "search"),
    ("Tell me a fun fact",                                              "direct"),
]


class _MockLLM:
    _SEARCH_ANSWERS = {
        "python 3.12": (
            "Python 3.12 achieved a 25% speedup over Python 3.11 on standard benchmarks, "
            "delivered through the Faster CPython project. It also introduced f-string "
            "improvements and a new type parameter syntax for generics."
        ),
        "pypi": (
            "The Python Package Index (PyPI) hosts over 500,000 packages as of 2024. "
            "pip is the standard package installer, and venv provides virtual environment "
            "isolation to prevent dependency conflicts."
        ),
        "django": (
            "Django follows a batteries-included philosophy, bundling an ORM, admin panel, "
            "and authentication. FastAPI is designed for high-performance async APIs, "
            "with automatic OpenAPI documentation and type-hint-based validation."
        ),
        "numpy": (
            "Pandas builds on NumPy's n-dimensional array operations, extending them with "
            "a DataFrame abstraction for tabular data manipulation. NumPy provides the "
            "underlying C-speed array computations that pandas relies on."
        ),
    }

    _DIRECT_ANSWERS = {
        "thank": "You're welcome! Feel free to ask if you have more questions.",
        "summarise": (
            "We've discussed Python's history, its 3.12 performance improvements, "
            "the PyPI ecosystem, popular web frameworks, and scientific computing libraries."
        ),
        "fun fact": (
            "Python is named after Monty Python's Flying Circus, not the snake. "
            "Guido van Rossum was reading scripts for the show while writing Python."
        ),
    }

    def complete(self, prompt: str) -> str:
        # Match only against the question portion to avoid context bleed
        search_text = prompt[prompt.rfind("Question:"):].lower() if "Question:" in prompt else prompt.lower()
        for kw, ans in self._SEARCH_ANSWERS.items():
            if kw in search_text:
                return ans
        for kw, ans in self._DIRECT_ANSWERS.items():
            if kw in search_text:
                return ans
        return "I can help with that. Based on context, here is a relevant answer."


def run(provider: str, demo: bool) -> dict:
    from src.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline, IntentType
    from unittest.mock import MagicMock, patch

    print("=" * 60)
    print("  Project #5 — Agentic RAG (Intent Routing)")
    print("=" * 60)

    cfg = AgenticRAGConfig(
        collection_name="agentic_demo",
        persist_dir="/tmp/agentic_demo",
        top_k=2,
    )

    llm = _MockLLM() if demo else _build_llm(provider)
    pipeline = AgenticRAGPipeline(config=cfg, llm=llm)

    # ── Index corpus ──────────────────────────────────────────────────────────
    print("\n[1/3] Indexing corpus …")
    mock_store = MagicMock()
    mock_store.add_texts.return_value = None

    def _search_side_effect(q, k=2):
        # Return relevant mocked docs based on query keywords
        q_lower = q.lower()
        relevant_content = ""
        for chunk in CORPUS:
            if any(w in chunk.lower() for w in q_lower.split()[:3]):
                relevant_content = chunk
                break
        return [MagicMock(page_content=relevant_content or CORPUS[0],
                          metadata={"source": "python_docs.txt"})]

    mock_store.similarity_search.side_effect = _search_side_effect

    with patch("langchain_community.vectorstores.Chroma", return_value=mock_store), \
         patch("langchain_huggingface.HuggingFaceEmbeddings"):
        pipeline.index(CORPUS, sources=["python_docs.txt"] * len(CORPUS))
        print(f"    Indexed {len(CORPUS)} chunks")

        # ── Run queries ───────────────────────────────────────────────────────
        print("\n[2/3] Running queries with intent routing …\n")
        results = []
        for query, expected_intent in QUERIES:
            result = pipeline.query(query)
            intent_label = result.intent.value if hasattr(result.intent, "value") else str(result.intent)
            match_symbol = "✓" if intent_label.lower() == expected_intent else "≈"
            print(f"  Q: {query}")
            print(f"     Intent: {intent_label} {match_symbol} (expected: {expected_intent})")
            print(f"     A: {result.answer[:110]}…\n")
            results.append({
                "query": result.query,
                "intent": intent_label,
                "expected_intent": expected_intent,
                "answer": result.answer,
                "retrieved_chunks": len(result.retrieved_chunks),
            })

    # ── Output ────────────────────────────────────────────────────────────────
    search_count = sum(1 for r in results if r["intent"] == "search")
    direct_count = sum(1 for r in results if r["intent"] == "direct")

    output = {
        "project": "05_agentic_rag",
        "description": "Agentic RAG — keyword-based intent routing (SEARCH vs DIRECT)",
        "routing_summary": {
            "total_queries": len(results),
            "routed_to_search": search_count,
            "routed_to_direct": direct_count,
        },
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "05_agentic_rag_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[3/3] Output saved → {out_path}")
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
