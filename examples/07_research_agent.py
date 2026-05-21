#!/usr/bin/env python3
"""
examples/07_research_agent.py — Project #7: Research Agent

Demonstrates multi-step agentic research:
  1. DuckDuckGo search for relevant URLs
  2. Scrape and chunk top pages
  3. Rank passages by TF-IDF cosine similarity
  4. Synthesise findings across sources
  5. Return a structured research report

Demo mode (no real network requests):
    python examples/07_research_agent.py --demo

Live mode (real web search + scraping):
    python examples/07_research_agent.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Research topics ───────────────────────────────────────────────────────────

TOPICS = [
    "What is retrieval-augmented generation and how does it improve LLMs?",
    "What are the main techniques for reducing hallucinations in language models?",
    "How do vector databases like Chroma and Pinecone compare?",
]

# ── Simulated search results for demo mode ────────────────────────────────────

SIMULATED_RESULTS = {
    TOPICS[0]: {
        "urls": [
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "https://arxiv.org/abs/2005.11401",
        ],
        "passages": [
            "Retrieval-augmented generation (RAG) is a technique that enhances LLM outputs "
            "by retrieving relevant documents from an external knowledge base before "
            "generating a response. This reduces hallucinations and keeps the model's "
            "knowledge current without costly fine-tuning.",
            "The original RAG paper by Lewis et al. (2020) introduced a differentiable "
            "retriever combined with a seq2seq generator. The retriever indexes documents "
            "using a dense bi-encoder (DPR) and returns top-k passages for the generator.",
            "Practical RAG pipelines typically use a vector store (Chroma, FAISS, Pinecone) "
            "to index chunks of text as embeddings. At query time, the question is embedded "
            "and the top-k most similar chunks are retrieved and concatenated into a prompt.",
        ],
        "synthesis": (
            "RAG improves LLMs by grounding responses in retrieved evidence from an external "
            "knowledge base. This addresses two key limitations: the model's static training "
            "cutoff and hallucination of unsupported facts. The retrieval step embeds the "
            "query and finds semantically similar passages, which are then injected into the "
            "prompt before generation. RAG is preferable to fine-tuning when knowledge "
            "changes frequently or when provenance tracing is required."
        ),
    },
    TOPICS[1]: {
        "urls": [
            "https://arxiv.org/abs/2303.08774",
            "https://huggingface.co/blog/hallucination",
        ],
        "passages": [
            "Hallucinations in language models can be reduced through several techniques: "
            "(1) RAG — grounding outputs in retrieved evidence; (2) Constitutional AI — "
            "training models to follow explicit principles; (3) RLHF — human preference "
            "feedback to penalise fabrications; (4) chain-of-thought prompting — making "
            "reasoning steps explicit.",
            "Self-consistency sampling generates multiple answers and selects the majority "
            "response, reducing variance and factual errors. Temperature=0 greedy decoding "
            "also reduces hallucination risk but can make outputs less creative.",
            "Fact-checking pipelines can post-process LLM outputs by running claims through "
            "a retrieval system and flagging unsupported assertions.",
        ],
        "synthesis": (
            "Main techniques for reducing hallucinations include: RAG (retrieval grounding), "
            "RLHF (human preference tuning), Constitutional AI (principle-based training), "
            "chain-of-thought prompting (explicit reasoning traces), self-consistency "
            "(majority voting across samples), and post-hoc fact-checking with retrieval. "
            "The most effective production approach combines RAG with temperature tuning "
            "and semantic similarity-based source attribution."
        ),
    },
    TOPICS[2]: {
        "urls": [
            "https://www.trychroma.com",
            "https://www.pinecone.io",
        ],
        "passages": [
            "Chroma is an open-source, embedded vector database designed for local and "
            "cloud deployments. It supports LangChain and LlamaIndex integrations and "
            "stores embeddings with metadata. Chroma is suitable for prototyping and "
            "small-to-medium scale applications.",
            "Pinecone is a fully managed vector database-as-a-service. It supports "
            "billions of vectors with low-latency similarity search. Pinecone offers "
            "namespaces for multi-tenant isolation and hybrid search combining dense "
            "and sparse vectors.",
            "FAISS (by Meta) is an in-process library for dense similarity search. It "
            "is not a database — it has no persistence, metadata storage, or server "
            "architecture. FAISS is best for research and offline processing.",
        ],
        "synthesis": (
            "Chroma is best for local development and prototyping with LangChain: open-source, "
            "no server required, easy setup. Pinecone is the production choice for "
            "large-scale or multi-tenant SaaS applications: managed, scalable, with "
            "hybrid search. FAISS suits research pipelines needing maximum speed without "
            "persistence. The choice depends on scale, operational complexity tolerance, "
            "and whether managed infrastructure is acceptable."
        ),
    },
}


class _MockLLM:
    def complete(self, prompt: str) -> str:
        p = prompt.lower()
        if "retrieval-augmented" in p or "rag" in p:
            return SIMULATED_RESULTS[TOPICS[0]]["synthesis"]
        if "hallucination" in p:
            return SIMULATED_RESULTS[TOPICS[1]]["synthesis"]
        if "chroma" in p or "pinecone" in p or "vector database" in p:
            return SIMULATED_RESULTS[TOPICS[2]]["synthesis"]
        return "Based on the retrieved passages, here is a synthesis of the research findings."


def run(live: bool) -> dict:
    from src.agents.research_agent import ResearchAgentConfig, ResearchAgent

    print("=" * 60)
    print("  Project #7 — Research Agent")
    print("=" * 60)
    print(f"  Mode: {'LIVE' if live else 'DEMO (simulated search)'}\n")

    cfg = ResearchAgentConfig(
        num_search_results=2,
        top_k_passages=3,
        chunk_size=400,
        scrape_timeout=10,
    )

    llm = _MockLLM()
    agent = ResearchAgent(config=cfg, llm=llm)

    results = []
    for topic in TOPICS:
        print(f"  Researching: {topic}")
        if live:
            result = agent.research(topic)
            report = {
                "topic": result.query,
                "synthesis": result.synthesis,
                "sources": result.sources,
                "passage_count": result.passage_count,
            }
        else:
            sim = SIMULATED_RESULTS[topic]
            report = {
                "topic": topic,
                "synthesis": sim["synthesis"],
                "sources": sim["urls"],
                "passage_count": len(sim["passages"]),
                "sample_passages": sim["passages"],
            }
            print(f"  Synthesis: {sim['synthesis'][:120]}…")
        print(f"  Sources: {report['sources']}\n")
        results.append(report)

    output = {
        "project": "07_research_agent",
        "description": "Research Agent — DuckDuckGo search + scrape + rank + synthesise",
        "mode": "live" if live else "demo",
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "07_research_agent_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Output saved → {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(live=args.live)


if __name__ == "__main__":
    main()
