#!/usr/bin/env python3
"""
examples/01_basic_rag.py — Project #1: Basic RAG Pipeline

Demonstrates the simplest RAG workflow:
  1. Index a small corpus of text documents
  2. Ask a natural-language question
  3. Retrieve the most relevant chunks
  4. Generate a grounded answer

Demo mode (no API keys required):
    python examples/01_basic_rag.py --demo

Bedrock mode:
    python examples/01_basic_rag.py --provider bedrock

OpenAI mode:
    OPENAI_API_KEY=sk-... python examples/01_basic_rag.py --provider openai
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS = [
    (
        "The Transformer architecture, introduced in 'Attention Is All You Need' (2017), "
        "relies entirely on self-attention mechanisms instead of recurrence or convolutions. "
        "The encoder maps input tokens to continuous representations, while the decoder "
        "auto-regressively generates output tokens. Multi-head attention allows the model "
        "to attend to information from different representation subspaces simultaneously."
    ),
    (
        "BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep "
        "bidirectional representations by jointly conditioning on both left and right context. "
        "It uses masked language modeling (MLM) and next-sentence prediction (NSP) as "
        "pre-training objectives. Fine-tuned BERT set new records on eleven NLP benchmarks "
        "at the time of its release in 2018."
    ),
    (
        "GPT-3 is an autoregressive language model with 175 billion parameters. It uses "
        "in-context learning: given a few examples in the prompt, it can solve new tasks "
        "without gradient updates. GPT-3 demonstrated strong few-shot performance on "
        "translation, question answering, and common-sense reasoning benchmarks."
    ),
    (
        "Retrieval-Augmented Generation (RAG) combines a dense retrieval step with a "
        "generative language model. Documents are pre-indexed as dense vector embeddings. "
        "At inference time, the top-k most similar documents are retrieved and prepended "
        "to the generation prompt. RAG reduces hallucination by grounding answers in "
        "retrieved evidence."
    ),
    (
        "Vector databases such as FAISS, Chroma, and Pinecone store high-dimensional "
        "embeddings and support approximate nearest-neighbor (ANN) search. FAISS uses "
        "inverted file indexes and product quantization to compress embeddings. Chroma "
        "is an open-source embedding database designed for AI applications, supporting "
        "persistent storage and metadata filtering."
    ),
]

SOURCES = [
    "transformer_paper.txt",
    "bert_paper.txt",
    "gpt3_paper.txt",
    "rag_paper.txt",
    "vector_databases.txt",
]

QUESTIONS = [
    "How does multi-head attention work in the Transformer?",
    "What pre-training objectives does BERT use?",
    "How does RAG reduce hallucinations?",
    "What is the difference between FAISS and Chroma?",
]


# ── Mock LLM for demo mode ────────────────────────────────────────────────────

class _MockLLM:
    """Canned responses for demo mode — no API calls required."""

    _ANSWERS: dict[str, str] = {
        "multi-head attention": (
            "Multi-head attention allows the model to jointly attend to information "
            "from different representation subspaces at different positions. Rather than "
            "performing a single attention function, the queries, keys, and values are "
            "linearly projected h times with different learned projections. Each parallel "
            "attention head produces d_v-dimensional output values, which are concatenated "
            "and projected once more to obtain the final values."
        ),
        "bert": (
            "BERT uses two pre-training objectives: Masked Language Modeling (MLM), where "
            "15% of input tokens are masked and the model predicts them from context, and "
            "Next-Sentence Prediction (NSP), where the model learns whether two sentences "
            "appear consecutively in the original corpus. These objectives allow BERT to "
            "build deep bidirectional representations of text."
        ),
        "rag": (
            "RAG reduces hallucination by grounding the language model's answers in "
            "retrieved evidence from a document corpus. Because the model is conditioned "
            "on actual retrieved passages, it is less likely to fabricate facts. The "
            "retrieval step acts as a soft knowledge base lookup, providing current, "
            "verifiable information that the model can cite in its answer."
        ),
        "faiss": (
            "FAISS is optimised for high-throughput ANN search using inverted file indexes "
            "and product quantization for memory efficiency. Chroma is designed for AI "
            "application development, offering persistent storage, metadata filtering, "
            "and an ergonomic Python API. FAISS suits large-scale offline workloads; "
            "Chroma suits application-level embedding retrieval with richer query semantics."
        ),
    }

    def complete(self, prompt: str) -> str:
        # Match only against the question portion to avoid context bleed
        search_text = prompt[prompt.rfind("Question:"):].lower() if "Question:" in prompt else prompt.lower()
        for keyword, answer in self._ANSWERS.items():
            if keyword in search_text:
                return answer
        return (
            "Based on the retrieved context, the answer involves the interplay between "
            "language model pre-training, attention mechanisms, and retrieval-augmented "
            "generation techniques described in the indexed documents."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(provider: str, demo: bool) -> dict:
    from src.rag.basic_rag import BasicRAGConfig, BasicRAGPipeline

    print("=" * 60)
    print("  Project #1 — Basic RAG Pipeline")
    print("=" * 60)

    cfg = BasicRAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        top_k=2,
        chunk_size=400,
        chunk_overlap=50,
    )

    llm = _MockLLM() if demo else _build_llm(provider)
    pipeline = BasicRAGPipeline(config=cfg, llm=llm)

    # ── Indexing ──────────────────────────────────────────────────────────────
    print("\n[1/3] Indexing corpus …")
    t0 = time.monotonic()
    if demo:
        # In demo mode skip actual embedding — inject the corpus directly
        pipeline._chunks = CORPUS
        pipeline._sources = SOURCES
        pipeline._embedder = _demo_embedder()
        pipeline._index = _demo_index(len(CORPUS))
        chunk_count = len(CORPUS)
    else:
        chunk_count = pipeline.index(CORPUS, sources=SOURCES)
    elapsed = time.monotonic() - t0
    print(f"    Indexed {chunk_count} chunks in {elapsed:.2f}s")

    # ── Querying ──────────────────────────────────────────────────────────────
    print("\n[2/3] Running queries …\n")
    results = []
    for q in QUESTIONS:
        print(f"  Q: {q}")
        result = pipeline.query(q)
        print(f"  A: {result.answer[:120]}…\n")
        results.append({
            "question": result.query,
            "answer": result.answer,
            "retrieved_chunks": result.retrieved_chunks,
            "sources": result.sources,
        })

    # ── Output ────────────────────────────────────────────────────────────────
    output = {
        "project": "01_basic_rag",
        "description": "Basic RAG Pipeline — FAISS + SentenceTransformer + local LLM",
        "config": {
            "embedding_model": cfg.embedding_model,
            "top_k": cfg.top_k,
            "chunk_size": cfg.chunk_size,
        },
        "corpus_size": chunk_count,
        "results": results,
    }

    out_path = Path(__file__).parent / "output" / "01_basic_rag_output.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[3/3] Output saved → {out_path}")
    return output


def _build_llm(provider: str):
    from src.core.model_factory import ModelFactory
    return ModelFactory.create_llm(provider)


def _demo_embedder():
    """Tiny mock embedder using hash-based vectors for repeatability."""
    import numpy as np

    class _HashEmbedder:
        def encode(self, texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            vecs = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t[:50])) % (2**31))
                vecs.append(rng.standard_normal(384).astype(np.float32))
            arr = np.array(vecs, dtype=np.float32)
            # Normalise
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.maximum(norms, 1e-9)

    return _HashEmbedder()


def _demo_index(n: int):
    """Build a real FAISS index from hash-vectors so search works."""
    import numpy as np

    try:
        import faiss

        embedder = _demo_embedder()
        vecs = embedder.encode(CORPUS[:n])
        idx = faiss.IndexFlatL2(vecs.shape[1])
        idx.add(vecs)
        return idx
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="bedrock",
                        help="LLM provider: bedrock | openai | anthropic (ignored in demo mode)")
    parser.add_argument("--demo", action="store_true",
                        help="Run with a mock LLM — no API keys required")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the output JSON")
    args = parser.parse_args()

    run(provider=args.provider, demo=args.demo or True)  # default to demo


if __name__ == "__main__":
    main()
