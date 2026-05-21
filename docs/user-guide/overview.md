# Overview

> **RAG-Projects** is a self-contained Python library for building and
> experimenting with ten distinct Retrieval-Augmented Generation (RAG) patterns,
> from a single-document basic pipeline to multimodal knowledge graphs.

Cross-references: [Get Started](get-started.md) · [Examples](../examples.md) · [API Reference](../reference/index.md)

---

## What is this library?

RAG-Projects bundles ten end-to-end RAG implementations under a single,
consistent Python package. Every pipeline shares the same `BaseLLM` abstraction,
making it trivial to swap between OpenAI, Anthropic Claude, Amazon Bedrock, and
local HuggingFace models without changing pipeline code.

The library is designed for:

- **Learning** — each example is self-contained and heavily commented
- **Prototyping** — all examples run in demo mode with no API keys
- **Production** — IBM Production RAG and LangChain Agent include retry logic,
  latency tracking, and audit-ready prompt delimiters

---

## The 10 RAG Patterns

| # | Pattern | Module | Key capability |
|---|---------|--------|---------------|
| 1 | **Basic RAG** | [`src/rag/basic_rag.py`](../../src/rag/basic_rag.py) | FAISS + chunk + generate |
| 2 | **IBM Production RAG** | [`src/rag/ibm_rag.py`](../../src/rag/ibm_rag.py) | Retry, latency, enterprise embeddings |
| 3 | **Graph RAG** | [`src/rag/graph_rag.py`](../../src/rag/graph_rag.py) | LLM triple extraction + NetworkX DFS |
| 4 | **Multi-Document RAG** | [`src/rag/multi_doc_rag.py`](../../src/rag/multi_doc_rag.py) | Cross-document retrieval with provenance |
| 5 | **Agentic RAG** | [`src/rag/agentic_rag.py`](../../src/rag/agentic_rag.py) | Intent routing (SEARCH vs DIRECT) |
| 6 | **Real-time RAG** | [`src/rag/realtime_rag.py`](../../src/rag/realtime_rag.py) | Live web search + scrape + rank |
| 7 | **Research Agent** | [`src/agents/research_agent.py`](../../src/agents/research_agent.py) | Multi-step web research + synthesis |
| 8 | **Multimodal RAG** | [`src/rag/multimodal_rag.py`](../../src/rag/multimodal_rag.py) | PDF images + VLM captioning + FAISS |
| 9 | **LangChain RAG Agent** | [`src/agents/langchain_rag_agent.py`](../../src/agents/langchain_rag_agent.py) | LCEL chain + ReAct agent |
| 10 | **Document Analysis** | [`src/core/document_analysis_pipeline.py`](../../src/core/document_analysis_pipeline.py) | PDF → summary + QA + NER |

---

## Repository layout

```
rag-projects/
├── src/                        # All library source code
│   ├── core/                   # LLM abstractions, pipeline orchestrator
│   ├── rag/                    # 7 RAG pipeline implementations
│   ├── agents/                 # 2 agent implementations + base
│   ├── knowledge_graph/        # Triple extraction + graph store
│   ├── loaders/                # PDF, text, URL, directory loaders
│   ├── processing/             # Chunking, preprocessing, tokenizer
│   ├── inference/              # Summariser, QA engine, question generator
│   └── prompts/                # Prompt templates + LangChain chains
├── examples/                   # 10 runnable example scripts
│   └── output/                 # Pre-computed JSON outputs
├── tests/
│   ├── unit/                   # Unit tests (95%+ coverage)
│   └── integration/            # Integration tests
├── docs/                       # This documentation
│   ├── user-guide/             # ← You are here
│   └── reference/              # API reference
├── scripts/                    # Shell helpers (setup, run, test, security)
└── config/                     # YAML configuration files
```

---

## Design principles

**Single `BaseLLM` abstraction**  
All pipelines call `llm.complete(prompt)` — never a provider-specific SDK
directly. Swapping providers is a one-line config change.

**Dataclass configs**  
Every pipeline has a typed `*Config` dataclass. Defaults are sensible for
local development; fields map 1:1 to documented parameters.

**No mutation**  
Pipeline results are immutable dataclass instances. Input documents are never
modified in-place.

**Security-first prompts**  
All LLM prompts that include retrieved content use `=== BEGIN SOURCES ===` /
`=== END SOURCES ===` delimiters and a "treat as data" instruction to mitigate
prompt injection (OWASP A08).

**95%+ test coverage**  
398 tests across unit and integration suites, verified with `pytest-cov`.
See [Developer Notes](developer-notes.md) for the testing strategy.
