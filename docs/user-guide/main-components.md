# Main Components

A module-by-module tour of every major class and its relationship to other components.

Cross-references: [Learn Basics](learn-basics.md) · [Developer Notes](developer-notes.md) · [API Reference](../reference/index.md)

---

## `src/core/` — LLM Layer

The core layer provides the LLM abstraction and the original document analysis pipeline.

### `base_llm.py` — [`BaseLLM`](../reference/core.md#basellm)

Abstract base class for all LLM backends.

```
Message          dataclass  Role + content pair
LLMResponse      dataclass  Generated text + usage + model name
LLMConfig        dataclass  model, temperature, max_tokens, timeout
BaseLLM          ABC        complete() / stream() / chat()
```

**Inheritors:** `GPTClient` · `ClaudeClient` · `BedrockClient` · `LocalLLM`

### `model_factory.py` — [`create_llm()`](../reference/core.md#create_llm)

Factory function. Reads credentials from environment, builds the correct `BaseLLM`.

```python
from src.core.model_factory import create_llm
llm = create_llm("bedrock")   # ModelProvider enum or string
```

**Used by:** all 10 examples · all pipelines in `--provider` mode

### `gpt_client.py`, `claude_client.py`, `bedrock_client.py`, `local_llm.py`

Concrete `BaseLLM` implementations. See [API Reference → Core](../reference/core.md).

### `document_analysis_pipeline.py` — [`DocumentAnalysisPipeline`](../reference/core.md#documentanalysispipeline)

The original all-in-one pipeline: PDF → summary + QG + QA + NER.

```
AnalysisConfig     dataclass  provider, llm_provider, batch_size, …
PassageAnalysis    dataclass  Per-passage QA result
AnalysisResult     dataclass  run() output — summary, qa_pairs, metadata
DocumentAnalysisPipeline      run() / save_results() / print_results()
```

**Example:** [`examples/10_document_analysis.py`](../../examples/10_document_analysis.py)

---

## `src/rag/` — RAG Pipelines

Seven distinct RAG implementations, all following the same `*Config` + pipeline pattern.

### `basic_rag.py` — [`BasicRAGPipeline`](../reference/rag.md#basicragpipeline)

Canonical RAG. FAISS index, `all-MiniLM-L6-v2` embeddings, top-k retrieval.

```python
from src.rag.basic_rag import BasicRAGConfig, BasicRAGPipeline
pipeline = BasicRAGPipeline(config=BasicRAGConfig(), llm=llm)
pipeline.index(texts, sources=sources)
result = pipeline.query("What is RAG?")
# result.answer, result.retrieved_chunks, result.sources
```

### `ibm_rag.py` — [`IBMProductionRAG`](../reference/rag.md#ibmproductionrag)

Adds enterprise features on top of basic RAG:

- `max_retries` + `retry_delay` exponential back-off
- `latency_ms` per query
- `stats` property: indexed sources, total chunks, vector store name
- IBM Granite embedding support

```python
from src.rag.ibm_rag import IBMRAGConfig, IBMProductionRAG
cfg = IBMRAGConfig(max_retries=3, retry_delay=0.5)
rag = IBMProductionRAG(config=cfg, llm=llm)
rag.load_text(text, source="ibm_docs.txt")
result = rag.query("What is watsonx?")
# result.latency_ms, result.retries_used
```

### `graph_rag.py` — [`GraphRAGPipeline`](../reference/rag.md#graphragpipeline)

Uses `KnowledgeGraphStore` for multi-hop retrieval.

```python
from src.rag.graph_rag import GraphRAGConfig, GraphRAGPipeline
pipeline = GraphRAGPipeline(config=GraphRAGConfig(max_hops=2), llm=llm)
pipeline.build_graph(texts, sources=sources)   # LLM extracts triples
result = pipeline.query("Where did Turing study?")
# result.graph_context, result.triples_used
```

**Dependencies:** [`KnowledgeGraphStore`](../reference/knowledge_graph.md#knowledgegraphstore) · [`TripleExtractor`](../reference/knowledge_graph.md#tripleextractor)

### `multi_doc_rag.py` — [`MultiDocumentRAG`](../reference/rag.md#multidocumentrag)

Indexes and queries across multiple heterogeneous documents.

```python
from src.rag.multi_doc_rag import MultiDocConfig, MultiDocumentRAG
rag = MultiDocumentRAG(config=MultiDocConfig(), llm=llm)
rag.load_directory(Path("./docs"))
result = rag.query("Which files mention carbon prices?")
# result.source_documents  ← per-chunk provenance
```

### `agentic_rag.py` — [`AgenticRAGPipeline`](../reference/rag.md#agenticragpipeline)

Intent-routing RAG.

```python
from src.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline, IntentType
pipeline = AgenticRAGPipeline(config=AgenticRAGConfig(), llm=llm)
pipeline.index(texts)
result = pipeline.query("Thank you!")
# result.intent == IntentType.DIRECT  → no retrieval
result2 = pipeline.query("What is FAISS?")
# result2.intent == IntentType.SEARCH → retrieved_chunks populated
```

### `realtime_rag.py` — [`RealtimeRAGAssistant`](../reference/rag.md#realtimeragassistant)

Web-search-grounded. Calls DuckDuckGo → scrape → chunk → cosine rank → generate.

```python
from src.rag.realtime_rag import RealtimeRAGConfig, RealtimeRAGAssistant
assistant = RealtimeRAGAssistant(config=RealtimeRAGConfig(), llm=llm)
result = assistant.query("Latest LLM releases 2025")
# result.search_urls, result.retrieved_passages
```

**Security:** SSRF-blocked URLs, prompt injection delimiters, query truncated to 1000 chars.

### `multimodal_rag.py` — [`MultimodalRAGPipeline`](../reference/rag.md#multimodalragpipeline)

Indexes PDF text + image captions in FAISS.

```python
from src.rag.multimodal_rag import MultimodalRAGConfig, MultimodalRAGPipeline
cfg = MultimodalRAGConfig(vision_model="amazon.nova-pro-v1:0")
pipeline = MultimodalRAGPipeline(config=cfg, llm=llm)
pipeline.load_pdf("paper.pdf")   # extracts text + captions images
result = pipeline.query("What does Figure 1 show?")
```

---

## `src/agents/` — Agent Layer

Agents add multi-step reasoning on top of RAG.

### `base_agent.py` — [`BaseAgent`](../reference/agents.md#baseagent)

Abstract base. Defines `run(query) -> AgentResult` and `_log_step()`.

### `research_agent.py` — [`ResearchAgent`](../reference/agents.md#researchagent)

```
ResearchAgentConfig   dataclass  num_search_results, top_k_passages, chunk_size
ResearchResult        dataclass  synthesis, sources, passage_count
ResearchAgent                    run() / _web_search() / _scrape() / _rank() / _synthesize()
```

**Used by:** [`examples/07_research_agent.py`](../../examples/07_research_agent.py)

### `langchain_rag_agent.py` — [`LangChainRAGAgent`](../reference/agents.md#langchainragagent)

```
LangChainRAGConfig    dataclass  llm_provider, mode ("chain"|"agent"), top_k, …
LangChainRAGResult    dataclass  answer, sources, mode_used, retrieved_docs, steps
LangChainRAGAgent               load_text() / load_pdf() / load_url() / run()
```

**Backends supported:** `openai`, `anthropic`, `bedrock`, `huggingface`

---

## `src/knowledge_graph/` — Knowledge Graph

### `graph_store.py` — [`KnowledgeGraphStore`](../reference/knowledge_graph.md#knowledgegraphstore)

NetworkX-backed triple store with DFS traversal.

```
Triple                dataclass  head, relation, tail
GraphSearchResult     dataclass  triples, paths, context_text
KnowledgeGraphStore              add_triple() / search() / get_context()
                                 node_count / edge_count / triples (props)
                                 save() / load()
```

### `triple_extractor.py` — [`TripleExtractor`](../reference/knowledge_graph.md#tripleextractor)

LLM-based `(head, relation, tail)` extraction from free text.

```python
extractor = TripleExtractor(llm=llm)
triples = extractor.extract("Einstein developed General Relativity in 1915.")
# [Triple("Albert Einstein", "developed", "Theory of General Relativity")]
```

---

## `src/loaders/` — Document Loaders

### `document_loader.py` — [`DocumentLoader`](../reference/loaders.md#documentloader)

Unified entry point for all document types.

```
LoadedDocument   dataclass   text, source, metadata, content_hash
DocumentLoader               load() / load_pdf() / load_text() / load_url() / load_directory()
```

**Security:** `load_pdf()` and `load_text()` call `Path.resolve()` before I/O.

### `web_scraper.py` — [`WebScraper`](../reference/loaders.md#webscraper)

```
ScrapedPage      dataclass   url, text, title, status_code
WebScraper                   fetch() / fetch_page() / fetch_many()
                             _validate_url()  ← SSRF protection
```

---

## `src/processing/` — Text Processing

| File | Key classes | Purpose |
|------|-------------|---------|
| [`chunking.py`](../../src/processing/chunking.py) | `TextChunker`, `SentenceChunker` | Split text into overlapping chunks |
| [`pdf_extractor.py`](../../src/processing/pdf_extractor.py) | `PDFExtractor` | pdfplumber extraction with caching |
| [`preprocessing.py`](../../src/processing/preprocessing.py) | `TextPreprocessor` | Normalise, strip HTML, remove URLs |
| [`tokenizer.py`](../../src/processing/tokenizer.py) | `Tokenizer` | tiktoken-based token counting |

See [API Reference → Processing](../reference/processing.md).

---

## `src/inference/` — Inference Layer

| File | Key classes | Purpose |
|------|-------------|---------|
| [`inference_engine.py`](../../src/inference/inference_engine.py) | `InferenceEngine` | Retrieval + generation orchestrator |
| [`summarizer.py`](../../src/inference/summarizer.py) | `Summarizer` | T5/LLM abstractive summary |
| [`qa_engine.py`](../../src/inference/qa_engine.py) | `QAEngine` | RoBERTa/LLM extractive + abstractive QA |
| [`question_generator.py`](../../src/inference/question_generator.py) | `QuestionGenerator` | T5-based question generation |
| [`response_parser.py`](../../src/inference/response_parser.py) | `ResponseParser` | JSON extraction, markdown strip, cleaning |

See [API Reference → Inference](../reference/inference.md).

---

## `src/prompts/` — Prompt Layer

| File | Contents | Purpose |
|------|---------|---------|
| [`templates.py`](../../src/prompts/templates.py) | Prompt string constants | System/user prompt templates |
| [`chain.py`](../../src/prompts/chain.py) | LangChain `PromptTemplate` helpers | LCEL-compatible prompt builders |

---

## Component interaction diagram

```
User code
    │
    ├─► create_llm("bedrock") ──────────────────► BedrockClient
    │
    ├─► BasicRAGPipeline(config, llm)
    │       │
    │       ├─ index(texts)
    │       │       └─ TextChunker.split()
    │       │       └─ HuggingFaceEmbeddings.embed()
    │       │       └─ Chroma / FAISS.add_texts()
    │       │
    │       └─ query(question)
    │               └─ VectorStore.similarity_search()
    │               └─ BaseLLM.complete(prompt)
    │               └─ RAGResult(answer, chunks, sources)
    │
    ├─► GraphRAGPipeline(config, llm)
    │       └─ build_graph(texts)
    │               └─ TripleExtractor.extract()
    │               └─ KnowledgeGraphStore.add_triples()
    │
    └─► ResearchAgent(config, llm)
            └─ run(query)
                    └─ _web_search() → WebScraper.fetch_many()
                    └─ _rank() → TF-IDF cosine
                    └─ _synthesize() → BaseLLM.complete()
```
