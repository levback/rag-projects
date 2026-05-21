# Architecture

This document describes the internal structure, component responsibilities, data flow, and key design decisions of the Document Analysis system.

---

## System Overview

The system is organised into five packages, each with a single clear responsibility. All heavy dependencies (model weights, API clients, vector stores) are **lazy-loaded** — they are imported and instantiated only on first use.

```
┌─────────────────────────────────────────────────────────────┐
│                     External Inputs                         │
│    PDF files · API keys · AWS credentials · config YAML     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    src/core/                                 │
│                                                             │
│  DocumentAnalysisPipeline ◄── AnalysisConfig                │
│         │                                                   │
│         │ uses                                              │
│         ▼                                                   │
│  ModelFactory ──► BedrockClient │ GPTClient │ ClaudeClient  │
│                   LocalLLM      (all extend BaseLLM)        │
└──────┬──────────────────────────────────────────────────────┘
       │ orchestrates
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   src/processing/                            │
│                                                             │
│  PDFExtractor → TextPreprocessor → SentenceChunker          │
│  (pdfplumber)   (normalise/clean)  (NLTK word-limit)        │
└──────┬───────────────────────────────────────────────────────┘
       │ passages[]
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   src/inference/                             │
│                                                             │
│  Summarizer → QuestionGenerator → QAEngine                  │
│  (t5-small    (valhalla/t5-qg-hl   (roberta-squad2          │
│   or LLM)      or LLM)              or LLM)                 │
└──────┬───────────────────────────────────────────────────────┘
       │ AnalysisResult
       ▼
┌──────────────────────────────────────────────────────────────┐
│              Output  (JSON + TXT + LOG)                      │
│              examples/output/                                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  src/rag/  (standalone, used independently of the pipeline)  │
│                                                             │
│  Embedder → VectorStore (Chroma / FAISS) → Retriever        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  src/prompts/  (shared utility)                              │
│                                                             │
│  PromptTemplate → PromptChain                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Package Descriptions

### `src/core/` — Orchestration & LLM Clients

| Module | Responsibility |
|---|---|
| `base_llm.py` | Abstract base class; defines `complete()`, `stream()`, `acomplete()`, `astream()`, `chat()` |
| `gpt_client.py` | OpenAI Chat Completions API (sync + async + streaming) |
| `claude_client.py` | Anthropic Messages API (sync + async + streaming) |
| `bedrock_client.py` | AWS Bedrock Converse + ConverseStream APIs |
| `local_llm.py` | HuggingFace `pipeline("text-generation")` for local models |
| `model_factory.py` | `create_llm(provider, model, **kwargs)` — lazy-imports the right client |
| `document_analysis_pipeline.py` | `DocumentAnalysisPipeline` — wires all components and runs the 7-step pipeline |

#### LLM Client Hierarchy

```
BaseLLM  (ABC)
  ├── GPTClient        openai.ChatCompletion
  ├── ClaudeClient     anthropic.messages
  ├── BedrockClient    boto3 bedrock-runtime.converse
  └── LocalLLM         transformers.pipeline("text-generation")
```

All clients expose the same interface (`complete`, `stream`, `acomplete`, `astream`, `chat`). The `Summarizer`, `QuestionGenerator`, and `QAEngine` accept any `BaseLLM` instance — they have no knowledge of which backend is used.

#### ModelFactory Lazy-Loading

`_PROVIDER_MAP` stores `(module_path, class_name)` tuples. The actual import only happens when `create_llm()` is called, keeping startup time fast and avoiding `ImportError` for providers whose packages aren't installed.

```python
_PROVIDER_MAP = {
    ModelProvider.OPENAI:     ("src.core.gpt_client",     "GPTClient"),
    ModelProvider.ANTHROPIC:  ("src.core.claude_client",  "ClaudeClient"),
    ModelProvider.LOCAL:      ("src.core.local_llm",      "LocalLLM"),
    ModelProvider.BEDROCK:    ("src.core.bedrock_client", "BedrockClient"),
}
```

---

### `src/processing/` — Document Processing

| Module | Responsibility |
|---|---|
| `pdf_extractor.py` | Extracts plain text from PDFs using `pdfplumber`; caches result to disk by SHA-256 hash |
| `preprocessing.py` | Cleans raw text: strips HTML, normalises whitespace, removes control characters |
| `chunking.py` | `TextChunker` (overlap-based, for embedding) + `SentenceChunker` (NLTK word-limit, for inference) |
| `tokenizer.py` | Thin wrapper for token counting (used by Summarizer to detect oversized inputs) |

#### Caching Strategy

`PDFExtractor` computes a `SHA-256 + file-size` hash of the PDF's binary content and writes the extracted text to `data/cache/<hash>.txt`. Subsequent runs with the same PDF skip pdfplumber entirely.

#### Two Chunking Modes

| Class | Use case | Split strategy |
|---|---|---|
| `TextChunker` | Embedding / RAG | Sentence-boundary-aware, character overlap |
| `SentenceChunker` | Pipeline passages | NLTK `sent_tokenize`, hard word limit |

---

### `src/inference/` — Generative Components

Each component supports two backends selected by the `provider` field (`"huggingface"` or `"llm"`). When `provider="llm"` a `BaseLLM` instance must be injected.

| Module | HuggingFace model | LLM prompt template |
|---|---|---|
| `summarizer.py` | `t5-small` | `DOCUMENT_SUMMARY` |
| `question_generator.py` | `valhalla/t5-base-qg-hl` | `QUESTION_GENERATION` |
| `qa_engine.py` | `deepset/roberta-base-squad2` | `DOCUMENT_QA` |

#### Summariser Long-Document Handling

When a document is too long for a single T5 pass, `Summarizer` splits it into ≤1 000-character chunks, summarises each chunk individually, concatenates the chunk summaries, then runs a final summarisation pass over the combined text.

For LLM mode the same idea applies using configurable token budgets (`llm_chunk_tokens`, default 3 000).

---

### `src/rag/` — Retrieval-Augmented Generation Layer

| Module | Responsibility |
|---|---|
| `embedder.py` | Dense vector embeddings via OpenAI or `sentence-transformers` (local) |
| `vector_store.py` | Thin wrapper over ChromaDB or FAISS; exposes `upsert`, `search`, `delete`, `count` |
| `retriever.py` | Embeds a query and calls `vector_store.search`; optional similarity threshold |
| `indexer.py` | Batch-indexes text chunks into the vector store |

The RAG layer is **independent of the document analysis pipeline** and can be used on its own for semantic search over any text corpus.

---

### `src/prompts/` — Prompt Management

| Module | Responsibility |
|---|---|
| `templates.py` | `PromptTemplate` dataclass with `$variable` substitution + `partial()` pre-fill; built-in templates: `RAG_QA`, `SUMMARIZATION`, `QUESTION_GENERATION`, `DOCUMENT_QA`, `DOCUMENT_SUMMARY` |
| `chain.py` | `PromptChain` — multi-step LLM workflow; `ChainStep` → `ChainContext`; async `arun()` |

---

## Data Flow

### Full Pipeline Run

```
pipeline.run("paper.pdf")
│
├─ PDFExtractor.extract()
│    └─ pdfplumber.open()  →  raw_text  →  cache write
│
├─ TextPreprocessor.process(raw_text)
│    └─  strip HTML, normalise whitespace  →  clean_text
│
├─ SentenceChunker.split(clean_text)
│    └─  NLTK sent_tokenize + word limit  →  passages[]
│
├─ Summarizer.summarize_passages(passages)
│    ├─ [HF]  t5-small pipeline(chunk)  ×N  →  summaries[]  →  re-summarise
│    └─ [LLM] DOCUMENT_SUMMARY.format(text=...)  →  llm.chat()
│
├─ for passage in passages:
│    ├─ QuestionGenerator.generate(passage)
│    │    ├─ [HF]  valhalla/t5-base-qg-hl  →  "q1 <sep> q2 <sep> …"  →  split
│    │    └─ [LLM] QUESTION_GENERATION.format(passage=...)  →  llm.chat()  →  parse
│    │
│    └─ QAEngine.answer_batch(questions, passage)
│         ├─ [HF]  roberta-base-squad2  →  {"answer", "score"}
│         └─ [LLM] DOCUMENT_QA.format(context=..., question=...)  →  llm.chat()
│
└─ AnalysisResult(source, summary, passages, all_qa_pairs)
     └─ save_results()  →  examples/output/<stem>_analysis.json
```

---

## Key Design Decisions

### 1. Provider-Agnostic Inference Components

`Summarizer`, `QuestionGenerator`, and `QAEngine` do not import any LLM SDK directly. They accept an optional `llm: BaseLLM` parameter and call `llm.chat()`. This means:

- Tests can inject a `MagicMock` for the LLM without importing OpenAI / Anthropic / boto3.
- Switching backends requires only changing the `provider` flag in `AnalysisConfig`.

### 2. Lazy Imports Everywhere

All optional heavy dependencies (`openai`, `anthropic`, `boto3`, `torch`, `transformers`, `chromadb`, `faiss`) are imported inside the function or method that first needs them. This keeps module import time under 200 ms regardless of which packages are installed.

### 3. Immutable Dataclasses for Configuration

`AnalysisConfig`, `SummaryConfig`, `SentenceChunkingConfig`, `LLMConfig`, and `RetrievalConfig` are all `@dataclass` instances. They are created once, passed around read-only, and never mutated at runtime — preventing hidden side effects between pipeline runs.

### 4. Disk Cache by Content Hash

PDF extraction is cached by `SHA-256(binary_content) + file_size`. This means:

- Re-running on the same PDF is instant.
- The cache is portable: moving the PDF to a different path still gets a cache hit.
- Editing the PDF content invalidates the cache automatically.

### 5. Dual-Chunker Design

Two separate chunking classes serve different purposes:

- **`SentenceChunker`** — word-limited, used for inference passages. Keeps semantic units intact.
- **`TextChunker`** — character-overlap, used for embedding. Ensures every token appears in at least one chunk.

### 6. Bedrock via Converse API

`BedrockClient` uses the Bedrock **Converse** API rather than model-specific APIs (InvokeModel). This means:

- The same client code works for every model on Bedrock (Anthropic, Amazon, Meta, Mistral, Cohere, AI21).
- System prompts are handled uniformly.
- Streaming is available via `ConverseStream` with the same interface.

---

## Dependency Map

```
document_analysis_pipeline
    ├── PDFExtractor          (pdfplumber)
    ├── TextPreprocessor      (stdlib re / html)
    ├── SentenceChunker       (nltk)
    ├── Summarizer
    │       ├── [HF] transformers.pipeline("summarization")
    │       └── [LLM] BaseLLM + PromptTemplate
    ├── QuestionGenerator
    │       ├── [HF] transformers.pipeline("text2text-generation")
    │       └── [LLM] BaseLLM + PromptTemplate
    └── QAEngine
            ├── [HF] transformers.pipeline("question-answering")
            └── [LLM] BaseLLM + PromptTemplate

ModelFactory
    ├── GPTClient             (openai)
    ├── ClaudeClient          (anthropic)
    ├── BedrockClient         (boto3)
    └── LocalLLM              (transformers)

RAG layer (independent)
    ├── Embedder
    │       ├── OpenAI embeddings  (openai)
    │       └── HF embeddings      (sentence-transformers)
    └── VectorStore
            ├── ChromaDB  (chromadb)
            └── FAISS     (faiss-cpu)
```
