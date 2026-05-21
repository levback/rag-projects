# Examples Guide

Each of the 10 runnable examples lives in [`examples/`](../examples/) and has a
pre-computed JSON output in [`examples/output/`](../examples/output/).
All examples run in **demo mode** with no API keys required.

Cross-references: [Installation Guide](installation.md) · [User Guide](user-guide/get-started.md) · [API Reference](reference/index.md)

---

## Quick start — run all examples

```bash
bash scripts/run_examples.sh          # all 10, demo mode
bash scripts/run_examples.sh 01 03    # specific examples
bash scripts/run_examples.sh --live   # examples 06/07 use real web
```

Each script accepts the same flags:

| Flag | Effect |
|------|--------|
| *(none)* | Demo mode — built-in `_MockLLM`, no API calls |
| `--provider bedrock` | Use Amazon Bedrock (requires AWS credentials) |
| `--provider openai` | Use OpenAI (requires `OPENAI_API_KEY`) |
| `--live` | Examples 06/07 only: real DuckDuckGo + HTTP scraping |

---

## Example 01 — Basic RAG Pipeline

**File:** [`examples/01_basic_rag.py`](../examples/01_basic_rag.py)  
**Output:** [`examples/output/01_basic_rag_output.json`](../examples/output/01_basic_rag_output.json)  
**Source module:** [`src/rag/basic_rag.py`](../src/rag/basic_rag.py)  
**API reference:** [BasicRAGPipeline](reference/rag.md#basicragpipeline)

Demonstrates the canonical RAG loop:

1. Chunk a 5-document AI/NLP corpus
2. Embed with `all-MiniLM-L6-v2` (or demo hash embedder)
3. Index in FAISS
4. Retrieve top-k chunks per query
5. Generate a grounded answer via `BaseLLM.complete()`

**Questions answered:**
- Multi-head attention in the Transformer
- BERT pre-training objectives
- How RAG reduces hallucinations
- FAISS vs Chroma comparison

```bash
python examples/01_basic_rag.py
```

---

## Example 02 — IBM Production RAG

**File:** [`examples/02_ibm_production_rag.py`](../examples/02_ibm_production_rag.py)  
**Output:** [`examples/output/02_ibm_production_rag_output.json`](../examples/output/02_ibm_production_rag_output.json)  
**Source module:** [`src/rag/ibm_rag.py`](../src/rag/ibm_rag.py)  
**API reference:** [IBMProductionRAG](reference/rag.md#ibmproductionrag)

Enterprise-grade RAG with production operational features:

- Retry logic with exponential back-off (`max_retries`, `retry_delay`)
- Per-query latency tracking (`latency_ms` in result)
- Pipeline statistics (`stats` property)
- IBM Granite embedding support

```bash
python examples/02_ibm_production_rag.py
```

---

## Example 03 — Graph RAG

**File:** [`examples/03_graph_rag.py`](../examples/03_graph_rag.py)  
**Output:** [`examples/output/03_graph_rag_output.json`](../examples/output/03_graph_rag_output.json)  
**Source module:** [`src/rag/graph_rag.py`](../src/rag/graph_rag.py)  
**Supporting modules:** [`src/knowledge_graph/graph_store.py`](../src/knowledge_graph/graph_store.py) · [`src/knowledge_graph/triple_extractor.py`](../src/knowledge_graph/triple_extractor.py)  
**API reference:** [GraphRAGPipeline](reference/rag.md#graphragpipeline) · [KnowledgeGraphStore](reference/knowledge_graph.md#knowledgegraphstore)

Knowledge-graph-backed retrieval:

1. Extract `(head, relation, tail)` triples from text with an LLM
2. Store in a NetworkX DiGraph (`KnowledgeGraphStore`)
3. Answer questions via DFS multi-hop traversal
4. Demo includes 21 pre-seeded triples about Einstein, Turing, and Deep Learning

```bash
python examples/03_graph_rag.py
```

---

## Example 04 — Multi-Document RAG

**File:** [`examples/04_multi_doc_rag.py`](../examples/04_multi_doc_rag.py)  
**Output:** [`examples/output/04_multi_doc_rag_output.json`](../examples/output/04_multi_doc_rag_output.json)  
**Source module:** [`src/rag/multi_doc_rag.py`](../src/rag/multi_doc_rag.py)  
**API reference:** [MultiDocumentRAG](reference/rag.md#multidocumentrag)

Queries across multiple heterogeneous documents simultaneously:

- 3 documents: `climate_science.txt`, `renewable_energy.md`, `carbon_markets.txt`
- Source provenance tracked per retrieved chunk
- Queries span document boundaries (e.g. "which documents discuss carbon prices?")

```bash
python examples/04_multi_doc_rag.py
```

---

## Example 05 — Agentic RAG (Intent Routing)

**File:** [`examples/05_agentic_rag.py`](../examples/05_agentic_rag.py)  
**Output:** [`examples/output/05_agentic_rag_output.json`](../examples/output/05_agentic_rag_output.json)  
**Source module:** [`src/rag/agentic_rag.py`](../src/rag/agentic_rag.py)  
**API reference:** [AgenticRAGPipeline](reference/rag.md#agenticragpipeline)

Intent-aware routing:

- **SEARCH** intent → vector retrieval + generation
- **DIRECT** intent → LLM answers without retrieval (greetings, meta-questions)
- 7 queries: 4 factual (SEARCH), 3 conversational (DIRECT)

```bash
python examples/05_agentic_rag.py
```

---

## Example 06 — Real-time RAG

**File:** [`examples/06_realtime_rag.py`](../examples/06_realtime_rag.py)  
**Output:** [`examples/output/06_realtime_rag_output.json`](../examples/output/06_realtime_rag_output.json)  
**Source module:** [`src/rag/realtime_rag.py`](../src/rag/realtime_rag.py)  
**API reference:** [RealtimeRAGAssistant](reference/rag.md#realtimeragassistant)

Web-search-grounded answering:

- Simulated DuckDuckGo results (demo) or live search (`--live`)
- Scrape + chunk top pages
- Rank passages by cosine similarity
- Generate with prompt-injection-safe delimiters

```bash
python examples/06_realtime_rag.py          # demo
python examples/06_realtime_rag.py --live   # real web
```

---

## Example 07 — Research Agent

**File:** [`examples/07_research_agent.py`](../examples/07_research_agent.py)  
**Output:** [`examples/output/07_research_agent_output.json`](../examples/output/07_research_agent_output.json)  
**Source module:** [`src/agents/research_agent.py`](../src/agents/research_agent.py)  
**API reference:** [ResearchAgent](reference/agents.md#researchagent)

Multi-step agentic research workflow:

1. DuckDuckGo search → top URLs
2. Scrape + chunk pages
3. Rank passages by TF-IDF cosine similarity
4. Synthesise findings across sources
5. Return structured research report

```bash
python examples/07_research_agent.py          # demo
python examples/07_research_agent.py --live   # real web
```

---

## Example 08 — Multimodal RAG

**File:** [`examples/08_multimodal_rag.py`](../examples/08_multimodal_rag.py)  
**Output:** [`examples/output/08_multimodal_rag_output.json`](../examples/output/08_multimodal_rag_output.json)  
**Source module:** [`src/rag/multimodal_rag.py`](../src/rag/multimodal_rag.py)  
**API reference:** [MultimodalRAGPipeline](reference/rag.md#multimodalragpipeline)

Text + image fusion pipeline:

- PDF page images captioned by a Vision LLM (GPT-4V / Claude 3 / BLIP-2)
- Text chunks + image captions indexed in FAISS
- Queries retrieve from both modalities
- Demo uses 4 pre-captioned figures from "Attention Is All You Need"

```bash
python examples/08_multimodal_rag.py
```

---

## Example 09 — LangChain RAG Agent

**File:** [`examples/09_langchain_rag_agent.py`](../examples/09_langchain_rag_agent.py)  
**Output:** [`examples/output/09_langchain_rag_agent_output.json`](../examples/output/09_langchain_rag_agent_output.json)  
**Source module:** [`src/agents/langchain_rag_agent.py`](../src/agents/langchain_rag_agent.py)  
**API reference:** [LangChainRAGAgent](reference/agents.md#langchainragagent)

Full LangChain integration:

- **Chain mode** (`mode="chain"`): LCEL `prompt | llm | output_parser`
- **Agent mode** (`mode="agent"`): ReAct tool-calling with a retriever tool
- Supports OpenAI, Bedrock (Claude/Titan/Llama), Anthropic, HuggingFace backends

```bash
python examples/09_langchain_rag_agent.py
```

---

## Example 10 — Document Analysis Pipeline

**File:** [`examples/10_document_analysis.py`](../examples/10_document_analysis.py)  
**Output:** [`examples/output/10_document_analysis_output.json`](../examples/output/10_document_analysis_output.json)  
**Source module:** [`src/core/document_analysis_pipeline.py`](../src/core/document_analysis_pipeline.py)  
**API reference:** [DocumentAnalysisPipeline](reference/core.md#documentanalysispipeline)

Full document analysis pipeline:

- PDF text extraction (pdfplumber)
- Section detection and chunking
- Summarisation (T5 / LLM)
- Question generation
- Question answering (RoBERTa / LLM)
- Named entity extraction

```bash
python examples/10_document_analysis.py          # demo (pre-computed)
python examples/10_document_analysis.py --pdf path/to/doc.pdf  # real PDF
```

---

## Output format

Every example writes a JSON file to `examples/output/` with the structure:

```json
{
  "project": "NN_project_name",
  "description": "...",
  "config": { "...": "..." },
  "results": [
    {
      "question": "...",
      "answer": "..."
    }
  ]
}
```

See [API Reference → Output schemas](reference/schemas.md) for field-level documentation.
