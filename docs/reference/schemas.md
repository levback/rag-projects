# API Reference — Output Schemas

JSON output schemas for all 10 example pipelines.  
Every schema is produced by the pipeline's result dataclass `.to_dict()` / `dataclasses.asdict()` call.

Cross-references: [RAG](rag.md) · [Agents](agents.md) · [Examples](../examples.md)

---

## Common fields

All output JSON files share the top-level envelope:

```json
{
  "timestamp": "2025-01-01T00:00:00.000000",
  "example": "NN_example_name",
  "result": { ... }
}
```

---

## Example 01 — Basic RAG

Output file: [`examples/output/01_basic_rag_output.json`](../../examples/output/01_basic_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "retrieved_chunks": ["string", "..."],
  "sources": ["string", "..."]
}
```

---

## Example 02 — IBM Production RAG

Output file: [`examples/output/02_ibm_production_rag_output.json`](../../examples/output/02_ibm_production_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "sources": ["string"],
  "latency_ms": 0.0,
  "retries_used": 0
}
```

---

## Example 03 — Graph RAG

Output file: [`examples/output/03_graph_rag_output.json`](../../examples/output/03_graph_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "graph_context": "string",
  "triples_used": 0
}
```

---

## Example 04 — Multi-Document RAG

Output file: [`examples/output/04_multi_doc_rag_output.json`](../../examples/output/04_multi_doc_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "retrieved_chunks": ["string"],
  "source_documents": ["filename_a.txt", "filename_b.txt"]
}
```

---

## Example 05 — Agentic RAG

Output file: [`examples/output/05_agentic_rag_output.json`](../../examples/output/05_agentic_rag_output.json)

```json
{
  "query": "string",
  "intent": "search | direct",
  "answer": "string",
  "retrieved_chunks": ["string"]
}
```

---

## Example 06 — Realtime RAG

Output file: [`examples/output/06_realtime_rag_output.json`](../../examples/output/06_realtime_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "search_urls": ["https://..."],
  "retrieved_passages": ["string"]
}
```

---

## Example 07 — Research Agent

Output file: [`examples/output/07_research_agent_output.json`](../../examples/output/07_research_agent_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "synthesis": "string",
  "sources": ["https://..."],
  "passage_count": 0,
  "steps": ["step 1", "step 2"]
}
```

---

## Example 08 — Multimodal RAG

Output file: [`examples/output/08_multimodal_rag_output.json`](../../examples/output/08_multimodal_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "retrieved_items": [
    {
      "type": "text | image",
      "content": "string",
      "source": "filename.pdf",
      "page": 1
    }
  ]
}
```

---

## Example 09 — LangChain RAG

Output file: [`examples/output/09_langchain_rag_output.json`](../../examples/output/09_langchain_rag_output.json)

```json
{
  "query": "string",
  "answer": "string",
  "sources": ["string"],
  "mode_used": "chain | agent",
  "retrieved_docs": ["string"],
  "steps": ["string"]
}
```

---

## Example 10 — Document Analysis

Output file: [`examples/output/10_document_analysis_output.json`](../../examples/output/10_document_analysis_output.json)

```json
{
  "pdf_path": "string",
  "summary": "string",
  "passage_analyses": [
    {
      "passage": "string",
      "questions": ["string"],
      "answers": [
        {
          "question": "string",
          "answer": "string",
          "score": 0.0,
          "start": -1,
          "end": -1
        }
      ]
    }
  ],
  "metadata": {
    "chunk_count": 0,
    "provider": "string",
    "llm_provider": "string"
  }
}
```

---

## Saving output programmatically

```python
import json, dataclasses
from pathlib import Path

def save_result(result, name: str) -> Path:
    out = Path("examples/output") / f"{name}_output.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)
    return out
```
