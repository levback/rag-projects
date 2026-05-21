# Get Started

Get up and running in 5 minutes — no API keys required.

Cross-references: [Overview](overview.md) · [Installation Guide](../installation.md) · [Learn Basics](learn-basics.md)

---

## Step 1 — Install

```bash
git clone git@github.com:levback/rag-projects.git
cd rag-projects
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Full details in [Installation Guide](../installation.md).

---

## Step 2 — Run your first example

```bash
python examples/01_basic_rag.py
```

Expected output:

```
============================================================
  Project #1 — Basic RAG Pipeline
============================================================

[1/3] Indexing corpus …
    Indexed 5 chunks in 0.17s

[2/3] Running queries …

  Q: How does multi-head attention work in the Transformer?
  A: Multi-head attention allows the model to jointly attend to information …

[3/3] Output saved → examples/output/01_basic_rag_output.json
```

---

## Step 3 — Run all 10 examples

```bash
bash scripts/run_examples.sh
```

All examples complete with no API keys. Each writes a JSON file to
`examples/output/`.

---

## Step 4 — Use a real LLM

Set environment variables and pass `--provider`:

```bash
# Amazon Bedrock (Claude 3.5 Sonnet — recommended)
export AWS_DEFAULT_REGION=us-east-1
python examples/01_basic_rag.py --provider bedrock

# OpenAI GPT-4o-mini
export OPENAI_API_KEY=sk-...
python examples/01_basic_rag.py --provider openai

# Anthropic Claude direct
export ANTHROPIC_API_KEY=sk-ant-...
python examples/01_basic_rag.py --provider anthropic
```

---

## Step 5 — Use a RAG pipeline in your own code

```python
from src.rag.basic_rag import BasicRAGConfig, BasicRAGPipeline
from src.core.model_factory import create_llm

# Build the pipeline
cfg = BasicRAGConfig(embedding_model="all-MiniLM-L6-v2", top_k=3)
llm = create_llm("bedrock")          # or "openai", "anthropic", "local"
pipeline = BasicRAGPipeline(config=cfg, llm=llm)

# Index documents
pipeline.index(
    texts=["LangChain simplifies building LLM apps...", "..."],
    sources=["langchain_docs.txt", "openai_docs.txt"],
)

# Query
result = pipeline.query("What is LangChain?")
print(result.answer)
print(result.retrieved_chunks)
```

See [Main Components](main-components.md) for all available pipelines.

---

## Step 6 — Analyse a PDF

```bash
# Download the bundled example PDF
bash scripts/download_example_pdf.sh

# Run the full document analysis pipeline
python examples/10_document_analysis.py --pdf examples/pdfs/attention_is_all_you_need.pdf
```

Or use the original CLI:

```bash
bash scripts/run_analysis.sh --provider huggingface
```

---

## What next?

| Goal | Go to |
|------|-------|
| Understand RAG concepts | [Learn Basics](learn-basics.md) |
| Explore each pipeline class | [Main Components](main-components.md) |
| Understand the 10 examples | [Examples Guide](../examples.md) |
| Contribute or extend | [Developer Notes](developer-notes.md) |
| Look up a specific class/method | [API Reference](../reference/index.md) |
