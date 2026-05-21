# Document Analysis

A production-ready, end-to-end document analysis pipeline that extracts text from PDFs, generates summaries, auto-generates questions, and answers them — powered by your choice of **HuggingFace local models**, **Amazon Bedrock**, **OpenAI**, or **Anthropic Claude**.

```
PDF  →  Extract  →  Preprocess  →  Chunk  →  Summarise  →  Generate Questions  →  Answer
```

## Features

- **Four inference backends** — swap between HuggingFace (no API key), Amazon Bedrock, OpenAI, or Anthropic with a single flag
- **Disk-cached PDF extraction** — re-runs are instant; raw text is cached by content hash
- **Sentence-aware chunking** — NLTK-based splitting keeps semantic boundaries intact
- **Dual-mode summarisation** — T5-small locally or any LLM via a structured prompt
- **Auto question generation** — passage-level questions via `valhalla/t5-base-qg-hl` or LLM
- **Extractive + generative QA** — RoBERTa-SQuAD2 locally or LLM with context
- **RAG layer** — `Embedder` + `VectorStore` (Chroma / FAISS) + `Retriever` for document search
- **Prompt chaining** — `PromptTemplate` and `PromptChain` for multi-step LLM workflows
- **98 % test coverage** — 245 unit tests, pytest + pytest-cov
- **Docker-ready** — `Dockerfile` + `docker-compose.yml` with ChromaDB sidecar

## Quick Start

### 1. Set up the environment

```bash
bash scripts/setup_env.sh
```

### 2. Download the example PDF

```bash
bash scripts/download_example_pdf.sh
# → examples/pdfs/attention_is_all_you_need.pdf
```

### 3. Run the analysis

```bash
# HuggingFace — no credentials required
bash scripts/run_analysis.sh

# Amazon Bedrock
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock --aws-profile my-profile

# OpenAI
OPENAI_API_KEY=sk-... bash scripts/run_analysis.sh --provider llm --llm-provider openai

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-... bash scripts/run_analysis.sh --provider llm --llm-provider anthropic
```

Outputs are written to `examples/output/`:

| File | Content |
|---|---|
| `*_analysis.json` | Full structured result (passages, summary, all Q&A pairs) |
| `*_report_<ts>.txt` | Human-readable text report |
| `*_run_<ts>.log` | Pipeline execution log |

## Installation

**Requirements:** Python 3.10+, macOS / Linux

```bash
git clone <repo-url>
cd document_analysis
bash scripts/setup_env.sh        # creates .venv/ and installs all dependencies
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Provider Credentials

| Provider | What you need |
|---|---|
| `huggingface` | Nothing — models download automatically on first run |
| `bedrock` | AWS credentials via env vars, `~/.aws/credentials`, named profile, or IAM role |
| `openai` | `OPENAI_API_KEY` environment variable |
| `anthropic` | `ANTHROPIC_API_KEY` environment variable |
| `local` | Nothing — runs a local model via Transformers (GPU/MPS recommended) |

### Bedrock credential options

```bash
# Option A — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock

# Option B — named CLI profile
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock --aws-profile my-profile

# Option C — default chain (IAM role / ~/.aws/credentials)
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock
```

## Using the Python API

```python
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

# HuggingFace (no LLM client needed)
config = AnalysisConfig(provider="huggingface")
pipeline = DocumentAnalysisPipeline(config=config)
result = pipeline.run("path/to/document.pdf")

print(result.summary)
for pair in result.all_qa_pairs:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}")

pipeline.save_results(result, output_dir="examples/output")
```

```python
from src.core.model_factory import create_llm
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

# Amazon Bedrock
llm = create_llm("bedrock", model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                 region_name="us-east-1")

config = AnalysisConfig(provider="llm", llm_provider="bedrock")
pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
result = pipeline.run("path/to/document.pdf")
```

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup_env.sh` | Create / refresh `.venv/`, install all dependencies |
| `scripts/run_analysis.sh` | Run the full pipeline and save all outputs |
| `scripts/run_tests.sh` | Run unit tests with 95 % coverage threshold |
| `scripts/download_example_pdf.sh` | Download the "Attention Is All You Need" paper |
| `scripts/build_embeddings.sh` | Build and persist a vector index from documents |

Run `bash scripts/run_analysis.sh --help` for the full list of flags.

## Project Structure

```
document_analysis/
├── src/
│   ├── core/           # LLM clients, pipeline orchestrator, model factory
│   ├── inference/      # Summarizer, QuestionGenerator, QAEngine
│   ├── processing/     # PDFExtractor, TextPreprocessor, SentenceChunker, Tokenizer
│   ├── prompts/        # PromptTemplate, PromptChain, built-in templates
│   └── rag/            # Embedder, VectorStore (Chroma/FAISS), Retriever, Indexer
├── config/
│   ├── model_config.yaml     # Model defaults, provider settings
│   └── logging_config.yaml   # Log levels and handlers
├── examples/
│   ├── analyze_pdf.py        # Full CLI example script
│   ├── pdfs/                 # Example PDF files
│   └── output/               # Generated analysis outputs
├── scripts/                  # Bash automation scripts
├── tests/
│   ├── unit/                 # 245 unit tests (98 % coverage)
│   └── integration/          # Integration test stubs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Running Tests

```bash
bash scripts/run_tests.sh
# or directly:
.venv/bin/python -m pytest tests/unit --cov=src --cov-fail-under=95
```

Current status: **245 tests · 98 % coverage · ~5.5 s**

## Docker

```bash
# Start the app + ChromaDB
docker compose up --build

# App only
docker build -t document-analysis .
docker run --env-file .env document-analysis
```

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, component diagram, data-flow |
| [User Guide](docs/user-guide.md) | Step-by-step usage for every backend |
| [Configuration](docs/configuration.md) | Config YAML reference, env vars, model options |
| [API Reference](docs/api-reference.md) | Public Python API for every module |
| [Development Guide](docs/development.md) | Dev setup, testing, adding new backends |
