# User Guide

Step-by-step instructions for running the Document Analysis pipeline with every supported backend.

---

## Prerequisites

- Python 3.10 or later
- macOS or Linux (arm64 and x86_64 both supported)
- The virtual environment set up via `scripts/setup_env.sh`

```bash
# Clone and bootstrap
git clone <repo-url>
cd document_analysis
bash scripts/setup_env.sh
```

---

## 1. Prepare a PDF

### Use the bundled example

```bash
bash scripts/download_example_pdf.sh
# Downloads: examples/pdfs/attention_is_all_you_need.pdf
```

### Use your own PDF

Pass any PDF path with `--pdf`:

```bash
bash scripts/run_analysis.sh --pdf /path/to/my-document.pdf
```

---

## 2. Choose a Backend

The `--provider` flag controls the inference backend:

| `--provider` | What it uses | Credentials needed |
|---|---|---|
| `huggingface` | Local models (T5, RoBERTa) | None |
| `llm` | External LLM API | See `--llm-provider` |

When `--provider llm` is set, `--llm-provider` selects the API:

| `--llm-provider` | Service | Credential |
|---|---|---|
| `bedrock` | Amazon Bedrock | AWS credentials (see below) |
| `openai` | OpenAI | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Claude | `ANTHROPIC_API_KEY` |
| `local` | Local model via Transformers | None |

---

## 3. Backend-Specific Instructions

### HuggingFace (default — no credentials)

Models download automatically on first run and are cached in the HuggingFace cache directory (`~/.cache/huggingface/`).

```bash
bash scripts/run_analysis.sh
# or explicitly:
bash scripts/run_analysis.sh --provider huggingface
```

**Models used:**

| Step | Model |
|---|---|
| Summarisation | `t5-small` |
| Question generation | `valhalla/t5-base-qg-hl` |
| Question answering | `deepset/roberta-base-squad2` |

**First-run download sizes** (approximate):

| Model | Size |
|---|---|
| `t5-small` | ~240 MB |
| `valhalla/t5-base-qg-hl` | ~850 MB |
| `deepset/roberta-base-squad2` | ~480 MB |

---

### Amazon Bedrock

Bedrock uses the standard **boto3 credential chain** — the same mechanism as the AWS CLI.

#### Option A — Environment variables

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

bash scripts/run_analysis.sh --provider llm --llm-provider bedrock
```

#### Option B — Named CLI profile

```bash
bash scripts/run_analysis.sh \
  --provider llm \
  --llm-provider bedrock \
  --aws-profile my-profile \
  --aws-region us-east-1
```

#### Option C — IAM role / default chain

If you are on an EC2 instance, ECS task, or have a default profile configured, no flags are needed:

```bash
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock
```

#### Choosing a Bedrock model

The default is `anthropic.claude-3-5-sonnet-20241022-v2:0`. Override with `--llm-model`:

```bash
# Amazon Nova Pro
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock \
  --llm-model amazon.nova-pro-v1:0

# Claude 3 Haiku (faster, cheaper)
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock \
  --llm-model anthropic.claude-3-haiku-20240307-v1:0

# Meta Llama 3 70B
bash scripts/run_analysis.sh --provider llm --llm-provider bedrock \
  --llm-model meta.llama3-70b-instruct-v1:0
```

> **Note:** Model availability varies by AWS region. Ensure you have requested access to the model in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess).

---

### OpenAI

```bash
export OPENAI_API_KEY=sk-...

bash scripts/run_analysis.sh --provider llm --llm-provider openai
```

Override the default model (`gpt-4o`):

```bash
bash scripts/run_analysis.sh \
  --provider llm \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

---

### Anthropic Claude (direct API)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

bash scripts/run_analysis.sh --provider llm --llm-provider anthropic
```

Override the default model (`claude-3-5-sonnet-20241022`):

```bash
bash scripts/run_analysis.sh \
  --provider llm \
  --llm-provider anthropic \
  --llm-model claude-3-haiku-20240307
```

---

### Local model via Transformers

Runs a local `text-generation` pipeline. GPU or Apple MPS is used automatically if available.

```bash
bash scripts/run_analysis.sh --provider llm --llm-provider local \
  --llm-model /path/to/local-model
```

---

## 4. All Script Flags

```
bash scripts/run_analysis.sh [OPTIONS]

  --provider      huggingface|llm        (default: huggingface)
  --llm-provider  bedrock|openai|anthropic|local  (default: bedrock)
  --llm-model     MODEL_ID               Override default model
  --aws-region    REGION                 AWS region for Bedrock (default: us-east-1)
  --aws-profile   PROFILE                Named AWS CLI profile
  --pdf           PATH                   PDF to analyse
  --word-limit    N                      Max words per passage (default: 200)
  --output-dir    DIR                    Output directory (default: examples/output)
  --help                                 Show help
```

---

## 5. Reading the Outputs

After a run, three files appear in `examples/output/` (or your `--output-dir`):

### `<stem>_analysis.json`

Complete structured result. Top-level keys:

```json
{
  "source": "examples/pdfs/attention_is_all_you_need.pdf",
  "extracted_text": "...",
  "text_preview": "...",
  "summary": "...",
  "num_passages": 5,
  "passages": [
    {
      "passage_index": 0,
      "passage": "...",
      "questions": ["Q1", "Q2", "Q3"],
      "qa_pairs": [
        { "question": "Q1", "answer": "...", "score": 0.87 }
      ]
    }
  ],
  "all_qa_pairs": [
    { "question": "...", "answer": "...", "score": 0.87 }
  ]
}
```

### `<stem>_report_<timestamp>.txt`

Human-readable text version of the same data: text preview, summary, all Q&A pairs.

### `<stem>_run_<timestamp>.log`

Execution log from stderr: step timings, model loading events, passage counts.

---

## 6. Using the Python API Directly

### Minimal example

```python
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

pipeline = DocumentAnalysisPipeline(config=AnalysisConfig(provider="huggingface"))
result = pipeline.run("examples/pdfs/attention_is_all_you_need.pdf")

print(result.summary)
pipeline.save_results(result, output_dir="examples/output")
```

### With Bedrock

```python
from src.core.model_factory import create_llm
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

llm = create_llm(
    "bedrock",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
    # profile_name="my-profile",  # optional
)

config = AnalysisConfig(provider="llm", llm_provider="bedrock")
pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
result = pipeline.run("my-document.pdf")
```

### With OpenAI

```python
import os
from src.core.model_factory import create_llm
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

llm = create_llm("openai", model="gpt-4o")   # reads OPENAI_API_KEY from env
config = AnalysisConfig(provider="llm", llm_provider="openai")
pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
result = pipeline.run("my-document.pdf")
```

### With Anthropic

```python
from src.core.model_factory import create_llm
from src.core.document_analysis_pipeline import AnalysisConfig, DocumentAnalysisPipeline

llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
config = AnalysisConfig(provider="llm", llm_provider="anthropic")
pipeline = DocumentAnalysisPipeline(config=config, llm=llm)
result = pipeline.run("my-document.pdf")
```

### Using an LLM directly (without the pipeline)

```python
from src.core.model_factory import create_llm
from src.core.base_llm import Message

llm = create_llm("bedrock")

# Simple completion
text = llm.chat("Explain the transformer attention mechanism in two sentences.")
print(text)

# Full message list with system prompt
response = llm.complete([
    Message(role="system", content="You are a concise technical writer."),
    Message(role="user", content="What is self-attention?"),
])
print(response.content, response.usage)
```

---

## 7. Building a Vector Index

The RAG layer lets you index documents and retrieve the most relevant passages for any query.

```bash
bash scripts/build_embeddings.sh --input-dir data/documents/
```

Or programmatically:

```python
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore, Document
from src.rag.indexer import Indexer
from src.rag.retriever import Retriever

embedder = Embedder(provider="openai")          # or "huggingface"
store = VectorStore(provider="chroma")           # or "faiss"
indexer = Indexer(embedder=embedder, vector_store=store)

# Index documents
indexer.index_texts(["passage one", "passage two", "passage three"])

# Retrieve
retriever = Retriever(embedder=embedder, vector_store=store)
results = retriever.retrieve("attention mechanism in transformers", top_k=3)
for r in results:
    print(r.score, r.document.text)
```

---

## 8. Running with Docker

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, AWS_* vars, etc.

# Start app + ChromaDB
docker compose up --build

# App container only
docker build -t document-analysis .
docker run --env-file .env -v $(pwd)/data:/app/data document-analysis \
  python examples/analyze_pdf.py --pdf /app/data/pdfs/my-doc.pdf
```

---

## 9. Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pdfplumber` | venv not activated or dep not installed | Run `bash scripts/setup_env.sh` |
| `OSError: … t5-small … 401` | HuggingFace Hub requires auth for `google/t5-small` | The config already uses `t5-small` without the `google/` prefix; no action needed |
| `sentencepiece` import error on first QG run | Missing tokenizer package | `pip install sentencepiece` |
| `NoCredentialsError` from boto3 | No AWS credentials configured | Set env vars or `--aws-profile` |
| `ValidationException` from Bedrock | Model not enabled in your account | Enable the model in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) |
| Slow first run | HuggingFace models downloading | Normal; subsequent runs use the cache in `~/.cache/huggingface/` |
| `Token indices sequence length > 512` warning | Input longer than model max | Expected; the pipeline chunks inputs before passing to the model |
