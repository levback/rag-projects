# Installation Guide

Cross-references: [Get Started](user-guide/get-started.md) · [Configuration](configuration.md) · [Examples](examples.md)

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12 |
| RAM | 4 GB | 8 GB (16 GB for local models) |
| Disk | 2 GB | 10 GB (model cache) |
| OS | macOS 12+, Ubuntu 20.04+ | macOS 14+ arm64 / Ubuntu 22.04 |
| GPU | — | Apple Silicon MPS or CUDA (optional) |

---

## 1. Clone the repository

```bash
git clone git@github.com:levback/rag-projects.git
cd rag-projects
```

---

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Or use the bootstrap script (also installs all dependencies):

```bash
bash scripts/setup_env.sh
```

---

## 3. Install dependencies

### Full install (all backends + extras)

```bash
pip install -r requirements.txt
```

### Minimal install (local HuggingFace only — no cloud APIs)

```bash
pip install pdfplumber nltk transformers torch accelerate \
            sentence-transformers faiss-cpu networkx \
            beautifulsoup4 requests pydantic python-dotenv pyyaml
```

### Cloud API install (OpenAI / Anthropic / Bedrock — no local models)

```bash
pip install pdfplumber nltk sentence-transformers faiss-cpu \
            openai anthropic boto3 \
            langchain langchain-community langchain-core \
            langchain-openai langchain-anthropic langchain-aws \
            pydantic python-dotenv pyyaml
```

---

## 4. Download NLTK data (required for sentence chunking)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 5. Configure credentials

Copy the example environment file and fill in the keys you need:

```bash
cp .env.example .env   # if provided, otherwise create .env manually
```

**.env file format:**

```dotenv
# ── OpenAI ───────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── Anthropic ────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Amazon Bedrock ───────────────────────────────────────────────
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

Credentials are never required for demo mode (`python examples/NN_*.py`).

### AWS Bedrock — additional setup

1. Enable models in the [Bedrock console](https://console.aws.amazon.com/bedrock/) → Model access
2. Ensure your IAM role/user has `bedrock:InvokeModel` permission
3. Default model: `anthropic.claude-3-5-sonnet-20241022-v2:0`

See [`config/model_config.yaml`](../config/model_config.yaml) to override model IDs.

---

## 6. Verify the installation

```bash
# Run all 10 examples in demo mode (no API keys)
bash scripts/run_examples.sh

# Run the test suite
bash scripts/run_tests.sh

# Run security checks
bash scripts/security_check.sh
```

Expected output from `run_examples.sh`: 10/10 examples pass with output files written to `examples/output/`.

---

## 7. Docker (optional)

A `Dockerfile` and `docker-compose.yml` are provided for containerised deployment.

```bash
docker compose up --build
```

The container:
- Uses Python 3.12-slim
- Installs all requirements
- Exposes the analysis CLI via `main.py`
- Mounts `data/` and `examples/output/` as volumes

---

## Troubleshooting

### `SIGABRT` on Apple Silicon with `sentence_transformers`

If you see a crash when importing `sentence_transformers`, ensure all native
extensions are compiled for arm64:

```bash
pip install --upgrade --force-reinstall pydantic-core cryptography
```

### `transformers` v5 breaks QA/summarisation pipelines

Pin to `<5.0.0` as in `requirements.txt`. v5 removed the seq2seq task APIs used
by `t5-small` and `valhalla/t5-base-qg-hl`.

### ChromaDB `sqlite3` version error

ChromaDB requires SQLite ≥ 3.35. On older Ubuntu systems:

```bash
pip install pysqlite3-binary
```

Then add to `src/__init__.py`:

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

### `duckduckgo-search` rate limits

The `RealtimeRAGAssistant` and `ResearchAgent` may hit DuckDuckGo rate limits
in live mode. Use demo mode for development, and add `scrape_timeout` and
`num_search_results` to `ResearchAgentConfig` to tune live mode.
