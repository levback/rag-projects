# Configuration Reference

All runtime behaviour is controlled through two YAML files and environment variables. No secrets are stored in config files — API keys and credentials must come from the environment.

---

## File Locations

```
config/
├── model_config.yaml    # Model defaults, provider settings, embedding config
└── logging_config.yaml  # Log levels, handlers, formatters
```

---

## `config/model_config.yaml`

### OpenAI

```yaml
openai:
  default_model: gpt-4o          # Model used when --llm-model is not supplied
  embedding_model: text-embedding-3-small
  temperature: 0.7
  max_tokens: 2048
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
  request_timeout: 60            # Seconds before request is abandoned
  max_retries: 3
```

Credential: set `OPENAI_API_KEY` in the environment. Never put the key in this file.

### Anthropic

```yaml
anthropic:
  default_model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 2048
  request_timeout: 60
  max_retries: 3
```

Credential: set `ANTHROPIC_API_KEY` in the environment.

### Amazon Bedrock

```yaml
bedrock:
  default_model: anthropic.claude-3-5-sonnet-20241022-v2:0
  region: us-east-1
  profile: null        # Set to a named AWS CLI profile, or null for the default chain
  temperature: 0.7
  max_tokens: 2048
  request_timeout: 120
  max_retries: 3
```

**Credential resolution order** (standard boto3 chain):

1. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` environment variables
2. Named profile: `profile: my-profile` in config or `--aws-profile` flag
3. `~/.aws/credentials` default profile
4. IAM instance role / ECS task role / Lambda execution role

**Available Bedrock models** (requires model access enabled in your AWS account):

| Model ID | Provider | Notes |
|---|---|---|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Anthropic | Default; best quality |
| `anthropic.claude-3-haiku-20240307-v1:0` | Anthropic | Fast, cost-efficient |
| `amazon.nova-pro-v1:0` | Amazon | Multimodal |
| `amazon.nova-lite-v1:0` | Amazon | Fastest Nova |
| `meta.llama3-70b-instruct-v1:0` | Meta | Open-weight |
| `mistral.mistral-large-2402-v1:0` | Mistral | Strong reasoning |

### Local Models

```yaml
local:
  model_path: models/            # Directory containing local model weights
  default_model: llama-3.1-8b-instruct
  device: auto                   # auto | cpu | cuda | mps
  quantization: 4bit             # none | 4bit | 8bit
  max_new_tokens: 512
  temperature: 0.7
```

### Document Analysis Pipeline

```yaml
document_analysis:
  provider: huggingface          # huggingface | llm
  llm_provider: bedrock          # bedrock | openai | anthropic | local
  llm_model: null                # null = use provider default
  passage_word_limit: 200        # Max words per passage chunk
  summarization_model: t5-small  # HuggingFace model for summarisation step
```

### Embeddings

```yaml
embeddings:
  provider: openai               # openai | huggingface
  model: text-embedding-3-small  # or e.g. all-MiniLM-L6-v2 for HuggingFace
  batch_size: 100
  dimensions: 1536               # Output vector size (must match your vector store)
```

---

## `config/logging_config.yaml`

Uses Python's standard `logging.config.dictConfig` format.

### Key settings

| Logger | Default level | Notes |
|---|---|---|
| `src.core` | DEBUG | LLM client calls, model creation |
| `src.inference` | DEBUG | Model loading, token counts |
| `src.processing` | INFO | PDF extraction, chunking |
| `src.rag` | DEBUG | Embedding, search operations |
| `root` | INFO | Everything not matched above |

### Handlers

| Handler | File | Max size | Description |
|---|---|---|---|
| `console` | stdout | — | INFO+ to terminal |
| `file` | `logs/app.log` | 10 MB | DEBUG+ rotating |
| `error_file` | `logs/error.log` | 10 MB | ERROR+ rotating |

### Changing the log level at runtime

The simplest way is to pass `--verbose` to the example script:

```bash
bash scripts/run_analysis.sh --verbose
```

Or change the level programmatically:

```python
import logging
logging.getLogger("src.inference").setLevel(logging.WARNING)
```

---

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `GPTClient`, `Embedder` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `ClaudeClient` | Anthropic API key |
| `AWS_ACCESS_KEY_ID` | `BedrockClient` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | `BedrockClient` | AWS secret key |
| `AWS_SESSION_TOKEN` | `BedrockClient` | AWS session token (temporary credentials) |
| `AWS_DEFAULT_REGION` | `BedrockClient` | AWS region (default: `us-east-1`) |
| `AWS_PROFILE` | `BedrockClient` | Named AWS CLI profile |
| `HUGGINGFACE_API_KEY` | HF Hub downloads | Required only for gated models |
| `PYTHONUNBUFFERED` | Docker / CI | Set to `1` for unbuffered stdout |

**Best practice:** store secrets in a `.env` file (never commit it) and load with:

```bash
set -a && source .env && set +a
# then run:
bash scripts/run_analysis.sh --provider llm --llm-provider openai
```

Or use `python-dotenv` in Python code:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ
```

---

## `AnalysisConfig` Python Dataclass

All fields have defaults; pass only what you want to override.

```python
from src.core.document_analysis_pipeline import AnalysisConfig

config = AnalysisConfig(
    provider="huggingface",         # "huggingface" | "llm"
    llm_provider="bedrock",         # which LLM backend when provider=="llm"
    llm_model=None,                 # None = use provider default
    output_dir="examples/output",   # where save_results() writes files
    cache_dir="data/cache",         # where PDFExtractor caches extracted text

    # Summarisation
    summarization_model="t5-small", # HF model name (only for provider="huggingface")
    summary_max_length=150,         # max output tokens for HF summariser
    summary_min_length=30,          # min output tokens for HF summariser

    # Chunking
    passage_word_limit=200,         # words per passage sent to inference

    # Question generation
    qg_model="valhalla/t5-base-qg-hl",
    min_questions_per_passage=3,

    # QA
    qa_model="deepset/roberta-base-squad2",
)
```

---

## `LLMConfig` Python Dataclass

Used internally by `ModelFactory` and passed to every LLM client constructor.

```python
from src.core.base_llm import LLMConfig

config = LLMConfig(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.7,        # 0.0 = deterministic, 1.0 = creative
    max_tokens=2048,        # Maximum tokens to generate
    top_p=1.0,
    stream=False,           # Enable streaming by default
    extra={},               # Provider-specific kwargs
)
```

---

## Data Directories

| Directory | Purpose | Gitignored |
|---|---|---|
| `data/cache/` | PDF extraction cache (content-hash keyed `.txt` files) | Yes |
| `data/embeddings/` | Persisted embedding files | Yes |
| `data/vectordb/` | ChromaDB / FAISS index files | Yes |
| `examples/output/` | Generated analysis outputs | Yes |
| `logs/` | Application log files | Yes |
| `.venv/` | Python virtual environment | Yes |
