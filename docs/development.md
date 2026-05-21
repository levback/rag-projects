# Development Guide

Instructions for setting up a development environment, running tests, understanding the test suite, and extending the system with new backends.

---

## Environment Setup

### Requirements

- Python 3.10+ (3.12 recommended)
- macOS (arm64 or x86_64) or Linux
- No GPU required; Apple MPS and CUDA are used automatically when available

### Bootstrap

```bash
git clone <repo-url>
cd document_analysis

# Create .venv and install all dependencies
bash scripts/setup_env.sh

# Or do it manually
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Clean reinstall

```bash
bash scripts/setup_env.sh --clean
```

---

## Running Tests

```bash
# Full suite with 95 % coverage threshold (recommended)
bash scripts/run_tests.sh

# Directly with pytest
.venv/bin/python -m pytest tests/unit -q --tb=short

# With coverage report
.venv/bin/python -m pytest tests/unit --cov=src --cov-report=term-missing

# Single test file
.venv/bin/python -m pytest tests/unit/test_document_analysis.py -v

# Single test by name
.venv/bin/python -m pytest tests/unit -k "test_summarize_passages"
```

### Current Status

| Metric | Value |
|---|---|
| Total tests | 245 |
| Coverage | 98 % |
| Runtime | ~5.5 s |
| Framework | pytest 9.0.3 + pytest-cov 7.1.0 + pytest-asyncio 1.3.0 |

### Test files

| File | What it tests |
|---|---|
| `test_document_analysis.py` | Full pipeline, `Summarizer`, `QuestionGenerator`, `QAEngine` |
| `test_clients_and_backends.py` | `GPTClient`, `ClaudeClient`, `LocalLLM`, HF backends, `TextChunker` |
| `test_bedrock_client.py` | `BedrockClient` (all auth paths, streaming, error handling) |
| `test_preprocessing.py` | `PDFExtractor`, `TextPreprocessor`, `SentenceChunker`, `Tokenizer` |
| `test_rag.py` | `Embedder`, `VectorStore` (Chroma + FAISS), `Retriever`, `Indexer` |
| `test_extended.py` | Edge cases, error paths, additional coverage |
| `test_llm_clients.py` | `BaseLLM` interface, `ModelFactory` |
| `test_prompts.py` | `PromptTemplate`, `PromptChain`, `ChainContext` |

---

## Architecture Conventions

### `from __future__ import annotations`

Every source file starts with this import. It enables PEP 563 postponed evaluation of annotations, which means forward references in type hints don't need to be quoted.

### Lazy imports

All optional heavy dependencies (`openai`, `anthropic`, `boto3`, `torch`, `transformers`, `chromadb`, `faiss`) are imported inside the function or method that first uses them. Never at module level. This prevents `ImportError` if an optional package is not installed.

```python
# GOOD — lazy
def _get_hf_pipeline(self):
    if self._hf_pipeline is None:
        from transformers import pipeline
        self._hf_pipeline = pipeline("summarization", model=self._cfg.hf_model)
    return self._hf_pipeline

# BAD — eager top-level import
import transformers
```

### Immutable dataclasses for config

All configuration objects (`AnalysisConfig`, `SummaryConfig`, `LLMConfig`, etc.) are `@dataclass` instances created once and never mutated.

### No mutation of inputs

Functions return new objects. They never modify their arguments in place.

---

## Adding a New LLM Backend

1. **Create the client** in `src/core/my_llm_client.py`:

```python
from __future__ import annotations
from src.core.base_llm import BaseLLM, LLMConfig, LLMResponse, Message
from typing import AsyncIterator, Iterator

class MyLLMClient(BaseLLM):
    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        self._api_key = api_key

    def complete(self, messages: list[Message]) -> LLMResponse:
        # Call your API here
        ...
        return LLMResponse(content=..., model=self.config.model, usage={})

    def stream(self, messages: list[Message]) -> Iterator[str]:
        ...

    async def acomplete(self, messages: list[Message]) -> LLMResponse:
        ...

    async def astream(self, messages: list[Message]) -> AsyncIterator[str]:
        ...
```

2. **Register in `ModelFactory`** (`src/core/model_factory.py`):

```python
class ModelProvider(str, Enum):
    # ... existing entries ...
    MY_PROVIDER = "my_provider"

_PROVIDER_MAP[ModelProvider.MY_PROVIDER] = ("src.core.my_llm_client", "MyLLMClient")
_DEFAULT_MODELS[ModelProvider.MY_PROVIDER] = "my-default-model-id"
```

3. **Add credential resolution** (if needed) in `_resolve_api_key()`:

```python
env_map = {
    ...
    ModelProvider.MY_PROVIDER: "MY_PROVIDER_API_KEY",
}
```

4. **Write tests** in `tests/unit/test_clients_and_backends.py`:

```python
class TestMyLLMClient:
    def _make_client(self):
        import src.core.my_llm_client  # ensure module is loaded before patching
        config = LLMConfig(model="my-model")
        return MyLLMClient(config, api_key="test-key")

    def test_complete_returns_response(self):
        with patch("src.core.my_llm_client.MySDK") as mock_sdk:
            mock_sdk.return_value.chat.return_value = MagicMock(content="answer")
            client = self._make_client()
            resp = client.complete([Message(role="user", content="hi")])
            assert resp.content == "answer"
```

5. **Add to the bash script** (optional) — add `my_provider` to the `--llm-provider` case statement in `scripts/run_analysis.sh`.

---

## Adding a New Vector Store Backend

1. Add an `elif self._provider == "my_store":` branch to `VectorStore._get_store()` in `src/rag/vector_store.py`.

2. Implement `_init_my_store()`, and ensure `upsert`, `search`, `delete`, and `count` are handled in the public methods.

3. Write tests in `tests/unit/test_rag.py` following the existing `TestVectorStoreFaiss` pattern.

---

## macOS arm64 Notes

On Apple Silicon, all packages must be **arm64 native** binaries. If you see `RuntimeWarning: ... arm64 ... x86_64` or cryptography/cffi import errors:

```bash
# Force-reinstall all native arm64 builds
.venv/bin/pip install --upgrade --force-reinstall \
  pydantic-core pydantic cryptography openai anthropic
```

The `pdfplumber` package uses `cryptography` / `cffi` internally. In tests, it cannot be imported directly on some configurations — always mock it at the `sys.modules` level:

```python
# CORRECT — avoids arch-related import errors in test environments
import sys
from unittest.mock import MagicMock, patch

mock_pdfplumber = MagicMock()
with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber}):
    from src.processing.pdf_extractor import PDFExtractor
    # ... test code ...

# WRONG — may fail on arm64 due to binary dependency
with patch("pdfplumber.open", ...):
    ...
```

---

## Handling HuggingFace Model Issues

### `google/t5-small` returns 401

HuggingFace Hub now requires authentication for organisation-namespaced public models. The config already uses `t5-small` (without the `google/` prefix) which works without a token.

If you add a new HF model and get a 401, try removing the `org/` prefix first. If the model genuinely requires auth, set `HUGGINGFACE_API_KEY` in the environment and call:

```python
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_API_KEY"])
```

### `transformers` version pinning

The `summarization` and `text2text-generation` pipeline tasks were removed in transformers v5. `requirements.txt` pins `transformers>=4.41.0,<5.0.0`. Do not upgrade past this constraint without testing all three HF inference components.

### `sentencepiece` missing

Required by `valhalla/t5-base-qg-hl`'s tokenizer. If you see a tokenizer error on first question-generation run:

```bash
.venv/bin/pip install sentencepiece
```

---

## Coverage Targets

The CI gate is **95 % overall coverage**. Current coverage is **98 %**.

Known gaps (not worth closing):

| Module | Lines | Reason |
|---|---|---|
| `chunking.py` | 116-117, 120-123 | NLTK `LookupError` branches — only triggered when NLTK data is missing |
| `chain.py` | 77, 103, 110 | Deeply nested async error paths |
| `vector_store.py` | 103-104, 119 | Chroma metadata filter edge cases |

To see the current gap:

```bash
.venv/bin/python -m pytest tests/unit --cov=src --cov-report=term-missing 2>&1 | grep -E "MISS|TOTAL"
```

---

## Continuous Integration

The project has no CI config yet. A GitHub Actions workflow would look like:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: bash scripts/setup_env.sh
      - run: bash scripts/run_tests.sh
```

---

## Project Conventions Checklist

Before opening a pull request:

- [ ] All new public functions/classes have type annotations
- [ ] `from __future__ import annotations` at the top of every `.py` file
- [ ] Heavy dependencies are lazily imported (not at module level)
- [ ] Configuration uses immutable dataclasses
- [ ] No mutation of function arguments
- [ ] New modules have corresponding tests in `tests/unit/`
- [ ] `bash scripts/run_tests.sh` passes with exit code 0
- [ ] No API keys or secrets in any committed file
