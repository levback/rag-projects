# Developer Notes

Notes for contributors, security reviewers, and developers extending the library.

Cross-references: [Main Components](main-components.md) · [API Reference](../reference/index.md) · [security_check.sh](../../scripts/security_check.sh)

---

## Test suite

### Running tests

```bash
# Full suite with coverage
bash scripts/run_tests.sh

# Or directly
source .venv/bin/activate
pytest --cov=src --cov-report=term-missing -q
```

### Coverage target

95%+ is enforced. The current baseline is **398 tests passing, 95% coverage**.

### Test structure

```
tests/
├── unit/
│   ├── test_loaders.py           # DocumentLoader, WebScraper
│   ├── test_knowledge_graph.py   # KnowledgeGraphStore, TripleExtractor
│   ├── test_new_rag_pipelines.py # All 7 RAG pipelines (68 tests)
│   └── test_agents.py            # ResearchAgent, LangChainRAGAgent (36 tests)
└── integration/
    └── test_pipeline.py          # End-to-end smoke tests
```

### Key testing patterns

**Patch at source, not importer** — lazy imports inside method bodies must be
patched at the module they come from:

```python
# WRONG: the name doesn't exist at module level
patch("src.rag.basic_rag.Chroma")

# CORRECT: patch the source module
patch("langchain_community.vectorstores.Chroma")
patch("langchain_huggingface.HuggingFaceEmbeddings")
```

**Never use `pytest.importorskip` for installed packages** — on arm64 macOS
this causes a `SIGABRT`. For installed packages, mock the method directly.

**Async tests** — `pytest.ini` sets `asyncio_mode = strict`. All async tests
need `@pytest.mark.asyncio`.

---

## Security

### Running the security check

```bash
bash scripts/security_check.sh           # check only
bash scripts/security_check.sh --report  # + write JSON reports to security_reports/
bash scripts/security_check.sh --fix     # auto-upgrade vulnerable packages
```

### OWASP Top 10 coverage

| OWASP | Risk | Mitigation in codebase |
|-------|------|----------------------|
| A01 Broken Access Control | Path traversal | `Path.resolve()` in `DocumentLoader` |
| A02 Cryptographic Failures | Weak hash/cipher | Checked by `security_check.sh` |
| A03 Injection | Shell injection | No `shell=True`, no `eval()` |
| A05 Security Misconfiguration | SSRF | `_BLOCKED_HOSTS` in `WebScraper._validate_url()` |
| A06 Vulnerable Components | CVEs in deps | `pip-audit` in CI |
| A07 Auth | Hardcoded credentials | `detect-secrets` scan |
| A08 Software Integrity | Prompt injection | `=== BEGIN/END SOURCES ===` delimiters |
| A09 Logging Failures | No audit trail | `logging` module throughout `src/` |
| A10 SSRF | Internal network access | Blocked prefixes: `10.`, `192.168.`, `172.16-31.` |

### NIST SP 800-218 (SSDF) compliance

| SSDF Practice | Implementation |
|---------------|---------------|
| PW.6 (Test software) | 95%+ test coverage via pytest |
| PW.7 (Review/analyse code) | bandit SAST + `security_check.sh` |
| RV.1 (Identify/fix vulnerabilities) | `pip-audit` for CVE scanning |
| PO.1 (Security requirements) | Input validation at all boundaries |
| PS.1 (Protect code) | No secrets in src/, `.gitignore` covers `.env` |

### Adding a new network-facing feature

1. Add URL validation: call `WebScraper._validate_url()` before any `requests.get()`
2. Truncate user input: `safe_input = input[:1000]`
3. Use source delimiters in prompts if injecting external content
4. Run `bash scripts/security_check.sh` before committing

---

## Adding a new RAG pipeline

1. Create `src/rag/my_pipeline.py`
2. Define a `MyConfig` dataclass with typed fields and defaults
3. Define a `MyResult` dataclass (extends nothing, or extend an existing result)
4. Implement `MyPipeline` with:
   - `__init__(self, config: MyConfig, llm: BaseLLM)`
   - `index(texts, sources)` or `load_*()`
   - `query(question) -> MyResult`
5. Write tests in `tests/unit/test_new_rag_pipelines.py` — patch lazy imports at source
6. Add an example in `examples/NN_my_pipeline.py` with a `_MockLLM` class
7. Add to [Examples](../examples.md) and [Main Components](main-components.md)

---

## Adding a new LLM backend

1. Implement `BaseLLM` in `src/core/my_client.py`:
   - `complete(messages: list[Message]) -> LLMResponse`
   - `stream(messages: list[Message]) -> Iterator[str]`
2. Add a new `ModelProvider` enum value in `model_factory.py`
3. Handle the new provider in `create_llm()`
4. Add unit tests for `complete()` and `stream()` with mocked HTTP responses

---

## Code style

- `from __future__ import annotations` at the top of every file
- `@dataclass` for configs and result objects — no mutation after construction
- Lazy imports inside methods for optional heavy dependencies (HuggingFace, LangChain)
- Functions ≤ 50 lines; files ≤ 800 lines
- No hardcoded values — use config dataclass fields
- `logger = logging.getLogger(__name__)` in every module that logs

---

## CI / pre-commit

```bash
# Lint (if ruff is installed)
ruff check src/ examples/

# Type check (if mypy is installed)
mypy src/ --ignore-missing-imports

# Full pre-commit check
bash scripts/run_tests.sh && bash scripts/security_check.sh
```

---

## Release checklist

- [ ] All tests pass: `pytest -q`
- [ ] Coverage ≥ 95%: `pytest --cov=src`
- [ ] Security clean: `bash scripts/security_check.sh`
- [ ] All 10 examples run: `bash scripts/run_examples.sh`
- [ ] No unpinned deps in `requirements.txt`
- [ ] Docs updated for any new public API
