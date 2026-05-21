# API Reference — `src.agents`

Source: [`src/agents/`](../../src/agents/)  
Cross-references: [Core](core.md) · [RAG](rag.md) · [Loaders](loaders.md) · [User Guide → Main Components](../user-guide/main-components.md#srcagents--agent-layer)

---

## `base_agent.py`

### `AgentResult`

```python
@dataclass
class AgentResult:
    query: str
    answer: str
    steps: list[str]   # Log of internal reasoning steps
    metadata: dict
```

Base result class inherited by all concrete agent results.

### `BaseAgent`

```python
class BaseAgent(ABC):
    def __init__(self, llm: BaseLLM) -> None: ...

    @abstractmethod
    def run(self, query: str) -> AgentResult: ...

    def _log_step(self, step: str) -> None:
        """Append to internal step log and call logger.debug()."""
```

---

## `research_agent.py`

### `ResearchAgentConfig`

```python
@dataclass
class ResearchAgentConfig:
    num_search_results: int = 5      # Max URLs from DuckDuckGo
    top_k_passages: int     = 3      # Passages selected after ranking
    chunk_size: int         = 400    # Characters per scraped chunk
    scrape_timeout: int     = 10     # HTTP timeout per page (seconds)
    embedding_model: str    = "all-MiniLM-L6-v2"
```

### `ResearchResult`

```python
@dataclass
class ResearchResult(AgentResult):
    synthesis: str           # Final synthesised answer
    sources: list[str]       # URLs used
    passage_count: int       # Passages used in synthesis prompt
```

### `ResearchAgent`

```python
class ResearchAgent(BaseAgent):
    def __init__(
        self,
        config: ResearchAgentConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None: ...

    def run(self, query: str) -> ResearchResult:
        """Full research loop: search → scrape → rank → synthesise."""

    # Internal helpers (overrideable in tests)
    def _web_search(self, query: str) -> list[str]: ...
    def _scrape(self, urls: list[str]) -> list[str]: ...
    def _rank(self, query: str,
              passages: list[str]) -> list[str]: ...
    def _synthesize(self, query: str,
                    passages: list[str]) -> str: ...
    def _split(self, text: str) -> list[str]: ...
```

**Security:** prompt injection protection via `=== BEGIN/END SOURCES ===` delimiters.

**Example:** [`examples/07_research_agent.py`](../../examples/07_research_agent.py) · **Output:** [`07_research_agent_output.json`](../../examples/output/07_research_agent_output.json)

---

## `langchain_rag_agent.py`

### `LangChainRAGConfig`

```python
@dataclass
class LangChainRAGConfig:
    llm_provider: str         = "bedrock"   # "openai" | "anthropic" | "bedrock" | "huggingface"
    mode: str                 = "chain"     # "chain" | "agent"
    top_k: int                = 3
    chunk_size: int           = 500
    chunk_overlap: int        = 50
    collection_name: str      = "langchain_rag"
    persist_dir: str          = "data/vectordb/langchain_rag"
    embedding_model: str      = "all-MiniLM-L6-v2"
    agent_max_iterations: int = 3
```

### `LangChainRAGResult`

```python
@dataclass
class LangChainRAGResult:
    query: str
    answer: str
    sources: list[str]
    mode_used: str              # "chain" | "agent"
    retrieved_docs: list[str]   # Chunk texts
    steps: list[str]            # Agent thought steps (empty for chain)
```

### `LangChainRAGAgent`

```python
class LangChainRAGAgent:
    def __init__(
        self,
        config: LangChainRAGConfig | None = None,
        llm: BaseLLM | None = None,
    ) -> None: ...

    # Document loading
    def load_text(self, text: str, source: str = "inline") -> int:
        """Chunk and index plain text. Returns chunk count."""

    def load_pdf(self, path: str | Path) -> int:
        """Extract PDF and index. Returns chunk count."""

    def load_url(self, url: str) -> int:
        """Scrape a URL and index the content. Returns chunk count."""

    def run(self, query: str) -> LangChainRAGResult:
        """Retrieve and generate using chain or agent mode."""

    # Internal methods
    def _run_chain(self, query: str,
                   docs: list) -> LangChainRAGResult: ...

    def _run_agent(self, query: str,
                   docs: list) -> LangChainRAGResult: ...

    def _get_lc_llm(self) -> Any:
        """Build LangChain-compatible LLM from llm_provider setting."""

    def _ensure_store(self) -> None:
        """Lazily initialise Chroma vector store on first call."""

    def _split(self, text: str) -> list[Any]:
        """Apply chunk_size / chunk_overlap splitting."""
```

**LLM provider routing table:**

| `llm_provider` | LangChain class | Env var |
|---------------|-----------------|---------|
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `anthropic` | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| `bedrock` | `ChatBedrockConverse` | AWS credentials |
| `huggingface` | `HuggingFacePipeline` | None |

**Example:** [`examples/09_langchain_rag.py`](../../examples/09_langchain_rag.py) · **Output:** [`09_langchain_rag_output.json`](../../examples/output/09_langchain_rag_output.json)
