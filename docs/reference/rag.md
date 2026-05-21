# API Reference — `src.rag`

Source: [`src/rag/`](../../src/rag/)  
Cross-references: [Core](core.md) · [Agents](agents.md) · [Knowledge Graph](knowledge_graph.md) · [User Guide → Main Components](../user-guide/main-components.md#srcrag--rag-pipelines)

---

## `basic_rag.py`

### `BasicRAGConfig`

```python
@dataclass
class BasicRAGConfig:
    embedding_model: str  = "all-MiniLM-L6-v2"
    top_k: int            = 3
    chunk_size: int       = 400
    chunk_overlap: int    = 50
    collection_name: str  = "basic_rag"
    persist_dir: str      = "data/vectordb/basic_rag"
```

### `BasicRAGResult`

```python
@dataclass
class BasicRAGResult:
    query: str
    answer: str
    retrieved_chunks: list[str]
    sources: list[str]
```

### `BasicRAGPipeline`

```python
class BasicRAGPipeline:
    def __init__(self, config: BasicRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    def index(self, texts: list[str],
              sources: list[str] | None = None) -> int:
        """Chunk, embed, and index texts. Returns total chunk count."""

    def query(self, question: str) -> BasicRAGResult:
        """Retrieve top-k chunks and generate a grounded answer."""
```

**Example:** [`examples/01_basic_rag.py`](../../examples/01_basic_rag.py) · **Output:** [`01_basic_rag_output.json`](../../examples/output/01_basic_rag_output.json)

---

## `ibm_rag.py`

### `IBMRAGConfig`

```python
@dataclass
class IBMRAGConfig:
    embedding_model: str  = "all-MiniLM-L6-v2"  # or IBM Granite model
    collection_name: str  = "ibm_rag"
    persist_dir: str      = "data/vectordb/ibm_rag"
    max_retries: int      = 3
    retry_delay: float    = 1.0     # seconds (exponential back-off base)
    chunk_size: int       = 400
    chunk_overlap: int    = 50
    top_k: int            = 3
```

### `IBMRAGResult`

```python
@dataclass
class IBMRAGResult:
    query: str
    answer: str
    sources: list[str]
    latency_ms: float
    retries_used: int
```

### `IBMProductionRAG`

```python
class IBMProductionRAG:
    def __init__(self, config: IBMRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    @property
    def is_ready(self) -> bool:
        """True when at least one document has been indexed."""

    @property
    def stats(self) -> dict:
        """{'indexed_sources': int, 'total_chunks': int, 'vector_store': str}"""

    def load_text(self, text: str, source: str = "inline") -> int:
        """Chunk and index plain text. Returns chunk count."""

    def load_pdf(self, path: str | Path) -> int:
        """Extract and index a PDF. Returns chunk count."""

    def query(self, question: str) -> IBMRAGResult:
        """Retrieve + generate with retry and latency tracking."""
```

**Example:** [`examples/02_ibm_production_rag.py`](../../examples/02_ibm_production_rag.py) · **Output:** [`02_ibm_production_rag_output.json`](../../examples/output/02_ibm_production_rag_output.json)

---

## `graph_rag.py`

### `GraphRAGConfig`

```python
@dataclass
class GraphRAGConfig:
    max_hops: int   = 2      # Maximum DFS depth during retrieval
    chunk_size: int = 1000   # Characters per text chunk for triple extraction
```

### `GraphRAGResult`

```python
@dataclass
class GraphRAGResult:
    query: str
    answer: str
    graph_context: str   # Formatted triple paths used as context
    triples_used: int    # Number of triples in context
```

### `GraphRAGPipeline`

```python
class GraphRAGPipeline:
    def __init__(self, config: GraphRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    @property
    def _graph(self) -> KnowledgeGraphStore:
        """Access the underlying graph store."""

    def build_graph(self, texts: list[str],
                    sources: list[str] | None = None) -> int:
        """Extract triples from texts and populate the graph. Returns triple count."""

    def query(self, question: str) -> GraphRAGResult:
        """DFS traversal from query-matching nodes → LLM generation."""
```

**Dependencies:** [`KnowledgeGraphStore`](knowledge_graph.md#knowledgegraphstore) · [`TripleExtractor`](knowledge_graph.md#tripleextractor)  
**Example:** [`examples/03_graph_rag.py`](../../examples/03_graph_rag.py) · **Output:** [`03_graph_rag_output.json`](../../examples/output/03_graph_rag_output.json)

---

## `multi_doc_rag.py`

### `MultiDocConfig`

```python
@dataclass
class MultiDocConfig:
    collection_name: str = "multi_doc_rag"
    persist_dir: str     = "data/vectordb/multi_doc_rag"
    chunk_size: int      = 500
    chunk_overlap: int   = 50
    top_k: int           = 3
    embedding_model: str = "all-MiniLM-L6-v2"
```

### `MultiDocResult`

```python
@dataclass
class MultiDocResult:
    query: str
    answer: str
    retrieved_chunks: list[str]
    source_documents: list[str]   # Unique source filenames
```

### `MultiDocumentRAG`

```python
class MultiDocumentRAG:
    def __init__(self, config: MultiDocConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    @property
    def document_count(self) -> int:
        """Number of distinct source documents loaded."""

    def load_text(self, text: str, source: str = "inline") -> int: ...
    def load_file(self, path: str | Path) -> int: ...
    def load_directory(self, directory: str | Path,
                       extensions: list[str] | None = None) -> int:
        """Load all .txt / .md / .pdf files in a directory."""

    def query(self, question: str) -> MultiDocResult: ...
```

**Example:** [`examples/04_multi_doc_rag.py`](../../examples/04_multi_doc_rag.py) · **Output:** [`04_multi_doc_rag_output.json`](../../examples/output/04_multi_doc_rag_output.json)

---

## `agentic_rag.py`

### `IntentType`

```python
class IntentType(str, Enum):
    SEARCH = "search"   # Factual → retrieve + generate
    DIRECT = "direct"   # Conversational → answer directly
```

### `AgenticRAGConfig`

```python
@dataclass
class AgenticRAGConfig:
    collection_name: str = "agentic_rag"
    persist_dir: str     = "data/vectordb/agentic_rag"
    top_k: int           = 3
    embedding_model: str = "all-MiniLM-L6-v2"
```

### `AgenticRAGResult`

```python
@dataclass
class AgenticRAGResult:
    query: str
    intent: IntentType
    answer: str
    retrieved_chunks: list[str]   # Empty when intent == DIRECT
```

### `AgenticRAGPipeline`

```python
class AgenticRAGPipeline:
    def __init__(self, config: AgenticRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    def index(self, texts: list[str],
              sources: list[str] | None = None) -> int: ...

    def query(self, question: str) -> AgenticRAGResult:
        """Classify intent, then route to retrieval or direct generation."""
```

**Example:** [`examples/05_agentic_rag.py`](../../examples/05_agentic_rag.py) · **Output:** [`05_agentic_rag_output.json`](../../examples/output/05_agentic_rag_output.json)

---

## `realtime_rag.py`

### `RealtimeRAGConfig`

```python
@dataclass
class RealtimeRAGConfig:
    num_search_results: int = 5
    top_k: int              = 3
    chunk_size: int         = 400
    scrape_timeout: int     = 10    # HTTP timeout per URL (seconds)
    embedding_model: str    = "all-MiniLM-L6-v2"
```

### `RealtimeRAGResult`

```python
@dataclass
class RealtimeRAGResult:
    query: str
    answer: str
    search_urls: list[str]
    retrieved_passages: list[str]
```

### `RealtimeRAGAssistant`

```python
class RealtimeRAGAssistant:
    def __init__(self, config: RealtimeRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    def query(self, question: str) -> RealtimeRAGResult:
        """DuckDuckGo search → scrape → rank → generate."""

    # Internal helpers (overrideable for testing)
    def _web_search(self, query: str) -> list[str]: ...
    def _scrape_and_chunk(self, urls: list[str]) -> tuple[list[str], list[str]]: ...
    def _retrieve_top_k(self, query: str, passages: list[str]) -> list[str]: ...
    def _generate(self, prompt: str) -> str: ...
    def _split(self, text: str) -> list[str]: ...
```

**Security:** queries truncated to 1000 chars; URLs validated against SSRF blocklist;
prompts use `=== BEGIN/END SOURCES ===` delimiters.

**Example:** [`examples/06_realtime_rag.py`](../../examples/06_realtime_rag.py) · **Output:** [`06_realtime_rag_output.json`](../../examples/output/06_realtime_rag_output.json)

---

## `multimodal_rag.py`

### `MultimodalRAGConfig`

```python
@dataclass
class MultimodalRAGConfig:
    embedding_model: str      = "all-MiniLM-L6-v2"
    vision_model: str | None  = None   # Bedrock multimodal model ID
    generation_model: str     = "google/flan-t5-base"
    top_k: int                = 5
    chunk_size: int           = 800
    chunk_overlap: int        = 100
    persist_dir: str          = "data/vectordb/multimodal_rag"
    image_prompt: str         = "If the image contains text or data, describe it in detail."
```

### `MultimodalRAGResult`

```python
@dataclass
class MultimodalRAGResult:
    query: str
    answer: str
    retrieved_items: list[dict]   # Each item: {type, content, source, page}
```

### `MultimodalRAGPipeline`

```python
class MultimodalRAGPipeline:
    def __init__(self, config: MultimodalRAGConfig | None = None,
                 llm: BaseLLM | None = None) -> None: ...

    def load_pdf(self, path: str | Path) -> dict:
        """Extract text chunks + caption images. Returns {'text': int, 'images': int}."""

    def query(self, question: str) -> MultimodalRAGResult:
        """Retrieve from text + image index, generate grounded answer."""
```

**Example:** [`examples/08_multimodal_rag.py`](../../examples/08_multimodal_rag.py) · **Output:** [`08_multimodal_rag_output.json`](../../examples/output/08_multimodal_rag_output.json)

---

## `embedder.py`

### `Embedder`

Internal helper. Wraps `sentence_transformers.SentenceTransformer` for consistent
embedding throughout the RAG pipelines.

```python
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...
```

---

## `indexer.py`

### `Indexer`

Internal helper. Wraps FAISS `IndexFlatL2` for the pipelines that use
in-process FAISS (BasicRAG, MultimodalRAG).

```python
class Indexer:
    def __init__(self, dim: int) -> None: ...
    def add(self, embeddings: list[list[float]],
            texts: list[str]) -> None: ...
    def search(self, query_embedding: list[float],
               k: int = 3) -> list[str]: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...
```
