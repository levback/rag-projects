# API Reference — `src.knowledge_graph`

Source: [`src/knowledge_graph/`](../../src/knowledge_graph/)  
Cross-references: [Core](core.md) · [RAG → GraphRAGPipeline](rag.md#graphragpipeline) · [User Guide → Learn Basics](../user-guide/learn-basics.md#knowledge-graphs)

---

## `graph_store.py`

### `Triple`

```python
@dataclass
class Triple:
    head: str       # Subject entity
    relation: str   # Predicate / edge label
    tail: str       # Object entity
    source: str = ""  # Document source this triple came from
```

Immutable value object. Stored as a directed edge in the NetworkX graph.

---

### `GraphSearchResult`

```python
@dataclass
class GraphSearchResult:
    triples: list[Triple]      # All triples found in traversal
    paths: list[list[str]]     # Node paths from seed to leaf
    context_text: str          # Human-readable triple summary
```

Returned by `KnowledgeGraphStore.search()` and `get_context()`.

---

### `KnowledgeGraphStore`

```python
class KnowledgeGraphStore:
    def __init__(self) -> None: ...
```

NetworkX `DiGraph`-backed triple store with DFS retrieval.

#### Properties

```python
@property
def node_count(self) -> int:
    """Number of unique entities in the graph."""

@property
def edge_count(self) -> int:
    """Number of triple edges."""

@property
def triples(self) -> list[Triple]:
    """All triples as a flat list (immutable copy)."""
```

#### Mutating methods

```python
def add_triple(self, triple: Triple) -> None:
    """Add a directed edge head → tail with relation as edge label."""

def add_triples(self, triples: list[Triple]) -> None:
    """Bulk add triples."""
```

#### Retrieval methods

```python
def search(
    self,
    query: str,
    max_hops: int = 2,
    top_k: int = 10,
) -> GraphSearchResult:
    """
    Find nodes matching query terms, then DFS up to max_hops.
    Returns all reachable triples and traversed paths.
    """

def get_context(self, query: str,
                max_hops: int = 2) -> str:
    """Convenience wrapper: returns context_text from search()."""
```

#### Persistence methods

```python
def save(self, path: str | Path) -> None:
    """Serialise graph to JSON (node/edge list)."""

def load(self, path: str | Path) -> None:
    """Load graph from a previously saved JSON file."""
```

**Example DFS traversal:**

```python
store = KnowledgeGraphStore()
store.add_triple(Triple("Alan Turing", "studied at", "King's College Cambridge"))
store.add_triple(Triple("King's College Cambridge", "part of", "University of Cambridge"))
result = store.search("Alan Turing", max_hops=2)
# result.paths = [["Alan Turing", "King's College Cambridge", "University of Cambridge"]]
# result.context_text includes both triples
```

---

## `triple_extractor.py`

### `ExtractionResult`

```python
@dataclass
class ExtractionResult:
    triples: list[Triple]
    raw_response: str   # LLM output before parsing
    chunk_text: str     # Input text chunk
```

### `TripleExtractor`

```python
class TripleExtractor:
    def __init__(self, llm: BaseLLM) -> None: ...

    def extract(self, text: str,
                source: str = "") -> list[Triple]:
        """
        Split text into chunks, call extract_chunk() per chunk,
        deduplicate, return flat list of triples.
        """

    def extract_chunk(self, text: str,
                      source: str = "") -> ExtractionResult:
        """
        Prompt LLM to extract (head, relation, tail) triples from one chunk.
        Returns raw LLM response and parsed triples.
        """

    def _parse_response(self, response: str,
                        source: str = "") -> list[Triple]:
        """
        Parse LLM output.  Accepts formats:
          - JSON array:  [{"head": ..., "relation": ..., "tail": ...}]
          - Pipe-delimited: head | relation | tail
          - Fallback:  skip unparseable lines
        """
```

**Used by:** [`GraphRAGPipeline.build_graph()`](rag.md#graphragpipeline)  
**Example:** [`examples/03_graph_rag.py`](../../examples/03_graph_rag.py)
