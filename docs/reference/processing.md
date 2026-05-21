# API Reference — `src.processing`

Source: [`src/processing/`](../../src/processing/)  
Cross-references: [Inference](inference.md) · [RAG](rag.md) · [User Guide → Learn Basics](../user-guide/learn-basics.md#chunking)

---

## `chunking.py`

### `ChunkingConfig`

```python
@dataclass
class ChunkingConfig:
    chunk_size: int    = 400    # Maximum characters per chunk
    chunk_overlap: int = 50     # Character overlap between consecutive chunks
    min_chunk_size: int = 20    # Chunks shorter than this are discarded
```

---

### `TextChunker`

Character-boundary splitter with configurable overlap.

```python
class TextChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None: ...

    def split(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        Splits prefer sentence boundaries ('. ') within the window
        when available.
        """

    def split_many(self, texts: list[str]) -> list[str]:
        """Apply split() to each text, return concatenated chunk list."""
```

---

### `SentenceChunkingConfig`

```python
@dataclass
class SentenceChunkingConfig:
    max_sentences: int = 5      # Sentences per chunk
    overlap_sentences: int = 1  # Sentence overlap between chunks
```

---

### `SentenceChunker`

NLTK `sent_tokenize`-based splitter. Groups sentences rather than characters.

```python
class SentenceChunker:
    def __init__(
        self,
        config: SentenceChunkingConfig | None = None,
    ) -> None: ...

    def split(self, text: str) -> list[str]: ...
    def split_many(self, texts: list[str]) -> list[str]: ...
```

---

## `pdf_extractor.py`

### `PDFExtractor`

pdfplumber-backed PDF text extractor with an in-memory page cache.

```python
class PDFExtractor:
    def __init__(self) -> None: ...

    def extract(self, path: str | Path) -> str:
        """
        Extract all pages and join with double newline.
        Result is cached by file path for the lifetime of the object.
        """

    def extract_pages(self, path: str | Path) -> list[str]:
        """Return per-page text as a list."""

    def extract_to_file(
        self,
        path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Write extracted text to a .txt file.
        Default output: <pdf_stem>.txt in the same directory.
        """

    def preview(self, path: str | Path,
                chars: int = 500) -> str:
        """Return first `chars` characters of extracted text."""
```

---

## `preprocessing.py`

### Module-level functions

```python
def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines to single spaces."""

def remove_control_characters(text: str) -> str:
    """Strip ASCII control chars (except \\n and \\t)."""

def strip_html_tags(text: str) -> str:
    """Remove HTML/XML tags, preserving inner text."""

def remove_urls(text: str) -> str:
    """Delete http/https/ftp URLs from text."""

def clean_text(text: str) -> str:
    """
    Full pipeline: remove_control_characters → strip_html_tags
                   → remove_urls → normalize_whitespace
    """
```

### `TextPreprocessor`

```python
class TextPreprocessor:
    def __init__(
        self,
        normalise_whitespace: bool = True,
        remove_html: bool = True,
        remove_urls: bool = False,
        remove_control_chars: bool = True,
        lowercase: bool = False,
    ) -> None: ...

    def process(self, text: str) -> str:
        """Apply configured cleaning steps in order."""

    def process_many(self, texts: list[str]) -> list[str]:
        """Apply process() to a list."""
```

---

## `tokenizer.py`

### `Tokenizer`

tiktoken-backed token counter. Used to stay within LLM context limits.

```python
class Tokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None: ...

    def count_tokens(self, text: str) -> int:
        """Return tiktoken token count for text."""

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to at most max_tokens tokens."""
```

Supported encodings: `cl100k_base` (GPT-4 / Claude), `p50k_base` (GPT-3.5).
