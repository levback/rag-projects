# API Reference — `src.loaders`

Source: [`src/loaders/`](../../src/loaders/)  
Cross-references: [Processing](processing.md) · [Agents](agents.md) · [User Guide → Learn Basics](../user-guide/learn-basics.md)

---

## `document_loader.py`

### `LoadedDocument`

```python
@dataclass
class LoadedDocument:
    text: str            # Extracted plain text
    source: str          # Original path or URL
    metadata: dict       # File metadata (size, page count, mime type…)
    content_hash: str    # SHA-256 hex digest of text (for dedup / caching)
```

`content_hash` is computed from `text` at construction time.

---

### `DocumentLoader`

```python
class DocumentLoader:
    def __init__(self) -> None: ...

    def load(self, source: str | Path) -> LoadedDocument:
        """
        Auto-detect source type (path extension or URL scheme)
        and dispatch to load_pdf(), load_text(), or load_url().
        """

    def load_pdf(self, path: str | Path) -> LoadedDocument:
        """
        Extract text from a PDF using pdfplumber.
        Security: resolves path with Path.resolve() before I/O.
        Raises: FileNotFoundError, ValueError (not a PDF)
        """

    def load_text(self, path: str | Path) -> LoadedDocument:
        """
        Read a plain-text file (any encoding via chardet detection).
        Security: resolves path with Path.resolve() before I/O.
        Raises: FileNotFoundError
        """

    def load_url(self, url: str) -> LoadedDocument:
        """
        Fetch a URL via WebScraper and wrap as LoadedDocument.
        Security: URL validated against SSRF blocklist before fetch.
        Raises: ValueError (SSRF block), requests.HTTPError
        """

    def load_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
    ) -> list[LoadedDocument]:
        """
        Load all files in a directory matching extensions
        (default: ['.txt', '.md', '.pdf']).
        Skips unreadable files with a warning.
        Returns list sorted by filename.
        """
```

**Security notes:**

| Threat | Mitigation |
|--------|-----------|
| Path traversal | `Path(path).resolve()` before any file open |
| SSRF | URL passed to `WebScraper._validate_url()` first |

**Used by:** `MultiDocumentRAG.load_directory()` · `LangChainRAGAgent.load_pdf()` · `DocumentAnalysisPipeline.run()`

---

## `web_scraper.py`

### `ScrapedPage`

```python
@dataclass
class ScrapedPage:
    url: str
    text: str          # Cleaned plain text (HTML stripped)
    title: str         # <title> tag content or ""
    status_code: int   # HTTP response status
```

---

### `WebScraper`

```python
class WebScraper:
    # SSRF protection lists (module-level constants)
    _BLOCKED_HOSTS: frozenset[str]  = frozenset({
        "localhost", "0.0.0.0", "::1", "127.0.0.1", "metadata.google.internal", ...
    })
    _BLOCKED_PREFIXES: tuple[str, ...] = (
        "10.", "192.168.", "172.16.", "172.17.", ... "172.31.",
    )

    def __init__(
        self,
        timeout: int = 10,
        rate_limit_delay: float = 0.5,
        headers: dict | None = None,
    ) -> None: ...

    def fetch(self, url: str) -> ScrapedPage:
        """Single URL fetch. Raises ValueError on SSRF block."""

    def fetch_page(self, url: str) -> ScrapedPage:
        """Alias for fetch() (backwards compatibility)."""

    def fetch_many(
        self,
        urls: list[str],
        skip_errors: bool = True,
    ) -> list[ScrapedPage]:
        """
        Fetch multiple URLs sequentially with rate-limit delay.
        If skip_errors=True (default), failed URLs are omitted.
        """

    def _validate_url(self, url: str) -> None:
        """
        Raises ValueError if URL resolves to a blocked host or prefix.
        Checks both raw hostname and DNS-resolved address.
        """

    def _extract_text(self, html: str) -> str:
        """BeautifulSoup HTML → plain text with <br>/<p> normalisation."""

    def _respect_rate_limit(self) -> None:
        """Sleep rate_limit_delay seconds since last request."""
```

**SSRF blocklist includes:** loopback addresses, RFC-1918 private ranges,
cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`).

**Used by:** `DocumentLoader.load_url()` · `RealtimeRAGAssistant._scrape_and_chunk()` · `ResearchAgent._scrape()`
