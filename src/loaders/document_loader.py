"""Unified document loader — PDF, plain text, and web pages."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """A loaded piece of text with provenance metadata."""

    content: str
    source: str                          # file path or URL
    doc_type: str = "text"               # "pdf" | "text" | "web"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class DocumentLoader:
    """Load documents from PDFs, plain-text files, or web URLs.

    Supported sources:
    - ``"pdf"``  — extracts text via :class:`~src.processing.pdf_extractor.PDFExtractor`
    - ``"text"`` — reads a plain ``.txt`` or ``.md`` file
    - ``"web"``  — fetches and parses an HTML page via :class:`~src.loaders.web_scraper.WebScraper`

    All methods return :class:`LoadedDocument` instances. No mutation.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = cache_dir

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, source: str) -> LoadedDocument:
        """Auto-detect source type and load content.

        Args:
            source: A file path (PDF, txt, md) or a URL (http/https).

        Returns:
            A :class:`LoadedDocument` with extracted text.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self.load_url(source)
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(path)
        return self.load_text(path)

    def load_pdf(self, path: str | Path) -> LoadedDocument:
        """Extract text from a PDF file."""
        from src.processing.pdf_extractor import PDFExtractor

        path = Path(path).resolve()
        extractor = PDFExtractor(cache_dir=self._cache_dir)
        content = extractor.extract(path)
        logger.info("Loaded PDF: %s (%d chars)", path.name, len(content))
        return LoadedDocument(
            content=content,
            source=str(path),
            doc_type="pdf",
            metadata={"filename": path.name, "size_bytes": path.stat().st_size},
        )

    def load_text(self, path: str | Path) -> LoadedDocument:
        """Read a plain-text or markdown file."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text(encoding="utf-8")
        logger.info("Loaded text file: %s (%d chars)", path.name, len(content))
        return LoadedDocument(
            content=content,
            source=str(path),
            doc_type="text",
            metadata={"filename": path.name, "suffix": path.suffix},
        )

    def load_url(self, url: str) -> LoadedDocument:
        """Fetch and extract text from a web URL."""
        from src.loaders.web_scraper import WebScraper

        scraper = WebScraper()
        content = scraper.fetch(url)
        logger.info("Loaded URL: %s (%d chars)", url, len(content))
        return LoadedDocument(
            content=content,
            source=url,
            doc_type="web",
            metadata={"url": url},
        )

    def load_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
    ) -> list[LoadedDocument]:
        """Load all files from *directory* matching the given *extensions*.

        Args:
            directory: Path to a folder.
            extensions: e.g. ``[".pdf", ".txt"]``. Defaults to ``[".pdf", ".txt", ".md"]``.

        Returns:
            List of :class:`LoadedDocument`, skipping files that fail to load.
        """
        extensions = extensions or [".pdf", ".txt", ".md"]
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        docs: list[LoadedDocument] = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in extensions:
                continue
            try:
                docs.append(self.load(str(path)))
            except Exception as exc:
                logger.warning("Skipping %s — %s", path.name, exc)
        logger.info("Loaded %d documents from %s", len(docs), directory)
        return docs
