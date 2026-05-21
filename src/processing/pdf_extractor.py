"""PDF text extraction utilities using pdfplumber."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts plain text from PDF files, with optional disk caching.

    Args:
        cache_dir: Directory for caching extracted text. Pass ``None`` to
                   disable caching.
    """

    def __init__(self, cache_dir: str | Path | None = "data/cache") -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, pdf_path: str | Path) -> str:
        """Extract and return all text from *pdf_path* as a single string.

        Results are cached on disk (keyed by the file's SHA-256 hash) so
        repeated calls on the same file incur no re-extraction cost.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            ImportError: If ``pdfplumber`` is not installed.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # ── Cache lookup ──────────────────────────────────────────────────────
        cache_key = self._cache_key(path)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Loaded extracted text from cache (%s)", cache_key)
            return cached

        # ── Extraction ────────────────────────────────────────────────────────
        text = self._extract_with_pdfplumber(path)
        self._save_cache(cache_key, text)
        return text

    def extract_pages(self, pdf_path: str | Path) -> list[str]:
        """Return a list of per-page text strings."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return self._extract_pages_with_pdfplumber(path)

    def extract_to_file(
        self, pdf_path: str | Path, output_path: str | Path | None = None
    ) -> Path:
        """Extract text and save it to *output_path* (or ``data/cache/<stem>.txt``).

        Returns the path to the saved text file.
        """
        text = self.extract(pdf_path)
        stem = Path(pdf_path).stem
        out = Path(output_path) if output_path else Path("data/cache") / f"{stem}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("Extracted text saved to %s", out)
        return out

    def preview(self, pdf_path: str | Path, chars: int = 500) -> str:
        """Return the first *chars* characters of the extracted text."""
        return self.extract(pdf_path)[:chars]

    # ── pdfplumber backend ────────────────────────────────────────────────────

    @staticmethod
    def _extract_with_pdfplumber(path: Path) -> str:
        try:
            import pdfplumber  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install with: pip install pdfplumber"
            ) from exc

        extracted = ""
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted += page_text + "\n"

        logger.info("Extracted %d characters from %s (%d pages)", len(extracted), path.name, len(pdf.pages) if False else 0)
        return extracted.strip()

    @staticmethod
    def _extract_pages_with_pdfplumber(path: Path) -> list[str]:
        try:
            import pdfplumber  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pdfplumber is required. pip install pdfplumber") from exc

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text.strip())
        logger.debug("Extracted %d pages from %s", len(pages), path.name)
        return pages

    # ── Cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(path: Path) -> str:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return f"{path.stem}_{sha}"

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.txt"

    def _load_cache(self, key: str) -> str | None:
        cp = self._cache_path(key)
        if cp and cp.exists():
            return cp.read_text(encoding="utf-8")
        return None

    def _save_cache(self, key: str, text: str) -> None:
        cp = self._cache_path(key)
        if cp:
            cp.write_text(text, encoding="utf-8")
            logger.debug("Cached extracted text to %s", cp)
