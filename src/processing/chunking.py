"""Text splitting / chunking utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Split on sentence boundaries before hard-splitting by character count
    sentence_split: bool = True


class TextChunker:
    """Splits text into overlapping chunks suitable for embedding.

    Strategy:
    1. Optionally split on sentence boundaries (``"."``/``"!"``/``"?"``) first.
    2. Accumulate sentences into chunks of at most *chunk_size* characters.
    3. Add *chunk_overlap* characters of the previous chunk as a prefix for context.
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._cfg = config or ChunkingConfig()

    def split(self, text: str) -> list[str]:
        """Split *text* into chunks. Returns an empty list for empty input."""
        if not text or not text.strip():
            return []

        if self._cfg.sentence_split:
            return self._sentence_aware_split(text)
        return self._character_split(text)

    # ── Sentence-aware split ──────────────────────────────────────────────────

    def _sentence_aware_split(self, text: str) -> list[str]:
        sentences = self._SENTENCE_RE.split(text.strip())
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= self._cfg.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Overlap: take tail of last chunk
                overlap = self._tail(current, self._cfg.chunk_overlap)
                current = (overlap + " " + sentence).strip() if overlap else sentence

                # Handle a single sentence longer than chunk_size
                if len(current) > self._cfg.chunk_size:
                    chunks.extend(self._character_split(current))
                    current = ""

        if current:
            chunks.append(current)

        return chunks

    # ── Character split ───────────────────────────────────────────────────────

    def _character_split(self, text: str) -> list[str]:
        step = self._cfg.chunk_size - self._cfg.chunk_overlap
        if step <= 0:
            step = self._cfg.chunk_size
        return [
            text[i : i + self._cfg.chunk_size]
            for i in range(0, len(text), step)
        ]

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _tail(text: str, n: int) -> str:
        return text[-n:] if n > 0 and len(text) > n else ""


@dataclass
class SentenceChunkingConfig:
    """Configuration for the NLTK sentence-based chunker."""

    word_limit: int = 200
    """Maximum words per passage before starting a new one."""
    require_nltk_punkt: bool = True
    """Automatically download the NLTK ``punkt`` tokenizer if missing."""


class SentenceChunker:
    """Splits text into passages using NLTK sentence tokenisation.

    This matches the document analysis article's approach:
    sentences are grouped into passages until the word limit is reached,
    then a new passage begins.

    Requires: ``pip install nltk``
    """

    def __init__(self, config: SentenceChunkingConfig | None = None) -> None:
        self._cfg = config or SentenceChunkingConfig()
        self._ensure_nltk()

    def _ensure_nltk(self) -> None:
        try:
            import nltk  # type: ignore[import]

            if self._cfg.require_nltk_punkt:
                try:
                    nltk.data.find("tokenizers/punkt_tab")
                except LookupError:
                    nltk.download("punkt_tab", quiet=True)
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt", quiet=True)
        except ImportError as exc:
            raise ImportError(
                "nltk is required for SentenceChunker. "
                "Install with: pip install nltk"
            ) from exc

    def split(self, text: str) -> list[str]:
        """Split *text* into word-limited sentence passages.

        Returns an empty list for empty/whitespace-only input.
        """
        if not text or not text.strip():
            return []

        from nltk.tokenize import sent_tokenize  # type: ignore[import]

        sentences = sent_tokenize(text)
        passages: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            word_count = len(candidate.split())
            if word_count < self._cfg.word_limit:
                current = candidate
            else:
                if current:
                    passages.append(current.strip())
                current = sentence

        if current:
            passages.append(current.strip())

        return passages

    def split_with_metadata(self, text: str) -> list[dict]:
        """Return passages with index and word-count metadata."""
        return [
            {"index": i, "text": p, "word_count": len(p.split())}
            for i, p in enumerate(self.split(text))
        ]
