"""Tokenization utilities: count tokens, truncate text."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Approximate characters-per-token ratio used as a fallback when tiktoken is unavailable
_CHARS_PER_TOKEN = 4


class Tokenizer:
    """Wraps tiktoken (for OpenAI models) or falls back to a simple character estimator.

    Args:
        model: Model name used to select the correct BPE encoding (e.g. ``"gpt-4o"``).
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoding = None
        self._init_tiktoken()

    def _init_tiktoken(self) -> None:
        try:
            import tiktoken  # type: ignore[import]

            self._encoding = tiktoken.encoding_for_model(self._model)
            logger.debug("Tiktoken encoding loaded for model '%s'", self._model)
        except Exception:  # noqa: BLE001
            logger.warning(
                "tiktoken not available or model '%s' unknown — using character estimator.",
                self._model,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to at most *max_tokens* tokens.

        Returns the truncated string (decoded back to text when tiktoken is available).
        """
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoding.decode(tokens[:max_tokens])

        # Fallback: character-based truncation
        max_chars = max_tokens * _CHARS_PER_TOKEN
        return text[:max_chars]

    def fits_in_context(self, text: str, max_tokens: int) -> bool:
        """Return True if *text* fits within *max_tokens*."""
        return self.count_tokens(text) <= max_tokens

    def split_by_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """Split *text* into parts each fitting within *max_tokens*."""
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            parts = []
            for i in range(0, len(tokens), max_tokens):
                parts.append(self._encoding.decode(tokens[i : i + max_tokens]))
            return parts

        # Fallback
        max_chars = max_tokens * _CHARS_PER_TOKEN
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
