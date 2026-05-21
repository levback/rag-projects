"""Output formatting and response parsing utilities."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Matches ```json ... ``` or ``` ... ``` fenced code blocks
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
# Matches the first JSON object/array anywhere in a string
_JSON_BARE_RE = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)


@dataclass
class ParsedResponse:
    """Container for a parsed LLM response."""

    raw: str
    text: str = ""
    json_data: Any = None
    parse_error: str | None = None


class ResponseParser:
    """Extracts structured data from raw LLM output strings.

    Common use-cases:
    - Extracting a JSON object embedded in a markdown response.
    - Stripping thinking tags and preamble.
    - Normalising whitespace.
    """

    # ── JSON extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_json(text: str) -> Any:
        """Parse the first JSON object/array from *text*.

        Checks fenced code blocks first, then falls back to bare JSON.

        Raises:
            ValueError: If no valid JSON is found.
        """
        # 1. Try fenced code block
        match = _JSON_FENCE_RE.search(text)
        if match:
            return json.loads(match.group(1))

        # 2. Try bare JSON
        match = _JSON_BARE_RE.search(text)
        if match:
            return json.loads(match.group(1))

        raise ValueError("No JSON object or array found in response.")

    @staticmethod
    def try_extract_json(text: str) -> ParsedResponse:
        """Like :meth:`extract_json` but returns a :class:`ParsedResponse` instead of raising."""
        result = ParsedResponse(raw=text, text=text)
        try:
            result.json_data = ResponseParser.extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            result.parse_error = str(exc)
        return result

    # ── Text cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def strip_thinking_tags(text: str) -> str:
        """Remove ``<think>…</think>`` and ``<thinking>…</thinking>`` tags."""
        cleaned = re.sub(r"<think(?:ing)?>(.*?)</think(?:ing)?>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def strip_markdown_formatting(text: str) -> str:
        """Remove common markdown formatting (bold, italic, headers, inline code)."""
        text = re.sub(r"#{1,6}\s*", "", text)         # headers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)      # italic
        text = re.sub(r"`(.+?)`", r"\1", text)         # inline code
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # code blocks
        return text.strip()

    @staticmethod
    def extract_answer(text: str, answer_prefix: str = "Answer:") -> str:
        """Extract text following *answer_prefix* (case-insensitive).

        Falls back to returning *text* as-is if the prefix is not found.
        """
        pattern = re.compile(re.escape(answer_prefix), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[match.end():].strip()
        return text.strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return re.sub(r"\s+", " ", text).strip()

    # ── Pipeline ──────────────────────────────────────────────────────────────

    @staticmethod
    def clean(text: str, strip_thinking: bool = True, normalize_ws: bool = True) -> str:
        """Convenience pipeline: optionally strip thinking tags and normalise whitespace."""
        if strip_thinking:
            text = ResponseParser.strip_thinking_tags(text)
        if normalize_ws:
            text = ResponseParser.normalize_whitespace(text)
        return text
