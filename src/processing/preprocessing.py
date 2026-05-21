"""Data cleaning and normalisation utilities."""
from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    """Remove non-printable control characters while preserving newlines and tabs."""
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalisation (NFC by default)."""
    return unicodedata.normalize(form, text)


def remove_urls(text: str) -> str:
    """Strip http/https URLs from *text*."""
    return re.sub(r"https?://\S+", "", text).strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML/XML tags from *text*."""
    return re.sub(r"<[^>]+>", "", text)


def truncate_repeated_chars(text: str, max_repeat: int = 3) -> str:
    """Collapse runs of the same character longer than *max_repeat*.

    E.g. ``"heeeeello"`` → ``"heeello"`` for ``max_repeat=3``.
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.sub(r"\1" * max_repeat, text)


class TextPreprocessor:
    """Configurable pipeline of text cleaning steps.

    Steps are applied in order. Disable any step by passing ``False`` to the
    corresponding constructor argument.

    Example::

        pp = TextPreprocessor(remove_urls=True, remove_html=True)
        clean = pp.process(raw_text)
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_control_chars: bool = True,
        remove_html: bool = False,
        remove_urls: bool = False,
        normalize_ws: bool = True,
        truncate_repeats: bool = False,
        lowercase: bool = False,
    ) -> None:
        self._steps: list = []

        if normalize_unicode:
            self._steps.append(globals()["normalize_unicode"])
        if remove_control_chars:
            self._steps.append(remove_control_characters)
        if remove_html:
            self._steps.append(remove_html_tags)
        if remove_urls:
            self._steps.append(globals()["remove_urls"])
        if truncate_repeats:
            self._steps.append(truncate_repeated_chars)
        if lowercase:
            self._steps.append(str.lower)
        if normalize_ws:
            self._steps.append(normalize_whitespace)

    def process(self, text: str) -> str:
        """Apply all enabled cleaning steps to *text* and return the result."""
        result = text
        for step in self._steps:
            result = step(result)
        return result

    def process_batch(self, texts: list[str]) -> list[str]:
        """Apply cleaning to every element of *texts*."""
        return [self.process(t) for t in texts]
