"""Unit tests for preprocessing utilities and TextPreprocessor."""
from __future__ import annotations

import pytest

from src.processing.preprocessing import (
    TextPreprocessor,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
    remove_html_tags,
    remove_urls,
    truncate_repeated_chars,
)


# ── Standalone functions ──────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_collapses_newlines_to_space(self):
        assert normalize_whitespace("a\n\nb") == "a b"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""


class TestRemoveControlCharacters:
    def test_removes_null_byte(self):
        result = remove_control_characters("hello\x00world")
        assert "\x00" not in result

    def test_preserves_newline(self):
        result = remove_control_characters("line1\nline2")
        assert "\n" in result

    def test_preserves_tab(self):
        result = remove_control_characters("col1\tcol2")
        assert "\t" in result

    def test_normal_text_unchanged(self):
        text = "The quick brown fox"
        assert remove_control_characters(text) == text


class TestNormalizeUnicode:
    def test_nfc_normalization(self):
        # é as combining character (e + combining accent) → precomposed
        composed = "\u00e9"         # é
        decomposed = "e\u0301"      # e + combining grave
        result = normalize_unicode(decomposed, form="NFC")
        assert result == composed

    def test_nfd_normalization(self):
        composed = "\u00e9"
        result = normalize_unicode(composed, form="NFD")
        assert len(result) == 2  # decomposed into base + combining mark

    def test_plain_ascii_unchanged(self):
        text = "Hello World"
        assert normalize_unicode(text) == text


class TestRemoveUrls:
    def test_removes_http_url(self):
        result = remove_urls("Visit http://example.com for info.")
        assert "http://example.com" not in result
        assert "Visit" in result

    def test_removes_https_url(self):
        result = remove_urls("Check https://secure.example.com/path?q=1")
        assert "https://" not in result

    def test_no_url_unchanged(self):
        text = "No links here."
        assert remove_urls(text) == text

    def test_multiple_urls_removed(self):
        text = "See http://a.com and http://b.com"
        result = remove_urls(text)
        assert "http://" not in result


class TestRemoveHtmlTags:
    def test_removes_simple_tag(self):
        result = remove_html_tags("<b>bold</b>")
        assert result == "bold"

    def test_removes_nested_tags(self):
        result = remove_html_tags("<div><p>Text</p></div>")
        assert result == "Text"

    def test_removes_self_closing_tag(self):
        result = remove_html_tags("Line one.<br/>Line two.")
        assert "<br/>" not in result

    def test_plain_text_unchanged(self):
        text = "No tags here."
        assert remove_html_tags(text) == text


class TestTruncateRepeatedChars:
    def test_truncates_long_run(self):
        result = truncate_repeated_chars("heeeeeeeello", max_repeat=3)
        assert result == "heeello"

    def test_short_run_unchanged(self):
        result = truncate_repeated_chars("hello", max_repeat=3)
        assert result == "hello"

    def test_exactly_at_limit_unchanged(self):
        result = truncate_repeated_chars("aaa", max_repeat=3)
        assert result == "aaa"

    def test_exceeds_limit_truncated(self):
        result = truncate_repeated_chars("aaaa", max_repeat=3)
        assert result == "aaa"


# ── TextPreprocessor ──────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def test_default_normalizes_whitespace(self):
        pp = TextPreprocessor()
        result = pp.process("  hello   world  ")
        assert result == "hello world"

    def test_remove_html_option(self):
        pp = TextPreprocessor(remove_html=True)
        result = pp.process("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_remove_urls_option(self):
        pp = TextPreprocessor(remove_urls=True)
        result = pp.process("Visit http://example.com now")
        assert "http://" not in result

    def test_lowercase_option(self):
        pp = TextPreprocessor(lowercase=True)
        result = pp.process("Hello World")
        assert result == "hello world"

    def test_truncate_repeats_option(self):
        pp = TextPreprocessor(truncate_repeats=True)
        result = pp.process("waaaait")
        assert result.count("a") <= 3

    def test_empty_string(self):
        pp = TextPreprocessor()
        assert pp.process("") == ""

    def test_all_steps_disabled_returns_input(self):
        pp = TextPreprocessor(
            normalize_unicode=False,
            remove_control_chars=False,
            normalize_ws=False,
        )
        text = "Hello  World"
        assert pp.process(text) == text

    def test_multiple_steps_chained(self):
        pp = TextPreprocessor(remove_html=True, lowercase=True, normalize_ws=True)
        result = pp.process("  <b>HELLO</b>   WORLD  ")
        assert result == "hello world"

    def test_process_removes_control_chars_by_default(self):
        pp = TextPreprocessor()
        result = pp.process("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result
