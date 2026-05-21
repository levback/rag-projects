"""Tests for src/loaders/document_loader.py and src/loaders/web_scraper.py"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_pdf_extractor_mock(content: str = "PDF text content") -> MagicMock:
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = content
    return mock_extractor


# ─────────────────────────────────────────────────────────────────────────────
# WebScraper tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWebScraper:
    def _make_scraper(self, **kwargs):
        from src.loaders.web_scraper import WebScraper
        return WebScraper(rate_limit_seconds=0, **kwargs)

    def _mock_response(self, text: str = "<html><body><p>" + "Hello world content. " * 10 + "</p></body></html>",
                       status_code: int = 200, content_type: str = "text/html"):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.headers = {"content-type": content_type}
        return resp

    def test_fetch_returns_text(self):
        scraper = self._make_scraper()
        with patch("requests.get", return_value=self._mock_response()):
            text = scraper.fetch("https://example.com")
        assert "Hello world" in text

    def test_fetch_page_returns_scraped_page(self):
        from src.loaders.web_scraper import ScrapedPage
        scraper = self._make_scraper()
        with patch("requests.get", return_value=self._mock_response()):
            page = scraper.fetch_page("https://example.com")
        assert isinstance(page, ScrapedPage)
        assert page.status_code == 200
        assert page.url == "https://example.com"

    def test_fetch_page_non_html_returns_empty(self):
        scraper = self._make_scraper()
        with patch("requests.get", return_value=self._mock_response(
            content_type="application/pdf"
        )):
            page = scraper.fetch_page("https://example.com/doc.pdf")
        assert page.text == ""

    def test_fetch_page_non_200_returns_empty(self):
        scraper = self._make_scraper()
        with patch("requests.get", return_value=self._mock_response(status_code=404)):
            page = scraper.fetch_page("https://example.com")
        assert page.text == ""

    def test_fetch_page_request_exception_returns_empty(self):
        import requests
        scraper = self._make_scraper()
        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            page = scraper.fetch_page("https://example.com")
        assert page.text == ""
        assert page.status_code == 0

    def test_validate_url_invalid_scheme_raises(self):
        scraper = self._make_scraper()
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            scraper.fetch("ftp://example.com")

    def test_validate_url_no_host_raises(self):
        scraper = self._make_scraper()
        with pytest.raises(ValueError, match="no host"):
            scraper.fetch("https://")

    def test_fetch_many_returns_only_successful(self):
        scraper = self._make_scraper()
        good_text = "<html><body><p>" + "Good content with lots of text here. " * 10 + "</p></body></html>"
        good_resp = self._mock_response(text=good_text)
        bad_resp = self._mock_response(status_code=404)
        with patch("requests.get", side_effect=[good_resp, bad_resp]):
            pages = scraper.fetch_many(["https://a.com", "https://b.com"])
        assert len(pages) == 1

    def test_noise_tags_removed(self):
        html = ("<html><body>" + "<p>Content text here. " * 10 + "</p>"
                "<script>alert(1)</script><nav>Nav bar navigation</nav></body></html>")
        scraper = self._make_scraper()
        with patch("requests.get", return_value=self._mock_response(text=html)):
            text = scraper.fetch("https://example.com")
        assert "alert" not in text
        assert "Content" in text

    def test_short_text_returns_empty(self):
        scraper = self._make_scraper(min_text_length=1000)
        with patch("requests.get", return_value=self._mock_response(text="<p>Short</p>")):
            text = scraper.fetch("https://example.com")
        assert text == ""


# ─────────────────────────────────────────────────────────────────────────────
# DocumentLoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDocumentLoader:
    def _make_loader(self):
        from src.loaders.document_loader import DocumentLoader
        return DocumentLoader()

    def test_load_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello from a text file", encoding="utf-8")
        loader = self._make_loader()
        doc = loader.load_text(f)
        assert doc.content == "Hello from a text file"
        assert doc.doc_type == "text"
        assert doc.source == str(f)

    def test_load_text_missing_raises(self):
        loader = self._make_loader()
        with pytest.raises(FileNotFoundError):
            loader.load_text("/nonexistent/file.txt")

    def test_load_pdf_calls_extractor(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")  # minimal bytes

        mock_extractor = _make_pdf_extractor_mock("Extracted PDF text")
        with patch("src.processing.pdf_extractor.PDFExtractor", return_value=mock_extractor):
            loader = self._make_loader()
            doc = loader.load_pdf(f)

        assert doc.content == "Extracted PDF text"
        assert doc.doc_type == "pdf"

    def test_load_url_calls_scraper(self):
        mock_scraper = MagicMock()
        mock_scraper.fetch.return_value = "Web page content"
        with patch("src.loaders.web_scraper.WebScraper", return_value=mock_scraper):
            loader = self._make_loader()
            doc = loader.load_url("https://example.com")

        assert doc.content == "Web page content"
        assert doc.doc_type == "web"
        assert doc.metadata["url"] == "https://example.com"

    def test_load_auto_detect_url(self):
        mock_scraper = MagicMock()
        mock_scraper.fetch.return_value = "Web content"
        with patch("src.loaders.web_scraper.WebScraper", return_value=mock_scraper):
            loader = self._make_loader()
            doc = loader.load("https://example.com")
        assert doc.doc_type == "web"

    def test_load_auto_detect_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Notes content", encoding="utf-8")
        loader = self._make_loader()
        doc = loader.load(str(f))
        assert doc.doc_type == "text"

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Text A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Text B", encoding="utf-8")
        (tmp_path / "c.jpg").write_bytes(b"image")  # should be skipped

        loader = self._make_loader()
        docs = loader.load_directory(tmp_path, extensions=[".txt"])
        assert len(docs) == 2
        contents = {d.content for d in docs}
        assert "Text A" in contents
        assert "Text B" in contents

    def test_load_directory_not_a_dir_raises(self):
        loader = self._make_loader()
        with pytest.raises(NotADirectoryError):
            loader.load_directory("/nonexistent/dir/")

    def test_loaded_document_content_hash(self, tmp_path):
        from src.loaders.document_loader import LoadedDocument
        doc = LoadedDocument(content="hello", source="test.txt")
        assert len(doc.content_hash) == 16

    def test_load_directory_skips_failed_files(self, tmp_path):
        (tmp_path / "good.txt").write_text("OK", encoding="utf-8")
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"bad")

        loader = self._make_loader()
        # Patch PDFExtractor to fail, loader should skip and return the txt
        with patch("src.processing.pdf_extractor.PDFExtractor", side_effect=Exception("bad")):
            docs = loader.load_directory(tmp_path)
        # at least the txt was loaded
        assert any(d.content == "OK" for d in docs)
