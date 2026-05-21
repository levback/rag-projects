"""Web scraping utility — fetch and clean text from HTML pages."""
from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tags whose content is never useful as knowledge
_NOISE_TAGS = frozenset(
    ["script", "style", "noscript", "header", "footer", "nav", "svg", "iframe", "aside"]
)


@dataclass
class ScrapedPage:
    """Result of scraping a single URL."""

    url: str
    text: str
    status_code: int
    content_type: str = "text/html"


class WebScraper:
    """Fetch and clean text from web pages.

    Uses ``requests`` + ``beautifulsoup4``. Both are lazy-imported so
    the class can be imported without those packages installed.

    Args:
        timeout: HTTP request timeout in seconds.
        user_agent: ``User-Agent`` header sent with every request.
        min_text_length: Pages with fewer characters than this are treated
            as empty (e.g., paywalled pages, login redirects).
        rate_limit_seconds: Minimum gap between successive requests (polite crawling).
    """

    def __init__(
        self,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (research-agent/1.0)",
        min_text_length: int = 100,
        rate_limit_seconds: float = 0.5,
    ) -> None:
        self._timeout = timeout
        self._user_agent = user_agent
        self._min_text_length = min_text_length
        self._rate_limit = rate_limit_seconds
        self._last_request_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, url: str) -> str:
        """Fetch *url* and return clean plain text.

        Returns an empty string if the request fails or the page is not HTML.
        """
        result = self.fetch_page(url)
        return result.text

    def fetch_page(self, url: str) -> ScrapedPage:
        """Fetch *url* and return a :class:`ScrapedPage` with full metadata."""
        import requests  # lazy import

        url = self._validate_url(url)
        self._respect_rate_limit()

        headers = {"User-Agent": self._user_agent}
        try:
            response = requests.get(
                url, timeout=self._timeout, headers=headers, allow_redirects=True
            )
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return ScrapedPage(url=url, text="", status_code=0)

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            logger.debug("Skipping non-HTML response from %s (%s)", url, content_type)
            return ScrapedPage(url=url, text="", status_code=response.status_code,
                               content_type=content_type)

        if response.status_code != 200:
            logger.warning("HTTP %d from %s", response.status_code, url)
            return ScrapedPage(url=url, text="", status_code=response.status_code)

        text = self._extract_text(response.text)
        return ScrapedPage(
            url=url,
            text=text,
            status_code=response.status_code,
            content_type=content_type,
        )

    def fetch_many(self, urls: list[str]) -> list[ScrapedPage]:
        """Fetch multiple URLs sequentially with rate limiting."""
        pages = []
        for url in urls:
            page = self.fetch_page(url)
            if page.text:
                pages.append(page)
        return pages

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_text(self, html: str) -> str:
        """Parse HTML and return clean plain text."""
        from bs4 import BeautifulSoup  # lazy import

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < self._min_text_length:
            return ""
        return text

    def _validate_url(self, url: str) -> str:
        """Raise ValueError if url is not a valid http/https URL."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid URL scheme {parsed.scheme!r}. Only http and https are allowed."
            )
        if not parsed.netloc:
            raise ValueError(f"URL has no host: {url!r}")
        # Block internal/private IP ranges to prevent SSRF
        host = parsed.hostname or ""
        _BLOCKED_HOSTS = (
            "localhost",
            "0.0.0.0",
        )
        _BLOCKED_PREFIXES = (
            "127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
            "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", "169.254.",
        )
        if host in _BLOCKED_HOSTS or any(host.startswith(p) for p in _BLOCKED_PREFIXES):
            raise ValueError(f"Blocked internal/private host: {host!r}")
        return url

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()
