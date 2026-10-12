from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import ClassVar, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ink_ai.memory.rag import split_into_chunks
from ink_ai.memory.vector_store import SQLiteKnowledgeIndex

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_PAGE_CHARS = 24_000
MAX_RESULTS = 5
USER_AGENT = "InkAI-LocalResearch/1.0 (user-requested documentation search)"
DOWNLOAD_EXTENSIONS = {
    ".7z", ".apk", ".csv", ".doc", ".docx", ".exe", ".gz", ".ipynb",
    ".jar", ".json", ".md", ".pdf", ".ppt", ".pptx", ".py", ".rar",
    ".sql", ".tar", ".txt", ".whl", ".zip",
}
FOLDER_PATH_MARKERS = {"/directory/", "/directories/", "/folder/", "/folders/", "/tree/"}


class ResearchLink(TypedDict):
    type: str
    label: str
    url: str


class ResearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    content_source: str
    links: list[ResearchLink]


def validate_public_https_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Yalnızca kimlik bilgisi içermeyen HTTPS kaynakları açılır.")
    host = parsed.hostname.rstrip(".")
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError) as error:
        raise ValueError(f"Kaynak sunucu çözümlenemedi: {host}") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Yerel veya özel ağ kaynaklarına erişim reddedildi.")
    return url


class _PublicHTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ReadableHTML(HTMLParser):
    _ignored_tags: ClassVar[set[str]] = {
        "script", "style", "noscript", "svg", "nav", "footer", "header",
        "aside", "form", "button", "iframe",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self._anchor_text: list[str] | None = None
        self._anchor_href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        elif tag == "a":
            self._anchor_text = []
            self._anchor_href = next(
                (value or "" for name, value in attrs if name == "href"),
                "",
            )
        elif tag in {"p", "div", "br", "li", "pre", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "a" and self._anchor_text is not None:
            label = " ".join(" ".join(self._anchor_text).split())[:300]
            if self._anchor_href and len(self.links) < 100:
                self.links.append({"href": self._anchor_href, "label": label})
            self._anchor_text = None
            self._anchor_href = ""
        elif tag in {"p", "div", "br", "li", "pre", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            normalized = " ".join(data.split())
            if normalized:
                self.parts.append(normalized)
                if self._anchor_text is not None:
                    self._anchor_text.append(normalized)


def _fetch_page(url: str) -> tuple[str, list[dict[str, str]]]:
    safe_url = validate_public_https_url(url)
    opener = build_opener(_PublicHTTPSRedirectHandler())
    request = Request(
        safe_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf",
        },
    )
    try:
        with opener.open(request, timeout=12) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {
                "text/html",
                "application/xhtml+xml",
                "application/pdf",
            }:
                return "", []
            content = response.read(MAX_PAGE_BYTES + 1)
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Kaynak indirilemedi ({safe_url}): {error}") from error
    if len(content) > MAX_PAGE_BYTES:
        raise ValueError("Kaynak sayfa 2 MB sınırını aşıyor.")
    if content_type == "application/pdf":
        from pypdf import PdfReader

        text = "\n".join(
            page.extract_text() or ""
            for page in PdfReader(BytesIO(content)).pages[:40]
        )
        links: list[dict[str, str]] = []
    else:
        parser = _ReadableHTML()
        parser.feed(content.decode("utf-8", errors="replace"))
        text = "\n".join(parser.parts)
        links = [
            {
                "url": urljoin(final_url, item["href"]),
                "label": item["label"],
            }
            for item in parser.links
            if item["href"]
        ]
    return text.strip()[:MAX_PAGE_CHARS], links


def extract_page_text(url: str) -> str:
    return _fetch_page(url)[0]


def _classify_discovered_link(url: str, label: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    path = parsed.path.casefold()
    extension = Path(path).suffix
    normalized_label = label.casefold()
    if extension in DOWNLOAD_EXTENSIONS or any(
        term in normalized_label
        for term in ("download", "indir", "indirilen dosya", "yükle")
    ):
        return "download"
    if any(marker in path for marker in FOLDER_PATH_MARKERS) or any(
        term in normalized_label for term in ("folder", "directory", "klasör", "dizin")
    ):
        return "folder"
    return None


class LocalWebResearch:
    def __init__(self, index: SQLiteKnowledgeIndex) -> None:
        self.index = index

    def search_and_index(
        self,
        query: str,
        limit: int = MAX_RESULTS,
    ) -> list[ResearchResult]:
        normalized_query = " ".join(query.split())
        if len(normalized_query) < 2 or len(normalized_query) > 500:
            raise ValueError("Arama sorgusu 2-500 karakter arasında olmalı.")
        if not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"Kaynak sayısı 1-{MAX_RESULTS} arasında olmalı.")
        try:
            search_results = DDGS().text(
                normalized_query,
                max_results=limit,
                safesearch="moderate",
            )
        except DDGSException as error:
            raise RuntimeError(f"DuckDuckGo araması başarısız: {error}") from error

        sources: list[ResearchResult] = []
        seen_urls: set[str] = set()
        indexed_sources: set[str] = set()
        for result in search_results:
            url = str(result.get("href") or result.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                validate_public_https_url(url)
            except ValueError:
                continue
            try:
                page_text, page_links = _fetch_page(url)
            except RuntimeError:
                page_text = ""
                page_links = []
            except ValueError:
                continue
            snippet = str(result.get("body") or result.get("snippet") or "").strip()
            content = page_text or snippet
            if not content:
                continue
            title = str(result.get("title") or url).strip()[:300]
            discovered_links: list[ResearchLink] = []
            seen_links: set[str] = set()
            result_link_type = _classify_discovered_link(url, title)
            if result_link_type:
                discovered_links.append(
                    {"type": result_link_type, "label": title or url, "url": url}
                )
                seen_links.add(url)
            for link in page_links:
                link_url = link["url"]
                link_type = _classify_discovered_link(link_url, link["label"])
                if not link_type or link_url in seen_links:
                    continue
                seen_links.add(link_url)
                try:
                    validate_public_https_url(link_url)
                except ValueError:
                    continue
                discovered_links.append(
                    {
                        "type": link_type,
                        "label": link["label"] or link_url,
                        "url": link_url,
                    }
                )
                if len(discovered_links) >= 10:
                    break
            link_context = ""
            if discovered_links:
                link_context = "\n\nBulunan klasör/indirme bağlantıları:\n" + "\n".join(
                    f"- [{link['type']}] {link['label']}: {link['url']}"
                    for link in discovered_links
                )
            indexed_text = (
                f"Title: {title}\nURL: {url}\n"
                f"Extracted from the source page; summarize, do not reproduce verbatim.\n"
                f"{content}{link_context}"
            )
            self.index.replace_source(url, split_into_chunks(indexed_text))
            indexed_sources.add(url)
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": content[:1_000],
                    "content_source": "page" if page_text else "search-result snippet",
                    "links": discovered_links,
                }
            )
        for match in self.index.search(normalized_query, limit=limit):
            if match.source not in indexed_sources:
                continue
            for source in sources:
                if source["url"] == match.source:
                    source["snippet"] = match.content[:2_000]
                    break
        return sources
