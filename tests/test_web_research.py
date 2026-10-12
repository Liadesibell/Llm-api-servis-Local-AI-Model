import pytest

from ink_ai.memory.manager import KnowledgeManager
from ink_ai.web_research import (
    _classify_discovered_link,
    _ReadableHTML,
    validate_public_https_url,
)


def test_web_research_collects_search_results_and_indexes_page_text(
    tmp_path, monkeypatch
):
    searched = {}

    class FakeDDGS:
        def text(self, query, *, max_results, safesearch):
            searched.update(
                query=query,
                max_results=max_results,
                safesearch=safesearch,
            )
            return [
                {
                    "title": "Docker deployment",
                    "href": "https://docs.example.test/docker",
                    "body": "Docker search snippet.",
                }
            ]

    monkeypatch.setattr("ink_ai.web_research.DDGS", FakeDDGS)
    monkeypatch.setattr(
        "ink_ai.web_research.validate_public_https_url",
        lambda url: url,
    )
    monkeypatch.setattr(
        "ink_ai.web_research._fetch_page",
        lambda _url: (
            "Docker Compose defines services and networks for an application.",
            [
                {
                    "url": "https://docs.example.test/files/compose.zip",
                    "label": "Download Compose ZIP",
                },
                {
                    "url": "https://docs.example.test/tree/main/examples",
                    "label": "Examples",
                },
            ],
        ),
    )
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "index.sqlite3")

    sources = manager.research_web("Docker Compose", limit=2)

    assert searched == {
        "query": "Docker Compose",
        "max_results": 2,
        "safesearch": "moderate",
    }
    assert sources[0]["url"] == "https://docs.example.test/docker"
    assert sources[0]["content_source"] == "page"
    assert "Compose defines services" in sources[0]["snippet"]
    assert sources[0]["links"] == [
        {
            "type": "download",
            "label": "Download Compose ZIP",
            "url": "https://docs.example.test/files/compose.zip",
        },
        {
            "type": "folder",
            "label": "Examples",
            "url": "https://docs.example.test/tree/main/examples",
        },
    ]
    assert manager.search("Docker Compose services")[0].source == sources[0]["url"]


def test_html_parser_extracts_public_download_and_folder_link_candidates():
    parser = _ReadableHTML()
    parser.feed(
        '<p>Docs</p><a href="/downloads/tool.zip">Download tool</a>'
        '<a href="/tree/main/examples">Examples</a>'
        '<script><a href="/bad.zip">bad</a></script>'
    )

    assert parser.links == [
        {"href": "/downloads/tool.zip", "label": "Download tool"},
        {"href": "/tree/main/examples", "label": "Examples"},
    ]
    assert _classify_discovered_link(
        "https://docs.example.test/downloads/tool.zip",
        "Download tool",
    ) == "download"
    assert _classify_discovered_link(
        "https://docs.example.test/tree/main/examples",
        "Examples",
    ) == "folder"


def test_direct_search_result_download_url_is_returned(tmp_path, monkeypatch):
    class FakeDDGS:
        def text(self, _query, *, max_results, safesearch):
            return [
                {
                    "title": "Download package",
                    "href": "https://downloads.example.test/package.zip",
                    "body": "Package release.",
                }
            ]

    monkeypatch.setattr("ink_ai.web_research.DDGS", FakeDDGS)
    monkeypatch.setattr(
        "ink_ai.web_research.validate_public_https_url",
        lambda url: url,
    )
    monkeypatch.setattr("ink_ai.web_research._fetch_page", lambda _url: ("", []))
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "index.sqlite3")

    sources = manager.research_web("download package", limit=1)

    assert sources[0]["links"] == [
        {
            "type": "download",
            "label": "Download package",
            "url": "https://downloads.example.test/package.zip",
        }
    ]


def test_duckduckgo_search_does_not_require_credentials(tmp_path, monkeypatch):
    called = {}

    class FakeDDGS:
        def text(self, query, *, max_results, safesearch):
            called.update(
                query=query,
                max_results=max_results,
                safesearch=safesearch,
            )
            return []

    monkeypatch.setattr("ink_ai.web_research.DDGS", FakeDDGS)
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "index.sqlite3")

    assert manager.research_web("DuckDuckGo search") == []
    assert called["query"] == "DuckDuckGo search"


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.example.test/page",
        "https://localhost/private",
        "https://user:password@docs.example.test/page",
        "https://docs.example.test:8443/page",
    ],
)
def test_web_research_rejects_insecure_or_private_urls(url):
    with pytest.raises(ValueError):
        validate_public_https_url(url)
