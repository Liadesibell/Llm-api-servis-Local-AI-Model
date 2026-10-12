from ink_ai.memory.manager import KnowledgeManager
from ink_ai.memory.rag import split_into_chunks


def test_split_into_chunks_respects_size_and_overlap():
    text = "first paragraph\n" + ("python docker deployment " * 100)

    chunks = split_into_chunks(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1_200 for chunk in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_ingestion_indexes_supported_files_and_removes_deleted_sources(tmp_path):
    incoming = tmp_path / "knowledge"
    incoming.mkdir()
    document = incoming / "docker.md"
    document.write_text(
        "# Docker deployment\nUse docker compose to run the application.",
        encoding="utf-8",
    )
    manager = KnowledgeManager(incoming, tmp_path / "index.sqlite3")

    result = manager.ingest()

    assert result["indexed"] == 1
    assert manager.search("docker compose deployment")[0].source == "docker.md"
    document.unlink()
    result = manager.ingest()
    assert result["documents"] == 0
    assert manager.search("docker compose deployment") == []


def test_ingestion_reports_unsupported_or_unreadable_documents(tmp_path):
    incoming = tmp_path / "knowledge"
    incoming.mkdir()
    (incoming / "archive.zip").write_bytes(b"not supported")
    (incoming / "empty.md").write_text(" \n", encoding="utf-8")
    manager = KnowledgeManager(incoming, tmp_path / "index.sqlite3")

    result = manager.ingest()

    assert result["indexed"] == 0
    assert result["skipped"][0]["file"] == "empty.md"


def test_bm25_search_returns_relevant_chunks_first(tmp_path):
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "index.sqlite3")
    manager.add_text("unrelated.md", "Bananas and apples are fruits.")
    manager.add_text(
        "docker.md",
        "Docker Compose defines services and networks for container deployments.",
    )

    results = manager.search("Docker Compose container services")

    assert results
    assert results[0].source.endswith("docker.md")


def test_filesystem_reindex_preserves_text_added_through_api(tmp_path):
    manager = KnowledgeManager(
        tmp_path / "incoming",
        tmp_path / "index.sqlite3",
    )
    manager.add_text("api-guide.md", "GitHub Actions runs automated CI workflows.")

    manager.ingest()

    assert manager.search("GitHub Actions automated workflows")


def test_knowledge_database_connection_closes_after_operations(tmp_path):
    database_path = tmp_path / "index.sqlite3"
    manager = KnowledgeManager(tmp_path / "incoming", database_path)
    manager.add_text("guide.md", "Docker Compose starts services.")
    manager.search("Docker Compose")

    renamed_path = tmp_path / "renamed.sqlite3"
    database_path.rename(renamed_path)

    assert renamed_path.is_file()
