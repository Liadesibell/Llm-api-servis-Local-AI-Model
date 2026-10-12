from __future__ import annotations

import os
from pathlib import Path

from ink_ai.conversations import default_database_path
from ink_ai.memory.rag import IngestionResult, KnowledgeIngestor
from ink_ai.memory.vector_store import SearchResult, SQLiteKnowledgeIndex
from ink_ai.web_research import LocalWebResearch, ResearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class KnowledgeManager:
    def __init__(
        self,
        knowledge_dir: Path | None = None,
        database_path: Path | None = None,
    ) -> None:
        configured_dir = knowledge_dir or Path(
            os.environ.get("INK_AI_KNOWLEDGE_DIR", PROJECT_ROOT / "knowledge" / "incoming")
        )
        configured_database = database_path or Path(
            os.environ.get(
                "INK_AI_KNOWLEDGE_DB",
                str(default_database_path().parent / "knowledge.sqlite3"),
            )
        )
        self.index = SQLiteKnowledgeIndex(configured_database)
        self.ingestor = KnowledgeIngestor(configured_dir, self.index)
        self.web_research = LocalWebResearch(self.index)

    @property
    def knowledge_dir(self) -> Path:
        return self.ingestor.knowledge_dir

    def ingest(self) -> IngestionResult:
        return self.ingestor.ingest()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.index.search(query, limit)

    def stats(self) -> dict[str, int]:
        return self.index.stats()

    def add_text(self, name: str, text: str) -> str:
        return self.ingestor.add_text(name, text)

    def research_web(self, query: str, limit: int = 5) -> list[ResearchResult]:
        return self.web_research.search_and_index(query, limit)
