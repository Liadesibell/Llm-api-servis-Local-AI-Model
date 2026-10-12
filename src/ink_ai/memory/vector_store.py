from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[\w+#.-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class SearchResult:
    source: str
    content: str
    score: float


class SQLiteKnowledgeIndex:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_source "
                "ON knowledge_chunks(source)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def replace_source(self, source: str, chunks: list[str]) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM knowledge_chunks WHERE source = ?", (source,)
            )
            connection.executemany(
                "INSERT INTO knowledge_chunks "
                "(source, chunk_index, content, token_count) VALUES (?, ?, ?, ?)",
                [
                    (source, index, content, len(tokenize(content)))
                    for index, content in enumerate(chunks)
                    if content.strip()
                ],
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM knowledge_chunks")

    def sources(self) -> set[str]:
        with self._connect() as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT source FROM knowledge_chunks"
                )
            }

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        terms = set(tokenize(query))
        if not terms or limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, content, token_count FROM knowledge_chunks"
            ).fetchall()
        if not rows:
            return []

        document_tokens = [tokenize(row[1]) for row in rows]
        document_term_sets = [set(tokens) for tokens in document_tokens]
        document_frequency = {
            term: sum(term in token_set for token_set in document_term_sets)
            for term in terms
        }
        average_length = sum(len(tokens) for tokens in document_tokens) / len(rows)
        results: list[SearchResult] = []
        for row, tokens in zip(rows, document_tokens, strict=True):
            frequencies: dict[str, int] = {}
            for token in tokens:
                if token in terms:
                    frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for term, frequency in frequencies.items():
                inverse_frequency = math.log(
                    1
                    + (len(rows) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                length_normalizer = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1)
                )
                score += inverse_frequency * frequency * 2.2 / length_normalizer
            if score > 0:
                results.append(
                    SearchResult(str(row[0]), str(row[1]), round(score, 6))
                )
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            document_count, chunk_count = connection.execute(
                "SELECT COUNT(DISTINCT source), COUNT(*) FROM knowledge_chunks"
            ).fetchone()
        return {"documents": int(document_count), "chunks": int(chunk_count)}
