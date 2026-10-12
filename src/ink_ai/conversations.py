from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def default_database_path() -> Path:
    data_dir = Path(
        os.environ.get("INK_AI_DATA_DIR", str(Path.home() / ".ink-ai"))
    ).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "conversations.sqlite3"


class ConversationStore:
    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or default_database_path()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS conversation_files (
                    conversation_id INTEGER NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, name)
                );
                CREATE TABLE IF NOT EXISTS conversation_images (
                    conversation_id INTEGER NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    content_base64 TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, name)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_conversation(self, title: str = "Yeni sohbet") -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (title) VALUES (?)", (title,)
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Sohbet kaydı oluşturuldu ancak kimliği alınamadı.")
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Mesaj rolü user veya assistant olmalı.")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )

    def set_title(self, conversation_id: int, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (title, conversation_id),
            )

    def list_conversations(self) -> list[tuple[int, str]]:
        return self.search_conversations("")

    def search_conversations(self, query: str) -> list[tuple[int, str]]:
        with self._connect() as connection:
            if not query.strip():
                rows = connection.execute(
                    "SELECT id, title FROM conversations "
                    "ORDER BY updated_at DESC, id DESC"
                )
            else:
                needle = f"%{query.strip()}%"
                rows = connection.execute(
                    "SELECT DISTINCT c.id, c.title FROM conversations c "
                    "LEFT JOIN messages m ON m.conversation_id = c.id "
                    "WHERE c.title LIKE ? OR m.content LIKE ? "
                    "ORDER BY c.updated_at DESC, c.id DESC",
                    (needle, needle),
                )
            return [
                (int(row[0]), str(row[1]))
                for row in rows
            ]

    def get_messages(self, conversation_id: int) -> list[tuple[str, str]]:
        with self._connect() as connection:
            return [
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = ? ORDER BY id",
                    (conversation_id,),
                )
            ]

    def set_files(
        self, conversation_id: int, files: list[dict[str, str]]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_files WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.executemany(
                "INSERT INTO conversation_files (conversation_id, name, content) "
                "VALUES (?, ?, ?)",
                [
                    (conversation_id, file["name"], file["content"])
                    for file in files
                ],
            )

    def get_files(self, conversation_id: int) -> list[dict[str, str]]:
        with self._connect() as connection:
            return [
                {"name": str(row[0]), "content": str(row[1])}
                for row in connection.execute(
                    "SELECT name, content FROM conversation_files "
                    "WHERE conversation_id = ? ORDER BY name",
                    (conversation_id,),
                )
            ]

    def set_images(
        self, conversation_id: int, images: list[dict[str, str]]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_images WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.executemany(
                "INSERT INTO conversation_images "
                "(conversation_id, name, content_base64) VALUES (?, ?, ?)",
                [
                    (conversation_id, image["name"], image["content_base64"])
                    for image in images
                ],
            )

    def get_images(self, conversation_id: int) -> list[dict[str, str]]:
        with self._connect() as connection:
            return [
                {"name": str(row[0]), "content_base64": str(row[1])}
                for row in connection.execute(
                    "SELECT name, content_base64 FROM conversation_images "
                    "WHERE conversation_id = ? ORDER BY name",
                    (conversation_id,),
                )
            ]
