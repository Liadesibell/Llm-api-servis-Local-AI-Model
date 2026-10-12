from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

from ink_ai.documents import SUPPORTED_EXTENSIONS, read_document
from ink_ai.memory.vector_store import SQLiteKnowledgeIndex

CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 180
MAX_KNOWLEDGE_FILE_SIZE = 10 * 1024 * 1024


class SkippedDocument(TypedDict):
    file: str
    error: str


class IngestionResult(TypedDict):
    indexed: int
    skipped: list[SkippedDocument]
    documents: int
    chunks: int


def split_into_chunks(text: str) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n", start + CHUNK_SIZE // 2, end)
            if boundary == -1:
                boundary = normalized.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        content = normalized[start:end].strip()
        if content:
            chunks.append(content)
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


class KnowledgeIngestor:
    def __init__(
        self,
        knowledge_dir: Path,
        index: SQLiteKnowledgeIndex,
    ) -> None:
        self.knowledge_dir = knowledge_dir.expanduser().resolve()
        self.index = index

    def ingest(self) -> IngestionResult:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        indexed = 0
        skipped: list[SkippedDocument] = []
        seen_sources: set[str] = set()
        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.is_symlink():
                skipped.append({"file": path.name, "error": "Sembolik bağlantı atlandı."})
                continue
            relative_path = path.relative_to(self.knowledge_dir).as_posix()
            seen_sources.add(relative_path)
            try:
                if path.stat().st_size > MAX_KNOWLEDGE_FILE_SIZE:
                    raise ValueError("Bilgi dosyası 10 MB sınırını aşıyor.")
                content = read_document(
                    path,
                    max_chars=MAX_KNOWLEDGE_FILE_SIZE,
                )
                chunks = split_into_chunks(content)
                if not chunks:
                    raise ValueError("Dosyada indekslenecek metin bulunamadı.")
                self.index.replace_source(relative_path, chunks)
                indexed += 1
            except (OSError, ValueError, ImportError) as error:
                skipped.append({"file": relative_path, "error": str(error)})

        self._remove_deleted_sources(seen_sources)
        stats = self.index.stats()
        return {
            "indexed": indexed,
            "skipped": skipped,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
        }

    def _remove_deleted_sources(self, seen_sources: set[str]) -> None:
        for source in self.index.sources() - seen_sources:
            if source.startswith("api/"):
                continue
            self.index.replace_source(source, [])

    def add_text(self, name: str, text: str) -> str:
        chunks = split_into_chunks(text)
        if not chunks:
            raise ValueError("İndekslenecek metin boş olamaz.")
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        source = f"api/{digest}/{Path(name).name}"
        self.index.replace_source(source, chunks)
        return source
