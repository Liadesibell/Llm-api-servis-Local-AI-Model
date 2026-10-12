from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

ALLOWED_ROLES = {"system", "user", "assistant"}


class TrainingMessage(TypedDict):
    role: str
    content: str


class TrainingExample(TypedDict):
    messages: list[TrainingMessage]


def validate_example(example: object, line_number: int) -> TrainingExample:
    if not isinstance(example, dict) or not isinstance(example.get("messages"), list):
        raise TypeError(f"{line_number}. satır: messages listesi gerekli.")
    messages = example["messages"]
    if len(messages) < 2:
        raise ValueError(f"{line_number}. satır: user ve assistant örnekleri gerekli.")
    normalized: list[TrainingMessage] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"{line_number}. satır: mesaj nesne olmalı.")
        role, content = message.get("role"), message.get("content")
        if role not in ALLOWED_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{line_number}. satır: geçersiz rol veya boş mesaj içeriği."
            )
        normalized.append({"role": role, "content": content.strip()})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError(f"{line_number}. satır: en az bir user mesajı gerekli.")
    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{line_number}. satır: son mesaj assistant olmalı.")
    return {"messages": normalized}


def prepare_dataset(source: Path, destination: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"Eğitim veri kümesi bulunamadı: {source}")
    examples: list[TrainingExample] = []
    with source.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{line_number}. satır geçerli JSON değil: {error}"
                ) from error
            examples.append(validate_example(value, line_number))
    if len(examples) < 10:
        raise ValueError(
            f"En az 10 kaliteli örnek gerekli; {len(examples)} örnek bulundu. "
            "İlk deneme için 100+, gerçekçi davranış için yüzlerce temiz örnek hedefle."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for example in examples:
            dataset_file.write(
                json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return len(examples)
