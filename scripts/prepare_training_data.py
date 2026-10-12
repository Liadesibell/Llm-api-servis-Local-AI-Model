from __future__ import annotations

import argparse
from pathlib import Path

from ink_ai.training_data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ink AI Coder için JSONL sohbet veri kümesini doğrula."
    )
    parser.add_argument("source", type=Path, help="Hazırladığın kaynak .jsonl dosyası")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/datasets/coder.jsonl"),
    )
    arguments = parser.parse_args()
    count = prepare_dataset(arguments.source, arguments.output)
    print(f"{count} örnek doğrulandı ve {arguments.output} konumuna kaydedildi.")


if __name__ == "__main__":
    main()
