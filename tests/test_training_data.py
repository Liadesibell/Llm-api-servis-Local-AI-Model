import json

import pytest

from ink_ai.training_data import prepare_dataset


def test_prepare_training_data_validates_and_normalizes_jsonl(tmp_path):
    source = tmp_path / "examples.jsonl"
    destination = tmp_path / "prepared" / "coder.jsonl"
    example = {
        "messages": [
            {"role": "user", "content": "  Fix this test  "},
            {"role": "assistant", "content": "  Add an assertion.  "},
        ]
    }
    source.write_text(
        "".join(json.dumps(example) + "\n" for _ in range(10)),
        encoding="utf-8",
    )

    assert prepare_dataset(source, destination) == 10
    prepared = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])
    assert prepared["messages"][0]["content"] == "Fix this test"


def test_prepare_training_data_rejects_too_few_examples(tmp_path):
    source = tmp_path / "examples.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="En az 10"):
        prepare_dataset(source, tmp_path / "prepared.jsonl")
