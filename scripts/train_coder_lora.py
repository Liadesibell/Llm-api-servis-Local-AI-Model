from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from ink_ai.training_data import TrainingExample, validate_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "training" / "datasets" / "coder.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "ink-ai-coder-lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQUENCE_LENGTH = 1_024


def load_examples(dataset_path: Path) -> list[TrainingExample]:
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Eğitim verisi bulunamadı: {dataset_path}. Önce "
            "scripts/prepare_training_data.py ile JSONL dosyasını doğrula."
        )
    with dataset_path.open(encoding="utf-8") as dataset_file:
        examples = [
            validate_example(json.loads(line), line_number)
            for line_number, line in enumerate(dataset_file, start=1)
            if line.strip()
        ]
    if len(examples) < 10:
        raise ValueError("LoRA eğitimi için en az 10 doğrulanmış örnek gerekli.")
    return examples


def train(dataset_path: Path, output_dir: Path, base_model: str) -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            "8 GB VRAM için QLoRA eğitimi Linux/WSL2 üzerinde desteklenir. "
            "Windows'tan WSL2 Ubuntu'ya geçip proje ve veri kümesini orada kullan."
        )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "Eğitim bağımlılıkları kurulu değil. WSL2 içinde "
            "`uv sync --group training` çalıştır."
        ) from error

    examples = load_examples(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    prepared_rows: list[dict[str, list[int]]] = []
    for example in examples:
        messages = example["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )["input_ids"]
        encoded = tokenizer(
            full_conversation,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) <= len(prompt_ids):
            continue
        labels = input_ids.copy()
        labels[: min(len(prompt_ids), len(labels))] = [
            -100
        ] * min(len(prompt_ids), len(labels))
        prepared_rows.append({"input_ids": input_ids, "labels": labels})

    if len(prepared_rows) < 10:
        raise ValueError(
            "Token sınırı sonrasında 10'dan az yanıt örneği kaldı; "
            "daha kısa veya daha çok örnek kullan."
        )
    dataset = Dataset.from_list(prepared_rows)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(os.environ.get("INK_AI_TRAIN_EPOCHS", "3")),
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print(
        f"LoRA adaptörü kaydedildi: {output_dir}. "
        "Bu adaptör mevcut GGUF dosyasını değiştirmez."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Coder modeli için 4-bit QLoRA eğit.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-model",
        default=os.environ.get("INK_AI_TRAIN_BASE_MODEL", DEFAULT_BASE_MODEL),
    )
    arguments = parser.parse_args()
    train(arguments.dataset, arguments.output, arguments.base_model)


if __name__ == "__main__":
    main()
