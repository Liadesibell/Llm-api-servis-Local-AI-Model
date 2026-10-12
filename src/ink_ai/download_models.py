from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ink_ai.brain.image_generation import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_SAFETY_CHECKER_MODEL,
)
from ink_ai.brain.model_router import MODEL_PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DOWNLOADS = {
    "flash": (
        "Qwen/Qwen3-4B-GGUF",
        "Qwen3-4B-Q4_K_M.gguf",
        2_497_280_256,
    ),
    "reasoning": (
        "unsloth/Qwen3.5-35B-A3B-GGUF",
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        22_016_023_168,
    ),
    "coder": (
        "unsloth/Qwen3.5-35B-A3B-GGUF",
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        22_016_023_168,
    ),
}
IMAGE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_MODEL_FILES = [
    "model_index.json",
    "scheduler/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
SAFETY_CHECKER_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SAFETY_CHECKER_FILES = [
    "feature_extractor/**",
    "safety_checker/config.json",
    "safety_checker/model.fp16.safetensors",
]


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for role, (repository, filename, expected_size) in MODEL_DOWNLOADS.items():
        destination = MODEL_PATHS[role]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == expected_size:
            print(f"{role}: zaten mevcut ({destination.name})")
            continue
        print(f"{role}: {filename} indiriliyor...")
        staging_dir = models_dir / ".downloads"
        staging_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            local_dir=str(staging_dir),
        )
        downloaded = staging_dir / filename
        if downloaded.is_file():
            destination.unlink(missing_ok=True)
            downloaded.replace(destination)
        if not destination.is_file() or destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"{role} modeli doğrulanamadı: {destination}. "
                "İndirme kesilmiş veya dosya beklenen boyutta değil."
            )

    print("Yerel Stable Diffusion XL bileşenleri indiriliyor/doğrulanıyor...")
    snapshot_download(
        repo_id=IMAGE_MODEL_ID,
        local_dir=str(DEFAULT_IMAGE_MODEL),
        allow_patterns=IMAGE_MODEL_FILES,
    )
    missing_files = [
        relative_path
        for relative_path in (
            "model_index.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        )
        if not (DEFAULT_IMAGE_MODEL / relative_path).is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "Stable Diffusion indirmesi eksik dosyalar bıraktı: "
            + ", ".join(missing_files)
        )
    print("Görsel güvenlik denetleyicisi indiriliyor/doğrulanıyor...")
    snapshot_download(
        repo_id=SAFETY_CHECKER_ID,
        local_dir=str(DEFAULT_SAFETY_CHECKER_MODEL),
        allow_patterns=SAFETY_CHECKER_FILES,
    )
    missing_checker_files = [
        relative_path
        for relative_path in (
            "feature_extractor/preprocessor_config.json",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
        )
        if not (DEFAULT_SAFETY_CHECKER_MODEL / relative_path).is_file()
    ]
    if missing_checker_files:
        raise RuntimeError(
            "Görsel güvenlik denetleyicisi eksik dosyalar bıraktı: "
            + ", ".join(missing_checker_files)
        )
    print("Model indirme işlemi tamamlandı.")


if __name__ == "__main__":
    main()
