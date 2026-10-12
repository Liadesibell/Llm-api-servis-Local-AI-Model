from __future__ import annotations

import gc
import os
import threading
import uuid
from pathlib import Path

import numpy as np

from ink_ai.conversations import default_database_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_MODEL = PROJECT_ROOT / "models" / "stable-diffusion-xl-base-1.0"
DEFAULT_SAFETY_CHECKER_MODEL = PROJECT_ROOT / "models" / "stable-diffusion-safety-checker"
IMAGE_MODEL_ID = "floppy"
IMAGE_MODEL_NAME = "Floppy"


class LocalImageGenerator:
    model_id = IMAGE_MODEL_ID
    model_name = IMAGE_MODEL_NAME

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or Path(
            os.environ.get("INK_AI_IMAGE_MODEL_PATH", str(DEFAULT_IMAGE_MODEL))
        ).expanduser()
        self.safety_checker_path = Path(
            os.environ.get(
                "INK_AI_SAFETY_CHECKER_PATH",
                str(DEFAULT_SAFETY_CHECKER_MODEL),
            )
        ).expanduser()
        self.pipeline = None
        self.safety_checker = None
        self.feature_extractor = None
        self.lock = threading.Lock()
        self.output_dir = default_database_path().parent / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, steps: int = 18, seed: int | None = None) -> Path:
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Yerel görsel modeli bulunamadı: {self.model_path}. "
                "uv run --no-sync python -m ink_ai.download_models komutuyla "
                "modeli indirin."
            )

        with self.lock:
            pipeline = self._get_pipeline()
            generator = None
            if seed is not None:
                import torch

                generator = torch.Generator(device="cpu").manual_seed(seed)
            result = pipeline(
                prompt=prompt,
                negative_prompt=(
                    "low quality, blurry, distorted, messy composition, "
                    "extra objects, illegible text, watermark, artifacts, "
                    "oversaturated"
                ),
                num_inference_steps=steps,
                width=1024,
                height=1024,
                generator=generator,
            )
            if any(self._check_nsfw(result.images)):
                raise ValueError(
                    "Görsel güvenlik denetiminden geçmedi; farklı bir istem deneyin."
                )
            image_path = self.output_dir / f"{uuid.uuid4().hex}.png"
            result.images[0].save(image_path)
            return image_path

    def _get_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        import torch
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        )
        if torch.cuda.is_available():
            pipeline.enable_attention_slicing()
            pipeline.vae.enable_slicing()
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            has_room_for_gpu_pipeline = (
                total_memory >= 16 * 1024**3 and free_memory >= 10 * 1024**3
            )
            if has_room_for_gpu_pipeline:
                pipeline.to("cuda")
            else:
                pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_attention_slicing()
            pipeline.vae.enable_slicing()
            pipeline.to("cpu")
        self.pipeline = pipeline
        return pipeline

    def _check_nsfw(self, images: list) -> list[bool]:
        if self.safety_checker is None or self.feature_extractor is None:
            if not self.safety_checker_path.is_dir():
                raise FileNotFoundError(
                    f"Görsel güvenlik modeli bulunamadı: {self.safety_checker_path}"
                )
            import torch
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPImageProcessor

            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                str(self.safety_checker_path),
                subfolder="feature_extractor",
                local_files_only=True,
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                str(self.safety_checker_path),
                subfolder="safety_checker",
                torch_dtype=torch.float32,
                variant="fp16",
                local_files_only=True,
            ).to("cpu")

        clip_input = self.feature_extractor(
            images=images,
            return_tensors="pt",
        ).pixel_values
        image_batch = np.stack([np.asarray(image) for image in images])
        _checked_images, flags = self.safety_checker(
            clip_input=clip_input,
            images=image_batch,
        )
        if flags is None or len(flags) != len(images):
            raise RuntimeError("Görsel güvenlik denetleyicisi geçerli sonuç döndürmedi.")
        return flags

    def unload(self) -> None:
        with self.lock:
            if self.pipeline is None and self.safety_checker is None:
                return
            self.pipeline = None
            self.safety_checker = None
            self.feature_extractor = None
            gc.collect()
            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
