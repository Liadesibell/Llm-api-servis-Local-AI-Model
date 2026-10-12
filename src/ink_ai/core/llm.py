import os

from llama_cpp import Llama

from ink_ai.brain.model_router import MODEL_PATHS


class ModelManager:
    _instance = None
    llm = None

    @classmethod
    def load_model(cls):
        model_path = MODEL_PATHS["coder"]
        if model_path.is_file():
            print(f"[Ink AI] Model yukleniyor: {model_path}")
            large_model = model_path.stat().st_size > 7 * 1024**3
            cls.llm = Llama(
                model_path=str(model_path),
                n_ctx=int(
                    os.environ.get(
                        "INK_AI_CONTEXT_SIZE",
                        "2048",
                    )
                ),
                n_gpu_layers=int(
                    os.environ.get(
                        "INK_AI_GPU_LAYERS",
                        "12" if large_model else "-1",
                    )
                ),
                verbose=False,
            )
            print("[Ink AI] Model basariyla yuklendi!")
        else:
            print(f"[Ink AI] UYARI: Model dosyasi bulunamadi: {model_path}")

    @classmethod
    def get_llm(cls):
        return cls.llm