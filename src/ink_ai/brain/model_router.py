from __future__ import annotations

import gc
import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ink_ai.brain.llm import LocalLLMEngine

LOGGER = logging.getLogger("ink_ai.models")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class GenerationCancelled(RuntimeError):
    pass

MODEL_PATHS = {
    "flash": Path(
        os.environ.get(
            "INK_AI_FLASH_MODEL_PATH",
            PROJECT_ROOT / "models" / "qwen3-4b-q4_k_m.gguf",
        )
    ).expanduser(),
    "reasoning": Path(
        os.environ.get(
            "INK_AI_REASONING_MODEL_PATH",
            PROJECT_ROOT / "models" / "qwen3.5-35b-a3b-q4_k_m.gguf",
        )
    ).expanduser(),
    "coder": Path(
        os.environ.get(
            "INK_AI_CODER_MODEL_PATH",
            PROJECT_ROOT / "models" / "qwen3.5-35b-a3b-q4_k_m.gguf",
        )
    ).expanduser(),
}

MODEL_LABELS = {
    "flash": "Flash · günlük görevler",
    "reasoning": "Reasoning · matematik/fizik",
    "coder": "Coder · yazılım",
}
MODEL_TEMPERATURES = {
    "flash": 0.3,
    "reasoning": 0.3,
    "coder": 0.2,
}
CONCISE_RESPONSE_INSTRUCTION = (
    "Yanıt biçimi: Kullanıcının dilinde önce doğrudan sonucu ver. Varsayılan "
    "olarak 2–4 kısa cümle veya en fazla 4 özlü madde kullan; yalnızca "
    "kullanıcı ayrıntı isterse genişlet. Dolgu, tekrar, uzun giriş, gereksiz "
    "başlık ve sembol yığınları üretme. İç muhakemeyi gösterme; sonucu "
    "anlaşılır, doğal ve okunabilir dille açıkla. Kısalık uğruna doğruluğu, "
    "gerekli kodu veya kritik güvenlik uyarılarını atlama."
)

CODER_WORDS = (
    "kod", "python", "javascript", "typescript", "program", "fonksiyon",
    "hata", "debug", "api", "html", "css", "sql", "class ", "import ",
    "refactor", "derle", "yazılım", "repo", "repository",
    "code", "coding", "function", "programming", "bug", "script",
)
REASONING_WORDS = (
    "matematik", "matematiksel", "denklem", "hesapla", "hesaplama",
    "integral", "türev", "fizik", "fiziksel", "kimya", "algoritma",
    "kanıtla", "ispatla", "optimizasyon", "olasılık", "istatistik",
    "problem çöz", "mantık sorusu", "formül", "solve", "calculate",
    "equation", "physics", "chemistry", "prove", "proof", "optimization",
    "analiz et", "karşılaştır", "tasarla", "mimari", "planla", "neden",
    "nasıl çalışır", "adım adım", "değerlendir", "strateji",
    "math", "derivative", "probability",
)
DEVOPS_WORDS = (
    "docker", "dockerfile", "compose", "kubernetes", "kubectl", "helm",
    "terraform", "ansible", "ci/cd", "github actions", "gitlab ci",
    "pipeline", "deployment", "deploy", "devops", "nginx", "linux",
    "powershell", "terminal", "bash", "shell", "ssh", "systemd",
    "firewall", "sunucu", "server", "log", "exception", "stack trace",
    "build", "test", "lint", "migration", "git ", "pull request",
)
IMAGE_GENERATION_PHRASES = (
    "görsel oluştur",
    "görsel üret",
    "resim oluştur",
    "resim üret",
    "resim çiz",
    "çizim yap",
    "logo oluştur",
    "logo tasarla",
    "logo yap",
    "profil fotoğrafı oluştur",
    "profil resmi oluştur",
    "profil resmi çiz",
    "afiş oluştur",
    "poster oluştur",
    "görsel yap",
    "resim yap",
    "fotoğraf oluştur",
    "fotoğrafını oluştur",
    "avatar oluştur",
    "avatar yap",
    "duvar kağıdı oluştur",
    "generate an image",
    "generate a picture",
    "create an image",
    "create a picture",
    "make an image",
    "draw me",
    "make a picture",
    "create a logo",
    "design a logo",
)
MODEL_INSTRUCTIONS = {
    "flash": (
        "Kısa ve günlük görevlerde doğrudan yanıt ver; istenmedikçe uzun analiz "
        "yapma. Emin olmadığın noktaları belirt."
    ),
    "reasoning": (
        "Matematik, fizik, kimya ve algoritma problemlerinde doğru yöntemi "
        "seç, sonucu kontrol et ve yalnızca gerekli çözüm adımlarını paylaş. "
        "Belirsizliği açıkça belirt; iç muhakemeyi yazma."
    ),
    "coder": (
        "Sen kıdemli yazılım ve platform mühendisisin. Önce amaç, mevcut kod, "
        "çalışma zamanı, işletim sistemi, kabuk ve kısıtları belirle. Mevcut "
        "dosya/proje bağlamı sağlandıysa onu kaynak kabul et; dosyaları görmeden "
        "değişiklik yaptığını iddia etme. Önce doğrudan çalışan en küçük çözümü "
        "ver; istenmedikçe gereksiz katman, soyutlama, bağımlılık veya tekrar "
        "ekleme. Kodu doğru dilin sözdizimi ve proje kalıplarıyla yaz; küçük, "
        "bütünlüklü yamalar, tür güvenliği, girdi "
        "doğrulama, açık hata yönetimi ve odaklı testler tercih et. Kod "
        "incelemesinde dosya/satır kanıtı, önem derecesi, etki ve düzeltme ver. "
        "Python, TypeScript/JavaScript, web API'leri, SQL/veritabanı, eşzamanlılık, "
        "test, paketleme ve performans konularında doğru algoritma ve veri "
        "yapılarını kullan; ölçüm olmadan hız iddiasında bulunma. Güvenilirlik "
        "ve güvenli varsayılanları görevin kapsamına uygun ele al. Sistem/DevOps "
        "işlerinde Git, "
        "PowerShell, Bash, Linux, Docker/Compose, Kubernetes, CI/CD, GitHub "
        "Actions, Terraform/IaC, ağ/DNS/TLS, proxy, bulut dağıtımı, servis "
        "yönetimi, log ve metrikleri ayırt et; gerekli bağlam eksikse sor. "
        "Komutları önce salt-okunur tanılama, sonra değişiklik ve doğrulama "
        "adımlarına ayır. Windows PowerShell ve POSIX shell sözdizimini "
        "karıştırma. Yıkıcı veya üretim etkili komutlarda riski, önkoşulu, "
        "etkiyi ve geri almayı açıkça belirt; kullanıcı onayı olmadan komut "
        "çalıştırdığını iddia etme. Gizli anahtarları/parolaları komut satırına "
        "koymayı önerme. Bilmediğin sürüm/API/dosya/log sonucunu uydurma; "
        "belirsizliği açıkça söyle ve doğrulanabilir komut/test öner."
    ),
}


class SohbetMotoru(Protocol):
    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.45,
        max_tokens: int | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str: ...


def route_model(prompt: str) -> str:
    if is_image_generation_request(prompt):
        return "floppy"
    text = prompt.casefold()
    if any(word in text for word in (*CODER_WORDS, *DEVOPS_WORDS)):
        return "coder"
    if any(word in text for word in REASONING_WORDS) or re.search(
        r"\b\d+(?:[.,]\d+)?\s*[+\-*/×÷^=]\s*\d+", text
    ):
        return "reasoning"
    return "flash"


def is_image_generation_request(prompt: str) -> bool:
    text = prompt.casefold()
    informational_markers = (
        "nasıl oluşturulur",
        "nasıl yapılır",
        "nasıl tasarlanır",
        "nedir",
        "hakkında bilgi",
        "açıkla",
        "anlat",
        "öğret",
    )
    if any(marker in text for marker in informational_markers):
        return False
    return any(phrase in text for phrase in IMAGE_GENERATION_PHRASES)


class LocalModelRouter:
    def __init__(
        self,
        model_paths: dict[str, Path] | None = None,
        engine_factory: Callable[[str], SohbetMotoru] | None = None,
        preload: str | None = None,
    ) -> None:
        self.model_paths = model_paths or MODEL_PATHS.copy()
        self.engine_factory = engine_factory or LocalLLMEngine
        self.loaded_model: str | None = None
        self.engine: SohbetMotoru | None = None
        self.lock = threading.RLock()
        if preload is not None:
            self.load(preload)

    def available_models(self) -> list[dict[str, str | bool]]:
        return [
            {
                "id": model_id,
                "label": MODEL_LABELS[model_id],
                "available": path.is_file(),
                "loaded": self.loaded_model == model_id,
            }
            for model_id, path in self.model_paths.items()
        ]

    def select_model(self, requested: str, prompt: str) -> str:
        if requested == "auto":
            return route_model(prompt)
        if requested not in MODEL_LABELS:
            raise ValueError(f"Bilinmeyen model rolü: {requested}")
        return requested

    def classify_request(self, prompt: str) -> str:
        selected = route_model(prompt)
        if selected == "flash" and not self.model_paths.get(
            "flash", Path()
        ).is_file():
            available = [
                model_id
                for model_id, path in self.model_paths.items()
                if path.is_file()
            ]
            if available:
                fallback = available[-1]
                LOGGER.warning(
                    "Flash modeli bulunamadı; mevcut %s modeli kullanılıyor.",
                    fallback,
                )
                return fallback
        return selected

    def load(self, model_id: str) -> SohbetMotoru:
        if model_id not in self.model_paths or model_id not in MODEL_LABELS:
            raise ValueError(f"Bilinmeyen model rolü: {model_id}")
        model_path = self.model_paths[model_id]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{MODEL_LABELS[model_id]} modeli bulunamadı: {model_path}"
            )
        if self.loaded_model == model_id and self.engine is not None:
            return self.engine

        with self.lock:
            if self.loaded_model == model_id and self.engine is not None:
                return self.engine
            if (
                self.engine is not None
                and self.loaded_model is not None
                and self.model_paths[self.loaded_model] == model_path
            ):
                loaded_lora = (
                    os.environ.get("INK_AI_CODER_LORA_PATH")
                    if self.loaded_model == "coder"
                    else None
                )
                requested_lora = (
                    os.environ.get("INK_AI_CODER_LORA_PATH")
                    if model_id == "coder"
                    else None
                )
                if loaded_lora == requested_lora:
                    self.loaded_model = model_id
                    return self.engine
            LOGGER.info("%s modeli belleğe yükleniyor.", MODEL_LABELS[model_id])
            self.engine = None
            self.loaded_model = None
            gc.collect()
            if self.engine_factory is LocalLLMEngine:
                lora_path = (
                    os.environ.get("INK_AI_CODER_LORA_PATH")
                    if model_id == "coder"
                    else None
                )
                self.engine = LocalLLMEngine(
                    str(model_path),
                    lora_path=lora_path,
                )
            else:
                self.engine = self.engine_factory(str(model_path))
            self.loaded_model = model_id
            return self.engine

    def unload(self) -> None:
        with self.lock:
            self.engine = None
            self.loaded_model = None
            gc.collect()

    def generate_response(
        self,
        requested_model: str,
        system_prompt: str,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
        stream_callback: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, str]:
        with self.lock:
            selected_model = (
                self.classify_request(user_input)
                if requested_model == "auto"
                else self.select_model(requested_model, user_input)
            )
            if progress_callback is not None:
                stage = {
                    "flash": "thinking",
                    "reasoning": "analyzing",
                    "coder": "coding",
                    "floppy": "generating",
                }[selected_model]
                message = {
                    "flash": "Kısa yanıt hazırlanıyor",
                    "reasoning": "Çözüm adımları analiz ediliyor",
                    "coder": "Kod ve teknik çözüm hazırlanıyor",
                    "floppy": "Floppy görseli oluşturuyor",
                }[selected_model]
                progress_callback(
                    stage,
                    message,
                )
            engine = self.load(selected_model)
            specialist_prompt = (
                f"{system_prompt}\n\nUzman rolü: {MODEL_INSTRUCTIONS[selected_model]}"
                f"\n\n{CONCISE_RESPONSE_INSTRUCTION}"
            )
            if stream_callback is None:
                engine_options: dict[str, object] = {
                    "history": history,
                    "temperature": (
                        MODEL_TEMPERATURES[selected_model]
                        if temperature is None
                        else temperature
                    ),
                }
                if max_tokens is not None:
                    engine_options["max_tokens"] = max_tokens
                answer = engine.generate_response(
                    specialist_prompt,
                    user_input,
                    **engine_options,
                )
            else:
                def send_stream_piece(piece: str) -> None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise GenerationCancelled("Generation was cancelled.")
                    stream_callback(piece)

                engine_options = {
                    "history": history,
                    "temperature": (
                        MODEL_TEMPERATURES[selected_model]
                        if temperature is None
                        else temperature
                    ),
                    "stream_callback": send_stream_piece,
                }
                if max_tokens is not None:
                    engine_options["max_tokens"] = max_tokens
                answer = engine.generate_response(
                    specialist_prompt,
                    user_input,
                    **engine_options,
                )
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("Generation was cancelled.")
            return answer, selected_model