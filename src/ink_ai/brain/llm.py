import os
import re
from collections.abc import Callable

from llama_cpp import Llama, llama_supports_gpu_offload
from llama_cpp.llama_types import ChatCompletionRequestMessage


def gpu_offload_supported() -> bool:
    return bool(llama_supports_gpu_offload())


class LocalLLMEngine:
    def __init__(
        self,
        model_path: str = "models/qwen3.5-35b-a3b-q4_k_m.gguf",
        lora_path: str | None = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                "İndirilen GGUF dosyasını 'models/' klasörüne kopyalayın."
            )
        if lora_path is not None and not os.path.isfile(lora_path):
            raise FileNotFoundError(f"GGUF LoRA adaptörü bulunamadı: {lora_path}")

        print("Yerel model belleğe yükleniyor...")
        large_model = os.path.getsize(model_path) > 7 * 1024**3
        default_gpu_layers = "12" if large_model else "-1"
        options = {
            "model_path": model_path,
            "n_gpu_layers": int(
                os.environ.get("INK_AI_GPU_LAYERS", default_gpu_layers)
            ),
            "n_ctx": int(
                os.environ.get("INK_AI_CONTEXT_SIZE", "2048")
            ),
            "verbose": False,
        }
        self.max_tokens = int(
            os.environ.get("INK_AI_MAX_RESPONSE_TOKENS", "512")
        )
        if lora_path is not None:
            options["lora_path"] = lora_path
        self.llm = Llama(
            **options,
        )
        print("Yerel model başarıyla yüklendi.")

    def generate_response(
        self,
        system_prompt: str,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.45,
        max_tokens: int | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        messages: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": system_prompt},
        ]
        for item in history or []:
            role = item.get("role")
            content = item.get("content")
            if content is None:
                raise ValueError("Sohbet geçmişindeki mesaj biçimi geçersiz.")
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "user":
                messages.append({"role": "user", "content": content})
            else:
                raise ValueError("Sohbet geçmişinde yalnızca user/assistant olabilir.")
        messages.append({"role": "user", "content": user_input})
        history_messages = messages[1:-1]
        while True:
            current_messages = [
                messages[0],
                *history_messages,
                messages[-1],
            ]
            try:
                if stream_callback is None:
                    output = self.llm.create_chat_completion(
                        messages=current_messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                        top_p=0.9,
                        repeat_penalty=1.12,
                        stream=False,
                    )
                    if not isinstance(output, dict):
                        raise TypeError("Yerel model beklenmeyen yanıt biçimi döndürdü.")
                    content = output["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise TypeError("Yerel model yanıtı metin biçiminde değil.")
                else:
                    output = self.llm.create_chat_completion(
                        messages=current_messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                        top_p=0.9,
                        repeat_penalty=1.12,
                        stream=True,
                    )
                    if isinstance(output, dict) or not hasattr(output, "__iter__"):
                        raise TypeError(
                            "Yerel model beklenen yanıt akışını döndürmedi."
                        )
                    content = self._read_stream(output, stream_callback)
                break
            except ValueError as error:
                if "exceed context window" not in str(error).casefold():
                    raise
                if not history_messages:
                    raise
                if (
                    len(history_messages) >= 2
                    and history_messages[0]["role"] == "user"
                    and history_messages[1]["role"] == "assistant"
                ):
                    history_messages = history_messages[2:]
                else:
                    history_messages = history_messages[1:]
        content = re.sub(
            r"\A\s*<think>.*?(?:</think>|\Z)\s*",
            "",
            content,
            count=1,
            flags=re.DOTALL,
        )
        if not content.strip():
            raise ValueError(
                "Yerel model görünür yanıt üretmedi. Yanıt bütçesi gizli düşünme "
                "çıktısıyla tükenmiş olabilir; INK_AI_MAX_RESPONSE_TOKENS değerini "
                "artırıp yeniden deneyin."
            )
        return content

    @staticmethod
    def _read_stream(output, stream_callback: Callable[[str], None]) -> str:
        parts: list[str] = []
        pending = ""
        think_resolved = False
        closing_tag = "</think>"

        for chunk in output:
            if not isinstance(chunk, dict):
                raise TypeError("Yerel model akışında geçersiz parça alındı.")
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("Yerel model akışında seçim bilgisi eksik.")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("Yerel model akışında geçersiz seçim alındı.")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise TypeError("Yerel model akışında delta bilgisi eksik.")
            piece = delta.get("content", "")
            if piece is None:
                continue
            if not isinstance(piece, str):
                raise TypeError("Yerel model akışında metin olmayan parça alındı.")

            if think_resolved:
                pending += piece
                if not pending.strip():
                    continue
                if not parts:
                    pending = pending.lstrip()
                if pending:
                    parts.append(pending)
                    stream_callback(pending)
                    pending = ""
                continue

            pending += piece
            stripped = pending.lstrip()
            if not stripped or "<think>".startswith(stripped):
                continue
            if stripped.startswith("<think>"):
                end = pending.find(closing_tag)
                if end < 0:
                    continue
                pending = pending[end + len(closing_tag) :].lstrip("\r\n ")
            think_resolved = True
            if not parts:
                pending = pending.lstrip()
            if pending:
                parts.append(pending)
                stream_callback(pending)
            pending = ""

        if pending.strip():
            stripped = pending.lstrip()
            if think_resolved or (
                not stripped.startswith("<think>")
                and not "<think>".startswith(stripped)
            ):
                if not parts:
                    pending = pending.lstrip()
                parts.append(pending)
                stream_callback(pending)
        return "".join(parts)