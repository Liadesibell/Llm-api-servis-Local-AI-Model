import pytest

from ink_ai.brain.llm import LocalLLMEngine
from ink_ai.brain.model_router import MODEL_PATHS
from ink_ai.core.llm import ModelManager


class FakeLlama:
    def __init__(self, **options):
        self.options = options
        self.request = None

    def create_chat_completion(self, **request):
        self.request = request
        return {"choices": [{"message": {"content": "Test response"}}]}


def test_local_llm_uses_configured_context_gpu_layers_and_output_length(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    monkeypatch.setenv("INK_AI_GPU_LAYERS", "-1")
    monkeypatch.setenv("INK_AI_CONTEXT_SIZE", "8192")
    monkeypatch.setenv("INK_AI_MAX_RESPONSE_TOKENS", "2048")

    engine = LocalLLMEngine(str(model_path))
    answer = engine.generate_response("System", "Write a program", temperature=0.1)

    assert engine_instance is not None
    assert engine_instance.options["n_gpu_layers"] == -1
    assert engine_instance.options["n_ctx"] == 8192
    assert engine_instance.request["max_tokens"] == 2048
    assert engine_instance.request["temperature"] == 0.1
    assert engine_instance.request["top_p"] == 0.9
    assert engine_instance.request["repeat_penalty"] == 1.12
    assert answer == "Test response"


def test_local_llm_drops_old_history_when_context_window_is_exceeded(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    class ContextLimitedLlama(FakeLlama):
        def create_chat_completion(self, **request):
            self.request = request
            if any(
                "old conversation" in str(message["content"])
                for message in request["messages"]
            ):
                raise ValueError(
                    "Requested tokens exceed context window of 2048"
                )
            return {"choices": [{"message": {"content": "Test response"}}]}

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = ContextLimitedLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)

    engine = LocalLLMEngine(str(model_path))
    engine.generate_response(
        "System",
        "Current question",
        history=[
            {"role": "user", "content": "old conversation question"},
            {"role": "assistant", "content": "old conversation answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ],
    )

    assert engine_instance is not None
    assert [message["content"] for message in engine_instance.request["messages"]] == [
        "System",
        "recent question",
        "recent answer",
        "Current question",
    ]


def test_large_local_llm_uses_bounded_context_and_response_defaults(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "deepseek-r1-distill-qwen-14b.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    monkeypatch.setattr(
        "ink_ai.brain.llm.os.path.getsize",
        lambda _path: 9 * 1024**3,
    )
    monkeypatch.delenv("INK_AI_MAX_RESPONSE_TOKENS", raising=False)

    engine = LocalLLMEngine(str(model_path))
    engine.generate_response("System", "Solve a math problem")

    assert engine_instance is not None
    assert engine_instance.request["max_tokens"] == 512


def test_local_llm_passes_gguf_lora_adapter_to_runtime(tmp_path, monkeypatch):
    model_path = tmp_path / "coder.gguf"
    adapter_path = tmp_path / "coder-lora.gguf"
    model_path.write_bytes(b"base")
    adapter_path.write_bytes(b"adapter")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)

    LocalLLMEngine(str(model_path), lora_path=str(adapter_path))

    assert engine_instance is not None
    assert engine_instance.options["lora_path"] == str(adapter_path)


def test_local_llm_hides_leading_think_block(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test model")

    def create_llama(**options):
        engine = FakeLlama(**options)
        engine.create_chat_completion = lambda **_request: {
            "choices": [
                {
                    "message": {
                        "content": "<think>private draft</think>\nVisible answer"
                    }
                }
            ]
        }
        return engine

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)

    engine = LocalLLMEngine(str(model_path))

    assert engine.generate_response("System", "Question") == "Visible answer"


def test_local_llm_streams_visible_answer_after_think_block(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test model")
    emitted = []
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine = FakeLlama(**options)
        engine_instance = engine

        def stream_completion(**request):
            engine.request = request
            return iter(
                [
                    {"choices": [{"delta": {"content": "<thi"}}]},
                    {"choices": [{"delta": {"content": "nk>private thought</think>"}}]},
                    {"choices": [{"delta": {"content": "\nVisible "}}]},
                    {"choices": [{"delta": {"content": "answer"}}]},
                ]
            )

        engine.create_chat_completion = stream_completion
        return engine

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    engine = LocalLLMEngine(str(model_path))

    answer = engine.generate_response(
        "System",
        "Question",
        stream_callback=emitted.append,
    )

    assert answer == "Visible answer"
    assert "".join(emitted) == answer
    assert all("private thought" not in part for part in emitted)
    assert engine_instance is not None
    assert engine_instance.request["top_p"] == 0.9
    assert engine_instance.request["repeat_penalty"] == 1.12


def test_local_llm_reports_when_stream_contains_no_visible_answer(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test model")

    def create_llama(**_options):
        engine = FakeLlama()
        engine.create_chat_completion = lambda **_request: iter(
            [
                {"choices": [{"delta": {"content": "<think>unfinished reasoning"}}]},
            ]
        )
        return engine

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    engine = LocalLLMEngine(str(model_path))

    with pytest.raises(ValueError, match="bütçesi gizli düşünme"):
        engine.generate_response(
            "System",
            "Question",
            stream_callback=lambda _text: None,
        )


def test_large_local_llm_uses_memory_conscious_defaults(tmp_path, monkeypatch):
    model_path = tmp_path / "large-model.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    monkeypatch.setattr(
        "ink_ai.brain.llm.os.path.getsize",
        lambda _path: 13 * 1024**3,
    )
    monkeypatch.delenv("INK_AI_GPU_LAYERS", raising=False)
    monkeypatch.delenv("INK_AI_CONTEXT_SIZE", raising=False)

    LocalLLMEngine(str(model_path))

    assert engine_instance is not None
    assert engine_instance.options["n_gpu_layers"] == 12
    assert engine_instance.options["n_ctx"] == 2048


def test_compact_local_llm_uses_gpu_with_bounded_context(tmp_path, monkeypatch):
    model_path = tmp_path / "compact-model.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setattr("ink_ai.brain.llm.Llama", create_llama)
    monkeypatch.delenv("INK_AI_GPU_LAYERS", raising=False)
    monkeypatch.delenv("INK_AI_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("INK_AI_MAX_RESPONSE_TOKENS", raising=False)

    engine = LocalLLMEngine(str(model_path))
    engine.generate_response("System", "Hello")

    assert engine_instance is not None
    assert engine_instance.options["n_gpu_layers"] == -1
    assert engine_instance.options["n_ctx"] == 2048
    assert engine_instance.request["max_tokens"] == 512


def test_legacy_model_manager_uses_shared_coder_model_path(tmp_path, monkeypatch):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test model")
    engine_instance = None

    def create_llama(**options):
        nonlocal engine_instance
        engine_instance = FakeLlama(**options)
        return engine_instance

    monkeypatch.setitem(MODEL_PATHS, "coder", model_path)
    monkeypatch.setattr("ink_ai.core.llm.Llama", create_llama)
    monkeypatch.setattr(ModelManager, "llm", None)

    ModelManager.load_model()

    assert engine_instance is not None
    assert engine_instance.options["model_path"] == str(model_path)
    assert engine_instance.options["n_gpu_layers"] == -1
    assert engine_instance.options["n_ctx"] == 2048
