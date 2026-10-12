import threading
from pathlib import Path

from ink_ai.brain.model_router import (
    MODEL_PATHS,
    GenerationCancelled,
    LocalModelRouter,
    is_image_generation_request,
    route_model,
)
from ink_ai.download_models import MODEL_DOWNLOADS


def test_default_model_paths_use_flash_and_shared_qwen35_specialist():
    assert MODEL_PATHS["flash"].name == "qwen3-4b-q4_k_m.gguf"
    assert MODEL_PATHS["reasoning"].name == "qwen3.5-35b-a3b-q4_k_m.gguf"
    assert MODEL_PATHS["coder"] == MODEL_PATHS["reasoning"]
    assert MODEL_DOWNLOADS["reasoning"] == MODEL_DOWNLOADS["coder"]
    assert MODEL_DOWNLOADS["reasoning"] == (
        "unsloth/Qwen3.5-35B-A3B-GGUF",
        "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        22_016_023_168,
    )


class FakeEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.temperature = None
        self.system_prompt = None

    def generate_response(
        self,
        system_prompt,
        user_input,
        history=None,
        temperature=0.45,
        stream_callback=None,
    ):
        self.temperature = temperature
        self.system_prompt = system_prompt
        answer = f"{Path(self.model_path).stem}: {user_input}"
        if stream_callback is not None:
            stream_callback(answer)
        return answer


def test_automatic_routing_chooses_specialist_roles():
    assert route_model("Selam, fikir verir misin?") == "flash"
    assert route_model("Hesapla: 2 + 2") == "reasoning"
    assert route_model("Python kodundaki hatayı düzelt") == "coder"
    assert route_model("Docker Compose dağıtımını düzelt") == "coder"
    assert route_model("Bu konuyu adım adım karşılaştır") == "reasoning"


def test_image_generation_requests_are_detected_without_matching_image_questions():
    assert is_image_generation_request("Bana neon yeşil bir logo oluştur")
    assert is_image_generation_request("Create an image of a mountain")
    assert not is_image_generation_request("Bu görselde ne görüyorsun?")
    assert not is_image_generation_request("Logo tasarımının tarihini anlat")
    assert not is_image_generation_request("Logo nasıl oluşturulur?")


def test_router_loads_only_requested_model_and_reports_missing(tmp_path):
    paths = {
        role: tmp_path / f"{role}.gguf"
        for role in ("flash", "reasoning", "coder")
    }
    for path in paths.values():
        path.write_bytes(b"test")
    created = []

    def engine_factory(model_path):
        engine = FakeEngine(model_path)
        created.append(engine)
        return engine

    router = LocalModelRouter(paths, engine_factory)
    answer, used_model = router.generate_response(
        "auto", "be helpful", "Python kodumu gözden geçir"
    )

    assert used_model == "coder"
    assert answer.startswith("coder:")
    assert [Path(engine.model_path).name for engine in created] == ["coder.gguf"]
    assert created[0].temperature == 0.2
    assert "2–4 kısa cümle" in created[0].system_prompt
    assert "sembol yığınları üretme" in created[0].system_prompt
    assert router.available_models()[2]["loaded"] is True

    missing_router = LocalModelRouter(
        {"coder": tmp_path / "missing.gguf"}, engine_factory
    )
    try:
        missing_router.generate_response("coder", "system", "test")
    except FileNotFoundError as error:
        assert "missing.gguf" in str(error)
    else:
        raise AssertionError("A missing selected model must be reported.")


def test_router_reuses_selected_model_then_releases_it_when_switching(tmp_path):
    paths = {}
    for model_id in ("flash", "coder"):
        paths[model_id] = tmp_path / f"{model_id}.gguf"
        paths[model_id].write_bytes(b"test")
    created = []

    def engine_factory(model_path):
        engine = FakeEngine(model_path)
        created.append(engine)
        return engine

    router = LocalModelRouter(paths, engine_factory)
    router.generate_response("flash", "system", "hello")
    router.generate_response("flash", "system", "again")
    router.generate_response("coder", "system", "code")

    assert len(created) == 2
    assert router.loaded_model == "coder"


def test_router_reuses_shared_model_for_reasoning_and_coder(tmp_path):
    shared_path = tmp_path / "qwen3.5.gguf"
    shared_path.write_bytes(b"test")
    created = []

    def engine_factory(model_path):
        engine = FakeEngine(model_path)
        created.append(engine)
        return engine

    router = LocalModelRouter(
        {"reasoning": shared_path, "coder": shared_path},
        engine_factory,
    )
    reasoning_engine = router.load("reasoning")
    coder_engine = router.load("coder")

    assert coder_engine is reasoning_engine
    assert len(created) == 1
    assert router.loaded_model == "coder"


def test_auto_router_sends_ambiguous_requests_directly_to_flash(tmp_path):
    paths = {
        role: tmp_path / f"{role}.gguf"
        for role in ("flash", "reasoning", "coder")
    }
    for path in paths.values():
        path.write_bytes(b"test")
    created = []

    def engine_factory(model_path):
        engine = FakeEngine(model_path)
        created.append(Path(model_path).name)
        return engine

    router = LocalModelRouter(paths, engine_factory)
    progress = []
    answer, selected = router.generate_response(
        "auto",
        "system",
        "Please classify this unusual request",
        progress_callback=lambda stage, message: progress.append((stage, message)),
    )

    assert selected == "flash"
    assert answer.startswith("flash:")
    assert created == ["flash.gguf"]
    assert [stage for stage, _message in progress] == ["thinking"]


def test_auto_router_sends_arithmetic_directly_to_reasoning_model(tmp_path):
    paths = {
        role: tmp_path / f"{role}.gguf"
        for role in ("flash", "reasoning", "coder")
    }
    for path in paths.values():
        path.write_bytes(b"test")
    created = []

    def engine_factory(model_path):
        engine = FakeEngine(model_path)
        created.append(engine)
        return engine

    router = LocalModelRouter(paths, engine_factory)
    _answer, selected = router.generate_response("auto", "system", "Calculate 12 * 8")

    assert selected == "reasoning"
    assert [Path(engine.model_path).name for engine in created] == ["reasoning.gguf"]
    assert created[0].temperature == 0.3


def test_router_forwards_streamed_text_from_the_selected_model(tmp_path):
    paths = {"coder": tmp_path / "coder.gguf"}
    paths["coder"].write_bytes(b"test")
    router = LocalModelRouter(paths, FakeEngine)
    chunks = []

    answer, selected = router.generate_response(
        "coder",
        "system",
        "Write code",
        stream_callback=chunks.append,
    )

    assert selected == "coder"
    assert chunks == [answer]


def test_router_stops_forwarding_tokens_after_cancellation(tmp_path):
    path = tmp_path / "coder.gguf"
    path.write_bytes(b"test")
    cancellation = threading.Event()

    class CancellingEngine(FakeEngine):
        def generate_response(self, system_prompt, user_input, **kwargs):
            cancellation.set()
            kwargs["stream_callback"]("partial output")
            return "partial output"

    router = LocalModelRouter({"coder": path}, CancellingEngine)

    try:
        router.generate_response(
            "coder",
            "system",
            "prompt",
            stream_callback=lambda _piece: None,
            cancel_event=cancellation,
        )
    except GenerationCancelled:
        pass
    else:
        raise AssertionError("A cancelled local generation must stop.")
