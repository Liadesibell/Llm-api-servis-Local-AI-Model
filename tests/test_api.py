import json
import threading

import pytest
from fastapi.testclient import TestClient

from ink_ai.api.app import create_app
from ink_ai.api.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderError,
)
from ink_ai.memory.manager import KnowledgeManager
from ink_ai.web_research import ResearchResult


@pytest.fixture(autouse=True)
def isolate_knowledge_database(monkeypatch, tmp_path):
    monkeypatch.setenv("INK_AI_KNOWLEDGE_DIR", str(tmp_path / "incoming"))
    monkeypatch.setenv("INK_AI_KNOWLEDGE_DB", str(tmp_path / "knowledge.sqlite3"))
    monkeypatch.setattr(
        KnowledgeManager,
        "research_web",
        lambda self, query, limit=5: [],
    )


class SahteSohbetMotoru:
    def __init__(self, model_yolu: str):
        self.model_yolu = model_yolu
        self.son_gecmis = None
        self.son_girdi = None
        self.sistem_komutu = None

    def generate_response(
        self,
        system_prompt,
        user_input,
        history=None,
        temperature=0.45,
        max_tokens=None,
        stream_callback=None,
    ):
        self.son_gecmis = history
        self.son_girdi = user_input
        self.sistem_komutu = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        answer = f"Yanıt: {user_input}"
        if stream_callback is not None:
            stream_callback(answer)
        return answer


def test_saglik_model_yokken_api_durumunu_bildirir(tmp_path):
    uygulama = create_app(model_path=tmp_path / "olmayan.gguf")

    with TestClient(uygulama) as istemci:
        yanit = istemci.get("/api/saglik")

    assert yanit.status_code == 200
    assert yanit.json()["durum"] == "model_bekleniyor"
    assert yanit.json()["model_yuklu"] is False
    assert len(yanit.json()["models"]) == 4
    assert isinstance(yanit.json()["gpu_offload_supported"], bool)
    assert isinstance(yanit.json()["image_model_available"], bool)


def test_sohbet_yerel_motoru_ve_gecmisi_kullanir(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    motorlar = []

    def motor_olustur(yol):
        motor = SahteSohbetMotoru(yol)
        motorlar.append(motor)
        return motor

    uygulama = create_app(model_path=model_yolu, engine_factory=motor_olustur)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={
                "mesaj": "Merhaba",
                "gecmis": [
                    {"rol": "kullanici", "icerik": "Önceki soru"},
                    {"rol": "asistan", "icerik": "Önceki yanıt"},
                ],
            },
        )

    assert yanit.status_code == 200
    assert yanit.json()["yanit"].startswith("Yanıt: Merhaba")
    assert motorlar[0].son_gecmis == [
        {"role": "user", "content": "Önceki soru"},
        {"role": "assistant", "content": "Önceki yanıt"},
    ]


def test_sohbet_akisi_model_parcalarini_ve_tamamlanma_olayini_dondurur(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    uygulama = create_app(model_path=model_yolu, engine_factory=SahteSohbetMotoru)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={"mesaj": "Merhaba", "stream": True},
        )

    events = [json.loads(line) for line in yanit.text.splitlines()]
    assert yanit.status_code == 200
    assert yanit.headers["content-type"].startswith("application/x-ndjson")
    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["content"].startswith("Yanıt:")
    assert events[1]["answer"].startswith("Yanıt:")
    assert events[1]["model"] == "coder"


def test_sohbet_akisi_hatasinda_hassas_istisna_metnini_gizler(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")

    class HataVerenMotor(SahteSohbetMotoru):
        def generate_response(self, *args, **kwargs):
            raise RuntimeError("private model path")

    uygulama = create_app(model_path=model_yolu, engine_factory=HataVerenMotor)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={"mesaj": "Merhaba", "stream": True, "progress_id": "e123"},
        )
        progress = istemci.get("/api/sohbet/durum/e123")

    events = [json.loads(line) for line in yanit.text.splitlines()]
    assert yanit.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["status_code"] == 500
    assert "RuntimeError" in events[-1]["detail"]
    assert "private model path" not in events[-1]["detail"]
    assert "private model path" not in yanit.text
    assert progress.json()["stage"] == "error"
    assert progress.json()["done"] is True


def test_sohbet_akisi_surfaces_actionable_model_value_errors(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")

    class EmptyAnswerEngine(SahteSohbetMotoru):
        def generate_response(self, *args, **kwargs):
            raise ValueError(
                "Yerel model görünür yanıt üretmedi. Yanıt bütçesi gizli düşünme "
                "çıktısıyla tükenmiş olabilir."
            )

    app = create_app(model_path=model_path, engine_factory=EmptyAnswerEngine)

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={"mesaj": "Merhaba", "stream": True},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert "Yanıt bütçesi gizli düşünme" in events[-1]["detail"]


def test_sohbet_hatasinda_ilerleme_durumu_error_olarak_kaydedilir(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")

    class HataVerenMotor(SahteSohbetMotoru):
        def generate_response(self, *args, **kwargs):
            raise RuntimeError("model failure")

    uygulama = create_app(model_path=model_yolu, engine_factory=HataVerenMotor)

    with TestClient(uygulama, raise_server_exceptions=False) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={"mesaj": "Merhaba", "progress_id": "e456"},
        )
        progress = istemci.get("/api/sohbet/durum/e456")

    assert yanit.status_code == 500
    assert progress.json()["stage"] == "error"
    assert progress.json()["done"] is True


def test_model_yokken_sohbet_acik_hata_dondurur(tmp_path):
    uygulama = create_app(model_path=tmp_path / "olmayan.gguf")

    with TestClient(uygulama) as istemci:
        yanit = istemci.post("/api/sohbet", json={"mesaj": "Merhaba"})

    assert yanit.status_code == 503
    assert "Yerel model yüklenmedi" in yanit.json()["detail"]


def test_bos_mesaj_kabul_edilmez(tmp_path):
    uygulama = create_app(model_path=tmp_path / "olmayan.gguf")

    with TestClient(uygulama) as istemci:
        yanit = istemci.post("/api/sohbet", json={"mesaj": ""})

    assert yanit.status_code == 422


def test_sohbet_eklenen_dosya_icerigini_analiz_icin_motora_aktarir(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    motorlar = []

    def motor_olustur(yol):
        motor = SahteSohbetMotoru(yol)
        motorlar.append(motor)
        return motor

    uygulama = create_app(model_path=model_yolu, engine_factory=motor_olustur)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={
                "mesaj": "Bu dosyadaki sorunu bul.",
                "files": [{"name": "main.py", "content": "print(unknown_name)"}],
            },
        )

    assert yanit.status_code == 200
    assert "DOSYA 1: main.py" in motorlar[0].son_girdi
    assert "print(unknown_name)" in motorlar[0].son_girdi
    assert "her dosyayı tek tek incele" in motorlar[0].sistem_komutu


def test_sohbet_gorsel_ekini_kabul_eder_ve_gorsel_yetenegini_aciklar(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    motorlar = []

    def motor_olustur(yol):
        motor = SahteSohbetMotoru(yol)
        motorlar.append(motor)
        return motor

    uygulama = create_app(model_path=model_yolu, engine_factory=motor_olustur)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={
                "mesaj": "Bu görseli incele",
                "images": [
                    {
                        "name": "clipboard.png",
                        "content_base64": "iVBORw0KGgo=",
                    }
                ],
            },
        )
        yetenekler = istemci.get("/api/yetenekler")

    assert yanit.status_code == 200
    assert "clipboard.png" in motorlar[0].son_girdi
    assert "görsel piksellerini okuyamaz" in motorlar[0].son_girdi
    assert yetenekler.status_code == 200
    assert yetenekler.json()["attachments"]["image_understanding"] is False
    assert yetenekler.json()["terminal"]["execution"] is False


def test_sohbet_gecersiz_gorsel_verisini_reddeder(tmp_path):
    uygulama = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={
                "mesaj": "Görseli incele",
                "images": [{"name": "bad.png", "content_base64": "not base64"}],
            },
        )

    assert yanit.status_code == 422


def test_sohbet_dosya_metin_limiti_asildiginda_acik_hata_dondurur(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    uygulama = create_app(model_path=model_yolu, engine_factory=SahteSohbetMotoru)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={
                "mesaj": "Analiz et",
                "files": [{"name": "large.txt", "content": "x" * 6_001}],
            },
        )

    assert yanit.status_code == 422


def test_manuel_model_secimi_response_headerinda_bildirilir(tmp_path):
    model_yolu = tmp_path / "model.gguf"
    model_yolu.write_bytes(b"sinama")
    uygulama = create_app(model_path=model_yolu, engine_factory=SahteSohbetMotoru)

    with TestClient(uygulama) as istemci:
        yanit = istemci.post(
            "/api/sohbet",
            json={"mesaj": "Python kodunu açıkla", "model": "coder"},
        )

    assert yanit.status_code == 200
    assert yanit.headers["x-ink-ai-model"] == "coder"


def test_chat_defaults_to_automatic_model_routing(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    (tmp_path / "flash.gguf").write_bytes(b"test")
    app = create_app(model_path=model_path, engine_factory=SahteSohbetMotoru)

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={"mesaj": "Selam", "progress_id": "abc123"},
        )
        progress = client.get("/api/sohbet/durum/abc123")

    assert response.status_code == 200
    assert response.headers["x-ink-ai-model"] == "flash"
    assert progress.json()["stage"] == "complete"
    assert progress.json()["done"] is True


def test_chat_automatically_uses_floppy_for_image_generation(tmp_path):
    app = create_app(model_path=tmp_path / "missing-coder.gguf")
    image_path = tmp_path / "floppy-image.png"
    image_path.write_bytes(b"png")

    with TestClient(app) as client:
        app.state.image_generator.generate = lambda prompt: image_path
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "Bana neon yeşil bir logo oluştur",
                "progress_id": "f10a123",
            },
        )
        progress = client.get("/api/sohbet/durum/f10a123")

    assert response.status_code == 200
    assert response.headers["x-ink-ai-model"] == "floppy"
    assert response.json() == {
        "yanit": "Floppy görseli oluşturdu.",
        "image_filename": "floppy-image.png",
    }
    assert progress.json() == {
        "stage": "complete",
        "message": "Floppy görseli hazır",
        "done": True,
    }


def test_modeller_endpoint_lists_file_availability(tmp_path):
    coder = tmp_path / "coder.gguf"
    coder.write_bytes(b"model")
    uygulama = create_app(
        model_path=coder,
        engine_factory=lambda path: SahteSohbetMotoru(path),
    )

    with TestClient(uygulama) as istemci:
        models = istemci.get("/api/modeller").json()["models"]

    assert {model["id"] for model in models} == {
        "flash",
        "reasoning",
        "coder",
        "floppy",
    }
    assert next(model for model in models if model["id"] == "coder")["available"]
    assert not next(model for model in models if model["id"] == "flash")["available"]
    assert next(model for model in models if model["id"] == "floppy")["label"].startswith(
        "Floppy"
    )


def test_gorsel_endpoint_reports_missing_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_IMAGE_MODEL_PATH", str(tmp_path / "missing-image-model"))
    uygulama = create_app(model_path=tmp_path / "missing-coder.gguf")

    with TestClient(uygulama) as istemci:
        yanit = istemci.post("/api/gorsel", json={"prompt": "Neon green ink logo"})

    assert yanit.status_code == 503
    assert "ink_ai.download_models" in yanit.json()["detail"]


def test_gorsel_endpoint_returns_local_generated_image(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_IMAGE_MODEL_PATH", str(tmp_path / "image-model"))
    uygulama = create_app(model_path=tmp_path / "missing-coder.gguf")
    image_dir = tmp_path / "generated"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    image_path.write_bytes(b"\x89PNG test data")

    class FakeImageGenerator:
        output_dir = image_dir

        def generate(self, prompt, steps=24, seed=None):
            assert prompt == "Create a green icon"
            assert steps == 10
            return image_path

        def unload(self):
            pass

    with TestClient(uygulama) as istemci:
        uygulama.state.image_generator = FakeImageGenerator()
        yanit = istemci.post(
            "/api/gorsel",
            json={"prompt": "Create a green icon", "steps": 10},
        )
        indirilen = istemci.get(yanit.json()["url"])

    assert yanit.status_code == 200
    assert indirilen.status_code == 200
    assert indirilen.content == b"\x89PNG test data"


def test_knowledge_api_indexes_searches_and_supplies_sources_to_chat(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(
        model_path=model_path,
        engine_factory=engine_factory,
        knowledge_manager_factory=lambda: KnowledgeManager(
            tmp_path / "incoming",
            tmp_path / "knowledge.sqlite3",
        ),
    )
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "git.md").write_text(
        "Git rebase reapplies commits on a new base commit.",
        encoding="utf-8",
    )

    with TestClient(app) as client:
        status = client.get("/api/bilgi/durumu")
        indexing = client.post("/api/bilgi/indeksle")
        search = client.get("/api/bilgi/ara", params={"q": "Git rebase"})
        answer = client.post(
            "/api/sohbet",
            json={"mesaj": "Git rebase nasıl çalışır?", "model": "coder"},
        )

    assert status.status_code == 200
    assert indexing.json()["indexed"] == 1
    assert search.json()["results"][0]["source"] == "git.md"
    assert "Git rebase reapplies commits" in engines[0].son_girdi
    assert answer.headers["x-ink-ai-sources"]


def test_knowledge_api_accepts_direct_text_indexing(tmp_path):
    app = create_app(
        model_path=tmp_path / "missing.gguf",
        knowledge_manager_factory=lambda: KnowledgeManager(
            tmp_path / "incoming",
            tmp_path / "knowledge.sqlite3",
        ),
    )

    with TestClient(app) as client:
        added = client.post(
            "/api/bilgi/metin",
            json={
                "name": "guide.md",
                "content": "PowerShell Get-Process displays running processes.",
            },
        )
        results = client.get("/api/bilgi/ara", params={"q": "PowerShell process"})

    assert added.status_code == 200
    assert results.json()["results"][0]["source"].endswith("guide.md")


def test_web_research_endpoint_returns_and_indexes_sources(tmp_path, monkeypatch):
    manager = KnowledgeManager(
        tmp_path / "incoming",
        tmp_path / "knowledge.sqlite3",
    )
    sources = [
        {
            "title": "Official Docker guide",
            "url": "https://docs.example.test/docker",
            "snippet": "Docker runs containers.",
            "content_source": "page",
        }
    ]
    monkeypatch.setattr(
        manager,
        "research_web",
        lambda query, limit: sources if query == "Docker containers" and limit == 3 else [],
    )
    app = create_app(
        model_path=tmp_path / "missing.gguf",
        knowledge_manager_factory=lambda: manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/arastir",
            json={"query": "Docker containers", "max_results": 3},
        )

    assert response.status_code == 200
    assert response.json()["provider"] == "DuckDuckGo"
    assert response.json()["indexed"] == 1
    assert response.json()["sources"][0]["url"] == "https://docs.example.test/docker"


def test_web_search_uses_only_user_query_and_adds_cited_sources(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engine_instances = []
    searched_queries = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engine_instances.append(engine)
        return engine

    def knowledge_factory():
        manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")

        def research(query: str, limit: int = 5) -> list[ResearchResult]:
            searched_queries.append(query)
            return [
                {
                    "title": f"Official Docker docs {index}",
                    "url": f"https://docs{index}.example.test/docker",
                    "snippet": f"Docker source {index} explains container isolation.",
                    "content_source": "page",
                    "links": [{
                        "type": "download",
                        "label": "Source ZIP",
                        "url": "https://docs.example.test/download.zip",
                    }],
                }
                for index in range(1, 5)
            ]

        manager.research_web = research
        return manager

    app = create_app(
        model_path=model_path,
        engine_factory=engine_factory,
        knowledge_manager_factory=knowledge_factory,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "Docker için güncel resmi doküman bilgisi nedir?",
                "files": [
                    {"name": "private.py", "content": "PRIVATE_CODE_MUST_NOT_BE_SEARCHED"}
                ],
            },
        )

    assert response.status_code == 200
    assert searched_queries == ["Docker için güncel resmi doküman bilgisi nedir?"]
    assert "PRIVATE_CODE_MUST_NOT_BE_SEARCHED" in engine_instances[0].son_girdi
    assert all(
        f"Docker source {index} explains" in engine_instances[0].son_girdi
        for index in range(1, 5)
    )
    assert "https://docs.example.test" not in engine_instances[0].son_girdi
    assert "https://" not in response.headers["x-ink-ai-sources"]
    assert all(
        f"Official Docker docs {index}" in response.headers["x-ink-ai-sources"]
        for index in range(1, 5)
    )
    assert response.headers["x-ink-ai-model"] == "coder"
    assert "kullanıcıyı başka bir siteye yönlendirme" in engine_instances[0].sistem_komutu
    assert "Kullanıcı özellikle" in engine_instances[0].son_girdi


def test_web_search_shows_download_link_only_when_explicitly_requested(
    tmp_path,
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    def knowledge_factory():
        manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")
        manager.research_web = lambda query, limit=5: [
            {
                "title": "Release page",
                "url": "https://example.test/releases",
                "snippet": "Latest release details.",
                "content_source": "page",
                "links": [
                    {
                        "type": "download",
                        "label": "Package ZIP",
                        "url": "https://example.test/app.zip",
                    }
                ],
            }
        ]
        return manager

    app = create_app(
        model_path=model_path,
        engine_factory=engine_factory,
        knowledge_manager_factory=knowledge_factory,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "Uygulama için indirme bağlantısını bul",
                "web_search": True,
                "model": "coder",
            },
        )

    assert response.status_code == 200
    assert "https://example.test/app.zip" in engines[0].son_girdi


def test_flash_model_does_not_automatically_search_the_web(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    (tmp_path / "flash.gguf").write_bytes(b"test")
    searched = []
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")
    manager.research_web = lambda query, limit=5: searched.append(query) or []
    app = create_app(
        model_path=model_path,
        engine_factory=SahteSohbetMotoru,
        knowledge_manager_factory=lambda: manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "Selam",
                "model": "flash",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-ink-ai-model"] == "flash"
    assert searched == []


def test_web_research_is_selected_by_query_need_not_model_role(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    (tmp_path / "flash.gguf").write_bytes(b"test")
    searched = []
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")
    manager.research_web = lambda query, limit=5: searched.append(query) or []
    app = create_app(
        model_path=model_path,
        engine_factory=SahteSohbetMotoru,
        knowledge_manager_factory=lambda: manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "2026 yılı için güncel ürün fiyatı nedir?",
                "model": "flash",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-ink-ai-model"] == "flash"
    assert searched == ["2026 yılı için güncel ürün fiyatı nedir?"]


def test_turkish_vps_price_query_automatically_searches_web(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    (tmp_path / "flash.gguf").write_bytes(b"test")
    searched = []
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")
    manager.research_web = lambda query, limit=5: searched.append(query) or []
    app = create_app(
        model_path=model_path,
        engine_factory=SahteSohbetMotoru,
        knowledge_manager_factory=lambda: manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={"mesaj": "VPS hosting fiyatlarını merak ediyorum"},
        )

    assert response.status_code == 200
    assert searched == ["VPS hosting fiyatlarını merak ediyorum"]


def test_explicit_web_search_is_honored_for_flash_model(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    (tmp_path / "flash.gguf").write_bytes(b"test")
    searched = []
    manager = KnowledgeManager(tmp_path / "incoming", tmp_path / "knowledge.sqlite3")
    manager.research_web = lambda query, limit=5: searched.append(query) or []
    app = create_app(
        model_path=model_path,
        engine_factory=SahteSohbetMotoru,
        knowledge_manager_factory=lambda: manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={"mesaj": "Docker", "model": "flash", "web_search": True},
        )

    assert response.status_code == 200
    assert searched == ["Docker"]


@pytest.mark.parametrize("stream", [False, True])
def test_unknown_answer_falls_back_to_web_research_without_exposing_marker(
    tmp_path, stream
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engines = []
    searched = []

    class UncertainEngine(SahteSohbetMotoru):
        def __init__(self, path):
            super().__init__(path)
            self.calls = 0

        def generate_response(
            self,
            system_prompt,
            user_input,
            history=None,
            temperature=0.45,
            max_tokens=None,
            stream_callback=None,
        ):
            self.calls += 1
            self.son_girdi = user_input
            answer = (
                "[[WEB_SEARCH_REQUIRED]]"
                if self.calls == 1
                else "Resmi Docker belgelerine göre çözüm şu şekildedir."
            )
            if stream_callback is not None:
                stream_callback(answer)
            return answer

    def engine_factory(path):
        engine = UncertainEngine(path)
        engines.append(engine)
        return engine

    def knowledge_factory():
        manager = KnowledgeManager(
            tmp_path / "incoming",
            tmp_path / "knowledge.sqlite3",
        )

        def research(query, limit=5):
            searched.append(query)
            return [
                {
                    "title": "Official Docker docs",
                    "url": "https://docs.example.test/docker",
                    "snippet": "The official guide explains Docker networking.",
                    "content_source": "page",
                }
            ]

        manager.research_web = research
        return manager

    app = create_app(
        model_path=model_path,
        engine_factory=engine_factory,
        knowledge_manager_factory=knowledge_factory,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet",
            json={
                "mesaj": "Docker ağ eklentisini nasıl yapılandırırım?",
                "model": "coder",
                "stream": stream,
            },
        )

    assert response.status_code == 200
    assert searched == ["Docker ağ eklentisini nasıl yapılandırırım?"]
    assert engines[0].calls == 2
    assert "[[WEB_SEARCH_REQUIRED]]" not in response.text
    assert "Resmi Docker belgelerine göre" in response.text
    if stream:
        done_event = json.loads(response.text.splitlines()[-1])
        assert "Official Docker docs" in done_event["sources"]
    else:
        assert "Official Docker docs" in response.headers["x-ink-ai-sources"]


def test_document_analysis_summarizes_attached_documents(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(model_path=model_path, engine_factory=engine_factory)

    with TestClient(app) as client:
        response = client.post(
            "/api/belge/analiz",
            json={
                "task": "summarize",
                "model": "auto",
                "documents": [
                    {
                        "name": "policy.md",
                        "content": "Refunds are available within 30 days.",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["task"] == "summarize"
    assert response.json()["documents"] == ["policy.md"]
    assert "policy.md" in engines[0].son_girdi
    assert "30 days" in engines[0].son_girdi


def test_document_extraction_returns_validated_json_object(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")

    class JsonEngine(SahteSohbetMotoru):
        def generate_response(self, system_prompt, user_input, **kwargs):
            self.son_girdi = user_input
            self.sistem_komutu = system_prompt
            return '{"company": "Ink AI", "date": null}'

    app = create_app(model_path=model_path, engine_factory=JsonEngine)

    with TestClient(app) as client:
        response = client.post(
            "/api/belge/analiz",
            json={
                "task": "extract",
                "model": "auto",
                "fields": ["company", "date"],
                "documents": [{"name": "report.txt", "content": "Company: Ink AI"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["result"] == {"company": "Ink AI", "date": None}


def test_document_qa_requires_question(tmp_path):
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        response = client.post(
            "/api/belge/analiz",
            json={
                "task": "qa",
                "documents": [{"name": "guide.txt", "content": "Some content"}],
            },
        )

    assert response.status_code == 422


def test_code_review_endpoint_passes_code_to_coder_model(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(model_path=model_path, engine_factory=engine_factory)

    with TestClient(app) as client:
        response = client.post(
            "/api/kod/analiz",
            json={
                "name": "app.py",
                "content": "print(unknown_name)",
                "kind": "code",
                "task": "debug",
                "language": "Python",
            },
        )

    assert response.status_code == 200
    assert response.json()["kind"] == "code"
    assert response.json()["task"] == "debug"
    assert "print(unknown_name)" in engines[0].son_girdi
    assert "Python" in engines[0].son_girdi


def test_log_analysis_rejects_incompatible_review_task(tmp_path):
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        response = client.post(
            "/api/kod/analiz",
            json={
                "content": "ERROR: database unavailable",
                "kind": "logs",
                "task": "review",
            },
        )

    assert response.status_code == 422


def test_structured_chat_returns_parsed_json_object(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")

    class JsonEngine(SahteSohbetMotoru):
        def generate_response(self, system_prompt, user_input, **kwargs):
            self.sistem_komutu = system_prompt
            assert "geçerli bir JSON nesnesi" in system_prompt
            return '{"answer": "42", "confidence": 0.9}'

    app = create_app(model_path=model_path, engine_factory=JsonEngine)

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet/json",
            json={"mesaj": "What is the answer?", "model": "auto"},
        )

    assert response.status_code == 200
    assert response.json()["result"] == {"answer": "42", "confidence": 0.9}
    assert response.json()["model"] == "coder"


def test_structured_chat_does_not_fallback_to_plain_text(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test")

    class PlainTextEngine(SahteSohbetMotoru):
        def generate_response(self, *args, **kwargs):
            return "This is not JSON."

    app = create_app(model_path=model_path, engine_factory=PlainTextEngine)

    with TestClient(app) as client:
        response = client.post(
            "/api/sohbet/json",
            json={"mesaj": "Give a JSON answer", "model": "auto"},
        )

    assert response.status_code == 502
    assert "geçerli JSON" in response.json()["detail"]


def test_capabilities_describe_all_five_api_features(tmp_path):
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        capabilities = client.get("/api/yetenekler").json()

    assert capabilities["document_analysis"]["endpoint"] == "POST /api/belge/analiz"
    assert capabilities["web_research"]["endpoint"] == "POST /api/arastir"
    assert capabilities["code_and_log_analysis"]["endpoint"] == "POST /api/kod/analiz"
    assert capabilities["structured_output"]["endpoint"] == "POST /api/sohbet/json"
    assert capabilities["image_generation"]["endpoint"] == "POST /api/gorsel"


def test_v1_models_lists_configured_openai_compatible_provider(tmp_path, monkeypatch):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    monkeypatch.setenv("INK_AI_OPENAI_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("INK_AI_OPENAI_API_KEY", "never-return-this-secret")

    def fake_request(self, path, payload=None, **kwargs):
        assert self.api_key == "never-return-this-secret"
        if path == "/models":
            return {"data": [{"id": "example-model"}]}
        if path == "/chat/completions":
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "remote answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        if path == "/embeddings":
            return {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            }
        raise AssertionError(path)

    monkeypatch.setattr(OpenAICompatibleProvider, "_request", fake_request)
    app = create_app(model_path=model_path, engine_factory=SahteSohbetMotoru)

    with TestClient(app) as client:
        models = client.get("/v1/models")
        completion = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/example-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        embeddings = client.post(
            "/v1/embeddings",
            json={"model": "openai/example-model", "input": "hello"},
        )

    assert models.status_code == 200
    assert any(item["id"] == "openai/example-model" for item in models.json()["data"])
    assert completion.json()["choices"][0]["message"]["content"] == "remote answer"
    assert embeddings.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert "never-return-this-secret" not in models.text


def test_v1_chat_supports_local_model_and_openai_response_shape(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(model_path=model_path, engine_factory=engine_factory)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "messages": [
                    {"role": "system", "content": "Use short answers."},
                    {"role": "user", "content": "Explain this Python code"},
                ],
                "max_tokens": 64,
                "temperature": 0.3,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0
    assert "Use short answers." in engines[0].sistem_komutu
    assert engines[0].max_tokens == 64


def test_v1_stream_uses_openai_server_sent_event_chunks(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    app = create_app(model_path=model_path, engine_factory=SahteSohbetMotoru)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "stream": True,
                "messages": [{"role": "user", "content": "Explain this API"}],
            },
        )

    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event["object"] for event in events] == [
        "chat.completion.chunk",
        "chat.completion.chunk",
    ]
    assert events[0]["choices"][0]["delta"]["content"].startswith("Yanıt:")
    assert events[1]["choices"][0]["finish_reason"] == "stop"
    assert response.text.rstrip().endswith("data: [DONE]")


def test_v1_tools_are_returned_as_proposals_and_never_executed(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    executions = []

    class ToolProposalEngine(SahteSohbetMotoru):
        def generate_response(self, system_prompt, user_input, **kwargs):
            assert "Never execute tools" in system_prompt
            return '{"tool_name": "lookup", "arguments": {"query": "weather"}}'

    def execute_tool(*_args):
        executions.append(True)

    app = create_app(model_path=model_path, engine_factory=ToolProposalEngine)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "messages": [{"role": "user", "content": "Check weather"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"query": "weather"}',
    }
    assert executions == []


def test_v1_collections_isolate_indexed_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_COLLECTIONS_DIR", str(tmp_path / "collections"))
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        created = client.post("/v1/collections", json={"id": "support"})
        added = client.post(
            "/v1/collections/support/documents",
            json={"name": "returns.md", "content": "Refunds are available within 30 days."},
        )
        searched = client.get(
            "/v1/collections/support/search",
            params={"q": "refunds 30 days"},
        )
        listed = client.get("/v1/collections")
        missing = client.get(
            "/v1/collections/other/search",
            params={"q": "refunds"},
        )

    assert created.status_code == 201
    assert added.status_code == 200
    assert searched.json()["data"][0]["source"].endswith("returns.md")
    assert listed.json()["data"][0]["id"] == "support"
    assert missing.status_code == 404


def test_v1_chat_uses_only_requested_collection_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_COLLECTIONS_DIR", str(tmp_path / "collections"))
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(model_path=model_path, engine_factory=engine_factory)

    with TestClient(app) as client:
        client.post("/v1/collections", json={"id": "support"})
        client.post(
            "/v1/collections/support/documents",
            json={
                "name": "returns.md",
                "content": "Refund policy allows refunds within 30 days.",
            },
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "collection_id": "support",
                "messages": [{"role": "user", "content": "What is the refund policy?"}],
            },
        )

    assert response.status_code == 200
    assert "Refund policy allows refunds within 30 days." in engines[0].son_girdi
    assert response.json()["sources"][0]["source"].endswith("returns.md")


def test_v1_api_key_request_ids_metrics_and_rate_limiting(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_API_KEY", "test-api-key")
    monkeypatch.setenv("INK_AI_RATE_LIMIT_PER_MINUTE", "10")
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        unauthorized = client.get("/v1/models")
        authorized = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-api-key", "X-Request-ID": "req_123"},
        )
        metrics = client.get("/metrics")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.headers["x-request-id"] == "req_123"
    assert "ink_ai_api_requests_total" in metrics.text


def test_v1_rate_limit_returns_retry_after(tmp_path, monkeypatch):
    monkeypatch.setenv("INK_AI_RATE_LIMIT_PER_MINUTE", "1")
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        first = client.get("/v1/models")
        second = client.get("/v1/models")

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1


def test_ollama_provider_normalizes_chat_tool_calls_and_embeddings():
    provider = OllamaProvider("http://127.0.0.1:11434", timeout=3)
    requests = []

    def fake_request(path, payload=None, **kwargs):
        requests.append((path, payload))
        if path == "/api/chat":
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "weather"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 4,
                "eval_count": 2,
            }
        return {"embeddings": [[0.1, 0.2]]}

    provider._request = fake_request
    chat = provider.chat(
        "model-name",
        {
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "max_tokens": 32,
        },
    )
    embeddings = provider.embeddings("embed-model", ["hello"])

    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    assert chat["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert requests[0][1]["options"]["num_predict"] == 32
    assert embeddings["data"][0]["embedding"] == [0.1, 0.2]


def test_openai_provider_uses_bearer_auth_without_exposing_key():
    secret = "provider-secret-test"
    provider = OpenAICompatibleProvider(
        "https://models.example.test/v1",
        secret,
        timeout=10,
    )

    assert provider.headers == {"Authorization": f"Bearer {secret}"}


def test_provider_stream_ignores_sse_metadata_and_rejects_non_object_events(
    monkeypatch,
):
    class FakeOpener:
        def __init__(self, body):
            self.body = body

        def open(self, *_args, **_kwargs):
            from io import BytesIO

            return BytesIO(self.body)

    provider = OpenAICompatibleProvider(
        "https://models.example.test/v1",
        None,
        timeout=10,
    )
    monkeypatch.setattr(
        "ink_ai.api.providers._URL_OPENER",
        FakeOpener(
            b": keep-alive\n"
            b"event: message\n"
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
            b"data: [DONE]\n"
        ),
    )

    events = list(provider.stream("/chat/completions", {}, {}))
    assert events == [{"choices": [{"delta": {"content": "hello"}}]}]

    monkeypatch.setattr(
        "ink_ai.api.providers._URL_OPENER",
        FakeOpener(b"[]\n"),
    )
    with pytest.raises(ProviderError, match="invalid stream event"):
        list(provider.stream("/chat/completions", {}, {}))


def test_ollama_stream_reports_upstream_generation_errors():
    provider = OllamaProvider("http://127.0.0.1:11434", timeout=3)
    provider.stream = lambda *_args, **_kwargs: iter(
        [{"error": "model not found"}]
    )

    with pytest.raises(ProviderError, match="failed to generate"):
        list(provider.stream_chat("missing-model", {}))


def test_local_openai_chat_accepts_text_content_blocks(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")
    engines = []

    def engine_factory(path):
        engine = SahteSohbetMotoru(path)
        engines.append(engine)
        return engine

    app = create_app(model_path=model_path, engine_factory=engine_factory)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            },
        )

    assert response.status_code == 200
    assert "Hi" in engines[0].son_girdi


def test_openai_image_generation_returns_openai_image_response(tmp_path):
    app = create_app(model_path=tmp_path / "missing.gguf")
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    image_path = output_dir / "image.png"
    image_path.write_bytes(b"image bytes")

    class FakeImageGenerator:
        def generate(self, prompt):
            assert prompt == "A green ink logo"
            return image_path

        def unload(self):
            pass

    with TestClient(app) as client:
        app.state.image_generator = FakeImageGenerator()
        response = client.post(
            "/v1/images/generations",
            json={"model": "floppy", "prompt": "A green ink logo", "response_format": "b64_json"},
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["b64_json"] == "aW1hZ2UgYnl0ZXM="


def test_v1_request_cancellation_sets_registered_event(tmp_path):
    app = create_app(model_path=tmp_path / "missing.gguf")

    with TestClient(app) as client:
        cancellation = threading.Event()
        app.state.active_request_cancellations["inflight-123"] = cancellation
        response = client.post("/v1/cancellations/inflight-123")

    assert response.status_code == 200
    assert response.json()["status"] == "cancellation_requested"
    assert cancellation.is_set()


def test_v1_local_json_mode_rejects_invalid_model_output(tmp_path):
    model_path = tmp_path / "coder.gguf"
    model_path.write_bytes(b"test")

    class PlainTextEngine(SahteSohbetMotoru):
        def generate_response(self, *args, **kwargs):
            return "not-json"

    app = create_app(model_path=model_path, engine_factory=PlainTextEngine)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "coder",
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            },
        )

    assert response.status_code == 502