import json
from types import SimpleNamespace
from unittest.mock import patch

from ink_ai.gui import InkAIApp, InkAIClient, split_answer_segments


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload
        self.headers = {"X-Ink-AI-Model": "auto"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_gui_client_sends_chat_history_and_reads_answer():
    response = FakeResponse({"yanit": "Merhaba!"})
    history = [("user", "Selam"), ("assistant", "Merhaba, nasıl yardımcı olayım?")]

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        answer = InkAIClient("http://127.0.0.1:8000/api/sohbet").send_message(
            "Nasılsın?", history
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8000/api/sohbet"
    assert json.loads(request.data) == {
        "mesaj": "Nasılsın?",
        "gecmis": [
            {"rol": "kullanici", "icerik": "Selam"},
            {"rol": "asistan", "icerik": "Merhaba, nasıl yardımcı olayım?"},
        ],
        "files": [],
        "images": [],
        "web_search": False,
        "model": "auto",
    }
    assert answer == ("Merhaba!", "auto")


def test_gui_client_points_health_check_at_model_status():
    client = InkAIClient("http://127.0.0.1:8000/api/sohbet")
    response = FakeResponse({"durum": "hazir", "model_yuklu": True})

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        health = client.check_health()

    assert urlopen.call_args.args[0].full_url == "http://127.0.0.1:8000/api/saglik"
    assert health["model_yuklu"] is True


def test_gui_client_sends_document_content_to_api():
    response = FakeResponse({"yanit": "Dosyanı inceledim."})
    files = [{"name": "main.py", "content": "print('hello')"}]

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        answer = InkAIClient("http://127.0.0.1:8000/api/sohbet").send_message(
            "Dosyayı incele", [], files
        )

    assert json.loads(urlopen.call_args.args[0].data)["files"] == files
    assert answer == ("Dosyanı inceledim.", "auto")


def test_gui_client_sends_pasted_image_to_api():
    response = FakeResponse({"yanit": "Görsel eklendi."})
    images = [{"name": "clipboard.png", "content_base64": "iVBORw=="}]

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        InkAIClient("http://127.0.0.1:8000/api/sohbet").send_message(
            "Görseli açıkla", [], images=images
        )

    assert json.loads(urlopen.call_args.args[0].data)["images"] == images


def test_gui_client_can_enable_opt_in_web_search():
    response = FakeResponse({"yanit": "Bulunan kaynaklara göre..."})

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        InkAIClient("http://127.0.0.1:8000/api/sohbet").send_message(
            "Güncel Docker dokümanları", [], web_search=True
        )

    assert json.loads(urlopen.call_args.args[0].data)["web_search"] is True


def test_gui_client_displays_local_knowledge_sources():
    response = FakeResponse({"yanit": "Git rebase commitleri taşır."})
    response.headers["X-Ink-AI-Sources"] = json.dumps(
        [{"id": "KAYNAK 1", "source": "git/rebase.md"}]
    )

    with patch("ink_ai.gui.urlopen", return_value=response):
        answer, _ = InkAIClient(
            "http://127.0.0.1:8000/api/sohbet"
        ).send_message("Git rebase nedir?", [])

    assert "Kaynaklar:\n- git/rebase.md" in answer


def test_gui_client_displays_web_source_titles_without_navigating():
    response = FakeResponse({"yanit": "Kaynakları karşılaştırınca sonuç..."})
    response.headers["X-Ink-AI-Sources"] = json.dumps(
        [{"id": "WEB KAYNAK 1", "source": "Python docs (docs.python.org)"}]
    )

    with patch("ink_ai.gui.urlopen", return_value=response):
        answer, _ = InkAIClient(
            "http://127.0.0.1:8000/api/sohbet"
        ).send_message("Python nedir?", [], web_search=True)

    assert "Kaynaklar:\n- Python docs (docs.python.org)" in answer
    assert "https://" not in answer


def test_gui_client_can_select_a_specialist_model():
    response = FakeResponse({"yanit": "42"})

    with patch("ink_ai.gui.urlopen", return_value=response) as urlopen:
        InkAIClient("http://127.0.0.1:8000/api/sohbet").send_message(
            "6 × 7", [], model="reasoning"
        )

    assert json.loads(urlopen.call_args.args[0].data)["model"] == "reasoning"


def test_gui_client_reads_streamed_response_incrementally():
    class FakeStreamResponse(FakeResponse):
        def __init__(self):
            super().__init__({})
            self.headers = {"X-Ink-AI-Model": "coder"}
            self.lines = iter(
                [
                    b'{"type":"heartbeat"}\n',
                    b'{"type":"token","content":"def add("}\n',
                    b'{"type":"token","content":"a, b): return a + b"}\n',
                    b'{"type":"done","answer":"def add(a, b): return a + b","model":"coder"}\n',
                ]
            )

        def readline(self):
            return next(self.lines, b"")

    received = []
    with patch("ink_ai.gui.urlopen", return_value=FakeStreamResponse()) as urlopen:
        answer, model = InkAIClient(
            "http://127.0.0.1:8000/api/sohbet"
        ).send_message("Bir fonksiyon yaz", [], on_chunk=received.append)

    request = urlopen.call_args.args[0]
    assert json.loads(request.data)["stream"] is True
    assert request.get_header("Accept") == "application/x-ndjson"
    assert received == ["def add(", "a, b): return a + b"]
    assert answer == "def add(a, b): return a + b"
    assert model == "coder"


def test_interrupted_stream_partial_answer_is_saved_to_conversation():
    saved_messages = []
    displayed_errors = []
    app = SimpleNamespace(
        busy=True,
        conversation_id=42,
        messages=[("user", "Question")],
        _stream_body=object(),
        _stream_full_text="Partial answer",
        store=SimpleNamespace(
            add_message=lambda conversation_id, role, content: saved_messages.append(
                (conversation_id, role, content)
            )
        ),
        _clear_activity=lambda: None,
        _finish_stream_message=lambda: None,
        _refresh_conversation_list=lambda: None,
        _add_message=lambda role, content, is_error=False: displayed_errors.append(
            (role, content, is_error)
        ),
        status_text=SimpleNamespace(set=lambda _text: None),
    )

    InkAIApp._finish_request(app, error="stream interrupted")

    assert saved_messages == [(42, "assistant", "Partial answer")]
    assert app.messages[-1] == ("assistant", "Partial answer")
    assert displayed_errors == [("assistant", "stream interrupted", True)]


def test_interrupted_stream_saves_chunks_not_yet_rendered_by_gui():
    saved_messages = []
    displayed_messages = []
    app = SimpleNamespace(
        busy=True,
        conversation_id=42,
        messages=[("user", "Question")],
        _stream_body=None,
        store=SimpleNamespace(
            add_message=lambda conversation_id, role, content: saved_messages.append(
                (conversation_id, role, content)
            )
        ),
        _clear_activity=lambda: None,
        _enable_composer=lambda: None,
        _add_message=lambda role, content, is_error=False: displayed_messages.append(
            (role, content, is_error)
        ),
        _refresh_conversation_list=lambda: None,
        status_text=SimpleNamespace(set=lambda _text: None),
    )

    InkAIApp._finish_request(
        app,
        error="stream interrupted",
        partial_answer="Buffered but not rendered",
    )

    assert saved_messages == [(42, "assistant", "Buffered but not rendered")]
    assert app.messages[-1] == ("assistant", "Buffered but not rendered")
    assert displayed_messages == [
        ("assistant", "Buffered but not rendered", False),
        ("assistant", "stream interrupted", True),
    ]


def test_gui_restores_the_latest_saved_conversation():
    opened = []
    app = SimpleNamespace(
        store=SimpleNamespace(list_conversations=lambda: [(17, "Latest"), (8, "Older")]),
        _open_conversation=opened.append,
    )

    InkAIApp._restore_latest_conversation(app)

    assert opened == [17]


def test_gui_client_downloads_automatically_generated_floppy_image():
    class ImageResponse(FakeResponse):
        def __init__(self, payload, headers=None):
            super().__init__({})
            self.payload = payload
            self.headers = headers or {}

        def read(self):
            return self.payload

    responses = [
        ImageResponse(
            json.dumps(
                {
                    "yanit": "Floppy görseli oluşturdu.",
                    "image_filename": "created.png",
                }
            ).encode(),
            {"X-Ink-AI-Model": "floppy"},
        ),
        ImageResponse(b"\x89PNG generated image"),
    ]
    client = InkAIClient("http://127.0.0.1:8000/api/sohbet")

    with patch("ink_ai.gui.urlopen", side_effect=responses) as urlopen:
        answer, model = client.send_message(
            "Bir logo oluştur", [], progress_id="abc123"
        )

    assert answer == "Floppy görseli oluşturdu."
    assert model == "floppy"
    assert client.last_generated_image == (b"\x89PNG generated image", "created.png")
    assert urlopen.call_args_list[1].args[0].full_url.endswith(
        "/api/gorsel/created.png"
    )


def test_gui_client_fetches_generated_image_from_local_api():
    class ImageResponse:
        status = 200

        def __init__(self, payload):
            self.payload = payload
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.payload

    responses = [
        ImageResponse(json.dumps({"filename": "generated.png"}).encode("utf-8")),
        ImageResponse(b"\x89PNG image"),
    ]

    with patch("ink_ai.gui.urlopen", side_effect=responses) as urlopen:
        image, filename = InkAIClient(
            "http://127.0.0.1:8000/api/sohbet"
        ).generate_image("Make a green logo")

    assert filename == "generated.png"
    assert image == b"\x89PNG image"
    assert urlopen.call_count == 2
    assert urlopen.call_args_list[1].args[0].full_url.endswith(
        "/api/gorsel/generated.png"
    )


def test_gui_splits_code_and_terminal_blocks_for_styled_messages():
    answer = (
        "Önce kod:\n```python\nprint('merhaba')\n```\n"
        "Sonra terminal:\n```bash\nuv run ink-ai\n```"
    )

    assert split_answer_segments(answer) == [
        ("text", "Önce kod:\n"),
        ("code", "print('merhaba')\n"),
        ("text", "\nSonra terminal:\n"),
        ("terminal", "uv run ink-ai\n"),
    ]
    assert InkAIApp._message_tokens(split_answer_segments(answer)) == [
        ("Önce kod:\n", None),
        ("●  ●  ●   code\n", "code_header"),
        ("print('merhaba')\n", "code"),
        ("\nSonra terminal:\n", None),
        ("●  ●  ●   terminal — $ \n", "code_header"),
        ("uv run ink-ai\n", "terminal"),
    ]


def test_gui_keeps_unformatted_answers_as_plain_text():
    assert split_answer_segments("Merhaba, yardımcı olayım.") == [
        ("text", "Merhaba, yardımcı olayım.")
    ]


def test_gui_defaults_to_neon_black_and_white_theme(tmp_path, monkeypatch):
    monkeypatch.setattr(
        InkAIApp,
        "_theme_preference_path",
        staticmethod(lambda: tmp_path / "missing-preferences.json"),
    )

    assert InkAIApp._load_theme_preference() == "dark"
