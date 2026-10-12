from __future__ import annotations

import base64
import json
import math
import re
import threading
import tkinter as tk
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import customtkinter as ctk

from ink_ai.conversations import ConversationStore, default_database_path
from ink_ai.documents import (
    MAX_DOCUMENT_CHARS,
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    read_document,
)

BACKGROUND = ("#F4F5F8", "#050507")
PANEL = ("#FFFFFF", "#0C0C10")
PANEL_HOVER = ("#F0F1F5", "#18181F")
BORDER = ("#D9DBE3", "#292931")
GREEN = ("#17171C", "#FFFFFF")
GREEN_BRIGHT = ("#35353D", "#F2F2F7")
GREEN_DARK = ("#E7E8ED", "#222229")
BUTTON_TEXT = ("#FFFFFF", "#050507")
WHITE = ("#17171C", "#F5F5F8")
MUTED = ("#686A75", "#A3A3AE")
ERROR = ("#B94038", "#FF8076")
CARD_BACKGROUND = ("#F8F8FA", "#0C0C10")
INPUT_BACKGROUND = ("#F8F8FA", "#0C0C10")
USER_BUBBLE = ("#ECEEF3", "#141419")
USER_BUBBLE_BORDER = ("#D2D4DC", "#35353D")
USER_HEADING = ("#35353D", "#F5F5F8")
ASSISTANT_HEADING = ("#35353D", "#FFFFFF")
TERMINAL_BACKGROUND = "#08080B"
TERMINAL_TEXT = "#F2F2F7"
TERMINAL_HEADER = "#1B1B22"
FILE_MARKER = "\n\n[Ek dosya: "
MAX_CHAT_FILE_CHARS = MAX_DOCUMENT_CHARS
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROMPT_PLACEHOLDER = "Mesajını yaz veya görsel / PDF yapıştır…"

DEFAULT_API_URL = "http://127.0.0.1:8000/api/sohbet"
SUGGESTIONS = (
    ("✧", "Bir fikri keşfet", "Yapay zekâ günlük hayatımızı nasıl değiştiriyor?"),
    ("⌘", "Kod yaz", "Python ile klasördeki dosyaları listeleyen bir kod yaz."),
    ("✎", "Metin oluştur", "Yeni bir projeye başlamak için motive edici bir metin yaz."),
    ("◈", "Bir konuyu öğren", "Kuantum bilgisayarı basitçe ve örneklerle anlat."),
)
TERMINAL_LANGUAGES = {"bash", "sh", "zsh", "shell", "console", "terminal"}
POWERSHELL_LANGUAGES = {"powershell", "pwsh", "ps1"}


def split_answer_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    pattern = re.compile(r"```([\w.+#-]*)[ \t]*\n(.*?)```", re.DOTALL)
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            segments.append(("text", text[position : match.start()]))
        language = match.group(1).casefold()
        kind = (
            "terminal"
            if language in TERMINAL_LANGUAGES | POWERSHELL_LANGUAGES
            else "code"
        )
        segments.append((kind, match.group(2)))
        position = match.end()
    if position < len(text):
        segments.append(("text", text[position:]))
    return segments or [("text", text)]


class InkAIClient:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url.strip()
        self.last_generated_image: tuple[bytes, str] | None = None
        parsed = urlsplit(self.api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API adresi http:// veya https:// ile başlamalı.")
        self.health_url = urlunsplit((parsed.scheme, parsed.netloc, "/api/saglik", "", ""))

    def check_health(self) -> dict[str, Any]:
        request = Request(self.health_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise ConnectionError(f"API sağlık kontrolü başarısız: HTTP {response.status}")
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
                raise TypeError("API sağlık yanıtı beklenen biçimde değil.")
        return data

    def send_message(
        self,
        prompt: str,
        history: list[tuple[str, str]],
        files: list[dict[str, str]] | None = None,
        model: str = "auto",
        images: list[dict[str, str]] | None = None,
        web_search: bool = False,
        progress_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        payload = {
            "mesaj": prompt,
            "gecmis": [
                {
                    "rol": "kullanici" if role == "user" else "asistan",
                    "icerik": message,
                }
                for role, message in history[-20:]
            ],
            "files": files or [],
            "images": images or [],
            "web_search": web_search,
            "model": model,
        }
        if on_chunk is not None:
            payload["stream"] = True
        if progress_id is not None:
            payload["progress_id"] = progress_id
        request = Request(
            self.api_url,
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": (
                    "application/x-ndjson" if on_chunk is not None else "application/json"
                ),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                if on_chunk is None:
                    data = json.loads(response.read().decode("utf-8"))
                    used_model = response.headers.get("X-Ink-AI-Model", model)
                    sources = response.headers.get("X-Ink-AI-Sources")
                else:
                    final_event = None
                    while line := response.readline():
                        event = json.loads(line.decode("utf-8"))
                        if not isinstance(event, dict) or not isinstance(
                            event.get("type"), str
                        ):
                            raise TypeError("API akışında geçersiz olay alındı.")
                        if event["type"] == "token":
                            chunk = event.get("content")
                            if not isinstance(chunk, str):
                                raise TypeError("API akışında geçersiz metin parçası alındı.")
                            on_chunk(chunk)
                        elif event["type"] == "error":
                            raise RuntimeError(
                                str(event.get("detail", "Akış yanıtı başarısız oldu."))
                            )
                        elif event["type"] == "done":
                            final_event = event
                            break
                        elif event["type"] == "heartbeat":
                            continue
                        else:
                            raise ValueError(
                                f"API akışında bilinmeyen olay: {event['type']}"
                            )
                    if final_event is None:
                        raise ValueError("API yanıt akışı tamamlanmadan kapandı.")
                    data = {
                        "yanit": final_event.get("answer"),
                        "image_filename": final_event.get("image_filename"),
                    }
                    used_model = str(final_event.get("model", model))
                    sources = final_event.get("sources")
            image_name = data.get("image_filename")
            if image_name is not None:
                self.last_generated_image = self.fetch_generated_image(image_name)
                used_model = "floppy"
            else:
                self.last_generated_image = None
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = str(json.loads(detail).get("detail", detail))
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"API isteği başarısız (HTTP {error.code}): {detail}") from error
        except URLError as error:
            raise ConnectionError(f"API'ye bağlanılamadı: {error.reason}") from error
        except TimeoutError as error:
            raise TimeoutError("Model yanıtı zaman aşımına uğradı.") from error

        answer = data.get("yanit")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("API yanıtında geçerli bir 'yanit' alanı bulunamadı.")
        if sources:
            try:
                source_items = json.loads(sources)
            except json.JSONDecodeError as error:
                raise ValueError("API kaynak başlığı geçerli JSON değil.") from error
            if not isinstance(source_items, list):
                raise ValueError("API kaynak başlığı beklenen biçimde değil.")
            names = [
                str(item["source"])
                for item in source_items
                if isinstance(item, dict) and isinstance(item.get("source"), str)
            ]
            if names:
                source_text = "\n\nKaynaklar:\n" + "\n".join(
                    f"- {name}" for name in names
                )
                answer = answer.strip() + source_text
                if on_chunk is not None:
                    on_chunk(source_text)
        return answer.strip(), used_model

    def get_chat_progress(self, progress_id: str) -> dict[str, Any]:
        parsed = urlsplit(self.api_url)
        status_url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"/api/sohbet/durum/{progress_id}",
                "",
                "",
            )
        )
        request = Request(status_url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"Sohbet durumu alınamadı (HTTP {error.code})."
            ) from error
        except URLError as error:
            raise ConnectionError(
                f"Sohbet durumu alınamadı: {error.reason}"
            ) from error
        if not isinstance(data, dict) or not isinstance(data.get("stage"), str):
            raise TypeError("Sohbet durumu beklenen biçimde değil.")
        if not isinstance(data.get("message"), str):
            raise TypeError("Sohbet durumu mesajı beklenen biçimde değil.")
        return data

    def generate_image(self, prompt: str) -> tuple[bytes, str]:
        parsed = urlsplit(self.api_url)
        request = Request(
            urlunsplit((parsed.scheme, parsed.netloc, "/api/gorsel", "", "")),
            data=json.dumps({"prompt": prompt}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
            image_name = data["filename"]
            return self.fetch_generated_image(image_name)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = str(json.loads(detail).get("detail", detail))
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"Görsel isteği başarısız (HTTP {error.code}): {detail}") from error
        except URLError as error:
            raise ConnectionError(f"API'ye bağlanılamadı: {error.reason}") from error

    def fetch_generated_image(self, image_name: str) -> tuple[bytes, str]:
        if Path(image_name).name != image_name or not image_name.endswith(".png"):
            raise ValueError("API geçersiz görsel dosya adı döndürdü.")
        parsed = urlsplit(self.api_url)
        image_request = Request(
            urlunsplit(
                (parsed.scheme, parsed.netloc, f"/api/gorsel/{image_name}", "", "")
            ),
            headers={"Accept": "image/png"},
        )
        try:
            with urlopen(image_request, timeout=30) as response:
                image_bytes = response.read()
        except HTTPError as error:
            raise RuntimeError(
                f"Oluşturulan görsel alınamadı (HTTP {error.code})."
            ) from error
        except URLError as error:
            raise ConnectionError(f"Görsel alınamadı: {error.reason}") from error
        return image_bytes, image_name


class InkAIApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.theme_mode = self._load_theme_preference()
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")

        self.title("Ink AI  |  Yerel Yapay Zekâ")
        self.geometry("1280x820")
        self.minsize(960, 640)
        self.configure(fg_color=BACKGROUND)

        self.messages: list[tuple[str, str]] = []
        self.store = ConversationStore()
        self.conversation_id: int | None = None
        self.pending_files: list[dict[str, str]] = []
        self.active_files: list[dict[str, str]] = []
        self.pending_images: list[dict[str, str]] = []
        self.active_images: list[dict[str, str]] = []
        self.image_refs: list[ctk.CTkImage] = []
        self.busy = False
        self._active_progress_id: str | None = None
        self._activity_frame: ctk.CTkFrame | None = None
        self._activity_icon: ctk.CTkLabel | None = None
        self._activity_message: ctk.CTkLabel | None = None
        self._stream_body: ctk.CTkTextbox | None = None
        self._stream_bubble: ctk.CTkFrame | None = None
        self._stream_pending = ""
        self._stream_mode = "text"
        self._stream_code_kind = "code"
        self._stream_update_count = 0
        self._stream_full_text = ""
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)
        self.model_choice = tk.StringVar(value="Otomatik")
        self.theme_choice = tk.StringVar(
            value="Neon siyah" if self.theme_mode == "dark" else "Açık"
        )
        self.status_text = tk.StringVar(value="API kontrol ediliyor…")
        self.conversation_search = tk.StringVar()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_workspace()
        self._show_welcome()
        self._refresh_conversation_list()
        self._restore_latest_conversation()
        self._check_health()
        self.after(150, self._focus_prompt_box)

    @staticmethod
    def _theme_preference_path() -> Path:
        return default_database_path().parent / "gui_preferences.json"

    @classmethod
    def _load_theme_preference(cls) -> str:
        try:
            preferences = json.loads(
                cls._theme_preference_path().read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return "dark"
        except (OSError, json.JSONDecodeError):
            return "dark"
        theme = preferences.get("theme") if isinstance(preferences, dict) else None
        return theme if theme in {"light", "dark"} else "dark"

    def _change_theme(self, choice: str) -> None:
        self.theme_mode = "dark" if choice == "Neon siyah" else "light"
        ctk.set_appearance_mode(self.theme_mode)
        try:
            self._theme_preference_path().write_text(
                json.dumps({"theme": self.theme_mode}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as error:
            self.status_text.set(f"Tema tercihi kaydedilemedi: {error}")

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(
            self, width=258, corner_radius=0, fg_color=PANEL, border_width=0
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(7, weight=1, minsize=160)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=22, pady=(25, 32))
        ctk.CTkLabel(
            brand,
            text="✧",
            width=42,
            height=42,
            corner_radius=13,
            fg_color=GREEN,
            text_color=BACKGROUND,
            font=ctk.CTkFont(size=23, weight="bold"),
        ).pack(side="left")
        brand_words = ctk.CTkFrame(brand, fg_color="transparent")
        brand_words.pack(side="left", padx=(11, 0))
        ctk.CTkLabel(
            brand_words,
            text="INK AI",
            text_color=WHITE,
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand_words,
            text="YEREL ASİSTAN",
            text_color=MUTED,
            font=ctk.CTkFont(size=9, weight="bold"),
        ).pack(anchor="w", pady=(1, 0))

        self.new_chat_button = ctk.CTkButton(
            sidebar,
            text="＋    Yeni sohbet",
            height=43,
            corner_radius=12,
            anchor="w",
            fg_color=GREEN_DARK,
            hover_color=GREEN_BRIGHT,
            text_color=BUTTON_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._new_chat,
        )
        self.new_chat_button.grid(row=1, column=0, sticky="ew", padx=16)

        ctk.CTkLabel(
            sidebar,
            text="ÇALIŞMA ALANI",
            text_color=MUTED,
            font=ctk.CTkFont(size=9, weight="bold"),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=23, pady=(33, 10))

        status_card = ctk.CTkFrame(
            sidebar, fg_color=CARD_BACKGROUND, corner_radius=13, border_width=1, border_color=BORDER
        )
        status_card.grid(row=3, column=0, sticky="ew", padx=15)
        ctk.CTkLabel(
            status_card,
            text="◉",
            text_color=GREEN,
            font=ctk.CTkFont(size=18),
        ).pack(side="left", padx=(13, 9), pady=13)
        status_copy = ctk.CTkFrame(status_card, fg_color="transparent")
        status_copy.pack(side="left", fill="x", expand=True, pady=10)
        ctk.CTkLabel(
            status_copy,
            text="Yerel bağlantı",
            text_color=WHITE,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            status_copy,
            textvariable=self.status_text,
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", pady=(2, 0))

        ctk.CTkLabel(
            sidebar,
            text="API ADRESİ",
            text_color=MUTED,
            font=ctk.CTkFont(size=9, weight="bold"),
            anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=23, pady=(28, 9))
        self.api_entry = ctk.CTkEntry(
            sidebar,
            textvariable=self.api_url,
            height=37,
            corner_radius=9,
            border_color=BORDER,
            fg_color=INPUT_BACKGROUND,
            text_color=WHITE,
            font=ctk.CTkFont(size=10),
        )
        self.api_entry.grid(row=5, column=0, sticky="ew", padx=15)
        ctk.CTkButton(
            sidebar,
            text="Bağlantıyı denetle",
            height=34,
            corner_radius=9,
            fg_color="transparent",
            hover_color=PANEL_HOVER,
            border_width=1,
            border_color=BORDER,
            text_color=WHITE,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._check_health,
        ).grid(row=6, column=0, sticky="ew", padx=15, pady=(9, 0))

        history_panel = ctk.CTkFrame(sidebar, fg_color="transparent")
        history_panel.grid(row=7, column=0, sticky="nsew", padx=10, pady=(19, 0))
        history_panel.grid_columnconfigure(0, weight=1)
        history_panel.grid_rowconfigure(2, weight=1, minsize=120)
        ctk.CTkLabel(
            history_panel,
            text="SOHBET GEÇMİŞİ",
            text_color=MUTED,
            font=ctk.CTkFont(size=9, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=13, pady=(0, 7))
        self.history_list = ctk.CTkScrollableFrame(
            history_panel,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=GREEN_DARK,
        )
        self.history_list.grid(row=2, column=0, sticky="nsew", pady=(7, 0))
        self.history_list.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(
            history_panel,
            placeholder_text="⌕  Sohbetlerde ara",
            textvariable=self.conversation_search,
            height=34,
            corner_radius=10,
            border_color=BORDER,
            fg_color=INPUT_BACKGROUND,
            text_color=WHITE,
            font=ctk.CTkFont(size=11),
        )
        self.search_entry.grid(row=1, column=0, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.grid(row=8, column=0, sticky="sew", padx=20, pady=19)
        ctk.CTkFrame(footer, height=1, fg_color=BORDER).pack(fill="x", pady=(0, 13))
        ctk.CTkLabel(
            footer,
            text="◈  Verilerin bu bilgisayarda kalır",
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w")
        ctk.CTkLabel(
            footer,
            text="INK AI  •  LOCAL FIRST",
            text_color=MUTED,
            font=ctk.CTkFont(size=9, weight="bold"),
        ).pack(anchor="w", pady=(8, 0))

    def _build_workspace(self) -> None:
        workspace = ctk.CTkFrame(self, fg_color=BACKGROUND, corner_radius=0)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            workspace, height=68, fg_color=BACKGROUND, corner_radius=0
        )
        header.grid(row=0, column=0, sticky="ew", padx=34)
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Sohbet",
            text_color=WHITE,
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=21)
        self.model_selector = ctk.CTkOptionMenu(
            header,
            values=["Otomatik", "Flash", "Reasoning", "Coder"],
            variable=self.model_choice,
            width=128,
            height=34,
            corner_radius=10,
            fg_color=PANEL,
            button_color=GREEN_DARK,
            button_hover_color=GREEN_DARK,
            text_color=WHITE,
            font=ctk.CTkFont(size=10, weight="bold"),
            dropdown_fg_color=PANEL,
            dropdown_text_color=WHITE,
            dropdown_hover_color=GREEN_DARK,
        )
        self.model_selector.grid(row=0, column=1, sticky="e", padx=(0, 10), pady=15)
        self.model_badge = ctk.CTkLabel(
            header,
            text="✦  OTOMATİK UZMAN MODEL",
            width=180,
            height=34,
            fg_color=PANEL,
            text_color=WHITE,
            corner_radius=10,
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self.model_badge.grid(row=0, column=2, sticky="e", padx=(0, 12), pady=15)
        self.local_badge = ctk.CTkLabel(
            header,
            text="●  YEREL",
            text_color=GREEN,
            font=ctk.CTkFont(size=9, weight="bold"),
        )
        self.local_badge.grid(row=0, column=4, sticky="e", pady=21)
        self.theme_selector = ctk.CTkOptionMenu(
            header,
            values=["Açık", "Neon siyah"],
            variable=self.theme_choice,
            width=108,
            height=34,
            corner_radius=10,
            fg_color=PANEL,
            button_color=GREEN_DARK,
            button_hover_color=GREEN_DARK,
            text_color=WHITE,
            font=ctk.CTkFont(size=10, weight="bold"),
            dropdown_fg_color=PANEL,
            dropdown_text_color=WHITE,
            dropdown_hover_color=GREEN_DARK,
            command=self._change_theme,
        )
        self.theme_selector.grid(row=0, column=3, sticky="e", padx=(0, 12), pady=15)
        ctk.CTkFrame(workspace, height=1, fg_color=BORDER).grid(
            row=0, column=0, sticky="sew", padx=34
        )

        self.chat_area = ctk.CTkScrollableFrame(
            workspace,
            fg_color=BACKGROUND,
            corner_radius=0,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=GREEN_BRIGHT,
        )
        self.chat_area.grid(row=1, column=0, sticky="nsew", padx=26, pady=(15, 10))
        self.chat_area.grid_columnconfigure(0, weight=1)

        composer = ctk.CTkFrame(workspace, fg_color=BACKGROUND, corner_radius=0)
        composer.grid(row=2, column=0, sticky="ew", padx=34, pady=(0, 22))
        composer.grid_columnconfigure(0, weight=1)
        composer.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            composer,
            text="Uzman modele göre web araştırması otomatik",
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
        ).grid(row=0, column=0, sticky="w", padx=5, pady=(0, 7))
        input_shell = ctk.CTkFrame(
            composer,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        input_shell.grid(row=1, column=0, columnspan=2, sticky="ew")
        input_shell.grid_columnconfigure(1, weight=1)
        input_shell.grid_rowconfigure(0, weight=1)
        self.attach_button = ctk.CTkButton(
            input_shell,
            text="＋",
            width=42,
            height=40,
            corner_radius=14,
            fg_color=GREEN_DARK,
            hover_color=GREEN_BRIGHT,
            text_color=BUTTON_TEXT,
            font=ctk.CTkFont(size=21, weight="bold"),
            command=self._select_files,
        )
        self.attach_button.grid(row=1, column=0, sticky="sw", padx=(12, 5), pady=(2, 11))
        self.prompt_box = ctk.CTkTextbox(
            input_shell,
            height=58,
            corner_radius=12,
            fg_color="transparent",
            text_color=WHITE,
            border_width=0,
            wrap="word",
            font=ctk.CTkFont(size=13),
            activate_scrollbars=False,
        )
        self.prompt_box.grid(
            row=0, column=0, columnspan=3, sticky="nsew", padx=(15, 14), pady=(10, 0)
        )
        self.prompt_box.insert("1.0", PROMPT_PLACEHOLDER)
        self.prompt_box.configure(text_color=MUTED)
        self.prompt_box._textbox.bind("<KeyRelease>", self._update_prompt_placeholder)
        self.prompt_box._textbox.bind("<FocusIn>", self._on_prompt_focus_in)
        self.prompt_box._textbox.bind("<FocusOut>", self._on_prompt_focus_out)
        self.prompt_box._textbox.bind("<Control-Return>", self._send_from_keyboard)
        self.prompt_box._textbox.bind("<Control-v>", self._paste_clipboard)
        self.prompt_box._textbox.bind("<Control-V>", self._paste_clipboard)
        self.send_button = ctk.CTkButton(
            input_shell,
            text="↑",
            width=48,
            height=42,
            corner_radius=14,
            fg_color=GREEN,
            hover_color=GREEN_BRIGHT,
            text_color=BUTTON_TEXT,
            font=ctk.CTkFont(size=23, weight="bold"),
            command=self._send_message,
        )
        self.send_button.grid(row=1, column=2, sticky="se", padx=(5, 12), pady=(2, 11))
        self.attachment_status = ctk.CTkLabel(
            composer,
            text="",
            text_color=MUTED,
            font=ctk.CTkFont(size=9),
            anchor="w",
        )
        self.attachment_status.grid(row=2, column=0, sticky="w", padx=5, pady=(6, 0))
        ctk.CTkLabel(
            composer,
            text="Enter ile gönder  ·  Shift + Enter yeni satır",
            text_color=MUTED,
            font=ctk.CTkFont(size=9),
        ).grid(row=2, column=1, sticky="e", padx=4, pady=(7, 0))
        self.prompt_box._textbox.bind("<Return>", self._send_from_keyboard)
        self.prompt_box._textbox.bind("<Shift-Return>", lambda _event: None)

    def _on_prompt_focus_in(self, _event: tk.Event[Any] | None = None) -> None:
        if self.prompt_box.get("1.0", "end-1c") == PROMPT_PLACEHOLDER:
            self.prompt_box.delete("1.0", "end")
            self.prompt_box.configure(text_color=WHITE)

    def _focus_prompt_box(self) -> None:
        self.prompt_box._textbox.focus_force()

    def _on_prompt_focus_out(self, _event: tk.Event[Any] | None = None) -> None:
        if not self.prompt_box.get("1.0", "end-1c").strip():
            self.prompt_box.insert("1.0", PROMPT_PLACEHOLDER)
            self.prompt_box.configure(text_color=MUTED)

    def _update_prompt_placeholder(self, _event: tk.Event[Any] | None = None) -> None:
        if self.prompt_box.get("1.0", "end-1c") != PROMPT_PLACEHOLDER:
            self.prompt_box.configure(text_color=WHITE)

    def _show_welcome(self) -> None:
        welcome = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        welcome.grid(row=0, column=0, sticky="nsew", pady=(75, 30))
        welcome.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            welcome,
            text="✧",
            width=68,
            height=68,
            corner_radius=22,
            fg_color=GREEN_DARK,
            text_color=GREEN,
            font=ctk.CTkFont(size=34, weight="bold"),
        ).grid(row=0, column=0, pady=(0, 22))
        ctk.CTkLabel(
            welcome,
            text="Merhaba, ben Ink AI.",
            text_color=WHITE,
            font=ctk.CTkFont(size=29, weight="bold"),
        ).grid(row=1, column=0)
        ctk.CTkLabel(
            welcome,
            text="Aklındaki her şeyi sor. Birlikte güzel şeyler üretelim.",
            text_color=MUTED,
            font=ctk.CTkFont(size=13),
        ).grid(row=2, column=0, pady=(8, 27))

        cards = ctk.CTkFrame(welcome, fg_color="transparent")
        cards.grid(row=3, column=0)
        for index, (icon, title, prompt) in enumerate(SUGGESTIONS):
            card = ctk.CTkButton(
                cards,
                text=f"{icon}   {title}\n{prompt}",
                width=245,
                height=77,
                corner_radius=13,
                anchor="w",
                fg_color=PANEL,
                hover_color=PANEL_HOVER,
                border_width=1,
                border_color=BORDER,
                text_color=WHITE,
                font=ctk.CTkFont(size=11),
                command=lambda value=prompt: self._send_message(value),
            )
            card.grid(row=index // 2, column=index % 2, padx=6, pady=6)

    def _add_message(
        self,
        role: str,
        text: str,
        *,
        is_error: bool = False,
        image: ctk.CTkImage | None = None,
        animate: bool = False,
    ) -> None:
        row = len(self.chat_area.winfo_children())
        wrapper = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=8, pady=7)
        wrapper.grid_columnconfigure(0, weight=1)

        is_user = role == "user"
        bubble_color = USER_BUBBLE if is_user else PANEL
        border_color = USER_BUBBLE_BORDER if is_user else BORDER
        if is_error:
            bubble_color = ("#FFF1EF", "#321C1A")
            border_color = ("#F0C5C0", "#63332E")

        bubble = ctk.CTkFrame(
            wrapper,
            fg_color=bubble_color,
            corner_radius=15,
            border_width=1,
            border_color=border_color,
        )
        bubble.grid(row=0, column=0, sticky="e" if is_user else "w", padx=(100, 0) if is_user else (0, 100))
        bubble.grid_columnconfigure(0, weight=1)
        heading = "SEN" if is_user else ("BAĞLANTI HATASI" if is_error else "✧  INK AI")
        heading_color = USER_HEADING if is_user else (ERROR if is_error else ASSISTANT_HEADING)
        wraplength = max(280, min(620, self.winfo_width() - 530))
        ctk.CTkLabel(
            bubble,
            text=heading,
            text_color=heading_color,
            font=ctk.CTkFont(size=9, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=15, pady=(11, 3))
        content = self._message_preview(text)
        segments = split_answer_segments(content) if not is_user else [("text", content)]
        has_code = any(kind != "text" for kind, _ in segments)
        if not is_user and not is_error and (has_code or animate):
            body = ctk.CTkTextbox(
                bubble,
                height=45,
                corner_radius=0,
                fg_color="transparent",
                text_color=WHITE,
                border_width=0,
                wrap="word",
                font=ctk.CTkFont(size=12),
                activate_scrollbars=False,
            )
            body.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
            self._configure_message_tags(body)
            tokens = self._message_tokens(segments)
            if animate:
                self._animate_message(body, tokens)
            else:
                self._insert_message_tokens(body, tokens)
                self._fit_message_text(body)
                body.configure(state="disabled")
        else:
            message_label = ctk.CTkLabel(
                bubble,
                text=content,
                text_color=WHITE if not is_error else ERROR,
                font=ctk.CTkFont(size=12),
                justify="left",
                anchor="w",
                wraplength=wraplength,
            )
            message_label.grid(row=1, column=0, sticky="ew", padx=15, pady=(2, 13))
        if image is not None:
            image_label = ctk.CTkLabel(bubble, text="", image=image)
            image_label.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 10))
            self.image_refs.append(image)
        ctk.CTkButton(
            bubble,
            text="Kopyala",
            width=66,
            height=25,
            corner_radius=7,
            fg_color="transparent",
            hover_color=PANEL_HOVER,
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
            command=lambda value=text: self._copy_text(value),
        ).grid(row=3 if image is not None else 2, column=0, sticky="e", padx=10, pady=(0, 7))
        self.chat_area.after(40, self._scroll_to_bottom)

    @staticmethod
    def _configure_message_tags(textbox: ctk.CTkTextbox) -> None:
        text_widget = textbox._textbox
        text_widget.tag_configure(
            "code",
            background=TERMINAL_BACKGROUND,
            foreground="#F2F2F7",
            font=("Consolas", 11),
            lmargin1=12,
            lmargin2=12,
            rmargin=12,
            spacing1=8,
            spacing3=8,
        )
        text_widget.tag_configure(
            "terminal",
            background=TERMINAL_BACKGROUND,
            foreground=TERMINAL_TEXT,
            font=("Consolas", 11),
            lmargin1=12,
            lmargin2=12,
            rmargin=12,
            spacing1=8,
            spacing3=8,
        )
        text_widget.tag_configure(
            "code_header",
            background=TERMINAL_HEADER,
            foreground="#B8B8C2",
            font=("Consolas", 9),
            lmargin1=12,
            lmargin2=12,
            rmargin=12,
            spacing1=7,
            spacing3=5,
        )

    def _start_stream_message(self) -> None:
        self._clear_activity()
        wrapper = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        wrapper.grid(
            row=len(self.chat_area.winfo_children()),
            column=0,
            sticky="ew",
            padx=8,
            pady=7,
        )
        wrapper.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkFrame(
            wrapper,
            fg_color=PANEL,
            corner_radius=15,
            border_width=1,
            border_color=BORDER,
        )
        bubble.grid(row=0, column=0, sticky="w", padx=(0, 100))
        bubble.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            bubble,
            text="✧  INK AI",
            text_color=ASSISTANT_HEADING,
            font=ctk.CTkFont(size=9, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=15, pady=(11, 3))
        body = ctk.CTkTextbox(
            bubble,
            height=45,
            corner_radius=0,
            fg_color="transparent",
            text_color=WHITE,
            border_width=0,
            wrap="word",
            font=ctk.CTkFont(size=12),
            activate_scrollbars=False,
        )
        body.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self._configure_message_tags(body)
        self._stream_body = body
        self._stream_bubble = bubble
        self._stream_pending = ""
        self._stream_mode = "text"
        self._stream_update_count = 0
        self._stream_full_text = ""
        self.status_text.set("Yanıt yazılıyor…")

    def _append_stream_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        if self._stream_body is None:
            self._start_stream_message()
        self._stream_full_text += chunk
        self._stream_pending += chunk
        self._consume_stream_pending()
        self._stream_update_count += 1
        if self._stream_update_count % 4 == 0 and self._stream_body is not None:
            self._fit_message_text(self._stream_body)
            self._scroll_to_bottom()

    def _consume_stream_pending(self, *, final: bool = False) -> None:
        while self._stream_body is not None:
            if self._stream_mode == "text":
                opening = re.search(
                    r"```([\w.+#-]*)[ \t]*\n",
                    self._stream_pending,
                )
                if opening is not None:
                    self._emit_stream_text(self._stream_pending[: opening.start()])
                    language = opening.group(1).casefold()
                    self._stream_code_kind = (
                        "terminal"
                        if language in TERMINAL_LANGUAGES | POWERSHELL_LANGUAGES
                        else "code"
                    )
                    header = (
                        "●  ●  ●   terminal — $ \n"
                        if self._stream_code_kind == "terminal"
                        else "●  ●  ●   code\n"
                    )
                    self._emit_stream_text(header, "code_header")
                    self._stream_pending = self._stream_pending[opening.end() :]
                    self._stream_mode = "code"
                    continue
                if final:
                    self._emit_stream_text(self._stream_pending)
                    self._stream_pending = ""
                elif len(self._stream_pending) > 40:
                    safe_length = len(self._stream_pending) - 40
                    self._emit_stream_text(self._stream_pending[:safe_length])
                    self._stream_pending = self._stream_pending[safe_length:]
                break

            closing = self._stream_pending.find("```")
            if closing >= 0:
                code = self._stream_pending[:closing]
                self._emit_stream_text(code, self._stream_code_kind)
                if code and not code.endswith("\n"):
                    self._emit_stream_text("\n", self._stream_code_kind)
                self._stream_pending = self._stream_pending[closing + 3 :]
                self._stream_mode = "text"
                continue
            if final:
                self._emit_stream_text(self._stream_pending, self._stream_code_kind)
                self._stream_pending = ""
                self._stream_mode = "text"
            elif len(self._stream_pending) > 2:
                safe_length = len(self._stream_pending) - 2
                self._emit_stream_text(
                    self._stream_pending[:safe_length],
                    self._stream_code_kind,
                )
                self._stream_pending = self._stream_pending[safe_length:]
            break

    def _emit_stream_text(self, content: str, tag: str | None = None) -> None:
        if not content or self._stream_body is None:
            return
        text_widget = self._stream_body._textbox
        if tag:
            text_widget.insert("end", content, tag)
        else:
            text_widget.insert("end", content)

    def _finish_stream_message(self, _answer: str | None = None) -> None:
        if self._stream_body is None:
            return
        if _answer is not None:
            self._stream_full_text = _answer
        self._consume_stream_pending(final=True)
        body = self._stream_body
        bubble = self._stream_bubble
        self._fit_message_text(body)
        body._textbox.configure(state="disabled")
        if bubble is not None:
            ctk.CTkButton(
                bubble,
                text="Kopyala",
                width=66,
                height=25,
                corner_radius=7,
                fg_color="transparent",
                hover_color=PANEL_HOVER,
                text_color=MUTED,
                font=ctk.CTkFont(size=10),
                command=lambda value=self._stream_full_text: self._copy_text(value),
            ).grid(row=2, column=0, sticky="e", padx=10, pady=(0, 7))
        self._stream_body = None
        self._stream_bubble = None
        self.busy = False
        self._enable_composer()
        self._scroll_to_bottom()

    @staticmethod
    def _message_tokens(segments: list[tuple[str, str]]) -> list[tuple[str, str | None]]:
        tokens: list[tuple[str, str | None]] = []
        for kind, content in segments:
            if kind == "text":
                tokens.append((content, None))
                continue
            header = "●  ●  ●   terminal — $ \n" if kind == "terminal" else "●  ●  ●   code\n"
            tokens.append((header, "code_header"))
            tokens.append((content, kind))
            if content and not content.endswith("\n"):
                tokens.append(("\n", kind))
        return tokens

    @staticmethod
    def _insert_message_tokens(
        textbox: ctk.CTkTextbox,
        tokens: list[tuple[str, str | None]],
    ) -> None:
        for content, tag in tokens:
            if content:
                if tag:
                    textbox._textbox.insert("end", content, tag)
                else:
                    textbox._textbox.insert("end", content)

    @staticmethod
    def _fit_message_text(textbox: ctk.CTkTextbox) -> None:
        line_count = textbox._textbox.count("1.0", "end-1c", "displaylines")
        visible_lines = int(line_count[0]) if line_count else 1
        textbox.configure(height=min(480, max(38, visible_lines * 19 + 10)))

    def _animate_message(
        self,
        textbox: ctk.CTkTextbox,
        tokens: list[tuple[str, str | None]],
    ) -> None:
        text_widget = textbox._textbox
        text_widget.configure(state="normal")
        total_chars = sum(len(content) for content, _tag in tokens)
        chunk_size = max(1, min(32, math.ceil(total_chars / 180)))
        token_index = 0
        token_offset = 0
        ticks = 0

        def type_next() -> None:
            nonlocal token_index, token_offset, ticks
            try:
                if not textbox.winfo_exists():
                    return
            except tk.TclError:
                return
            while token_index < len(tokens):
                content, tag = tokens[token_index]
                available = len(content) - token_offset
                if available <= 0:
                    token_index += 1
                    token_offset = 0
                    continue
                if tag == "code_header":
                    text_widget.insert("end", content, tag)
                    token_index += 1
                    token_offset = 0
                    continue
                length = min(
                    available,
                    1 if tag == "terminal" else (3 if tag == "code" else chunk_size),
                )
                piece = content[token_offset : token_offset + length]
                if tag:
                    text_widget.insert("end", piece, tag)
                else:
                    text_widget.insert("end", piece)
                token_offset += length
                if token_offset == len(content):
                    token_index += 1
                    token_offset = 0
                break
            ticks += 1
            if ticks % 8 == 0:
                self._fit_message_text(textbox)
                self._scroll_to_bottom()
            if token_index < len(tokens):
                delay = 20 if tokens[token_index][1] == "terminal" else 12
                self.after(delay, type_next)
            else:
                self._fit_message_text(textbox)
                text_widget.configure(state="disabled")
                self.busy = False
                self._enable_composer()
                self._scroll_to_bottom()

        type_next()

    def _copy_text(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self.status_text.set("Mesaj panoya kopyalandı")

    @staticmethod
    def _message_preview(text: str) -> str:
        lines = text.splitlines()
        preview: list[str] = []
        for line in lines:
            if line.startswith("[Ek dosya: ") and line.endswith("]"):
                preview.append(f"📎  {line[len('[Ek dosya: '):-1]}")
            else:
                preview.append(line)
        return "\n".join(preview).strip()

    def _scroll_to_bottom(self) -> None:
        self.chat_area._parent_canvas.yview_moveto(1.0)

    def _refresh_conversation_list(self) -> None:
        for child in self.history_list.winfo_children():
            child.destroy()
        for conversation_id, title in self.store.search_conversations(
            self.conversation_search.get()
        ):
            ctk.CTkButton(
                self.history_list,
                text=f"◷  {title}",
                height=34,
                corner_radius=8,
                anchor="w",
                fg_color=GREEN_DARK if conversation_id == self.conversation_id else "transparent",
                hover_color=PANEL_HOVER,
                text_color=WHITE,
                font=ctk.CTkFont(size=10),
                command=lambda selected_id=conversation_id: self._open_conversation(
                    selected_id
                ),
            ).pack(fill="x", pady=2)

    def _on_search_changed(self, _event: tk.Event[Any]) -> None:
        self._refresh_conversation_list()

    def _open_conversation(self, conversation_id: int) -> None:
        if self.busy:
            return
        self.conversation_id = conversation_id
        self.messages = self.store.get_messages(conversation_id)
        self.active_files = self.store.get_files(conversation_id)
        self.active_images = self.store.get_images(conversation_id)
        self.pending_files.clear()
        self.pending_images.clear()
        self._update_attachment_status()
        self._render_messages()
        self._refresh_conversation_list()

    def _restore_latest_conversation(self) -> None:
        conversations = self.store.list_conversations()
        if conversations:
            self._open_conversation(conversations[0][0])

    def _render_messages(self) -> None:
        for child in self.chat_area.winfo_children():
            child.destroy()
        if not self.messages:
            self._show_welcome()
            return
        for role, content in self.messages:
            self._add_message(role, content)
        self._scroll_to_bottom()

    def _new_chat(self) -> None:
        if self.busy:
            return
        self.conversation_id = None
        self.messages.clear()
        self.pending_files.clear()
        self.active_files.clear()
        self.pending_images.clear()
        self.active_images.clear()
        self.attachment_status.configure(text="")
        for child in self.chat_area.winfo_children():
            child.destroy()
        self._show_welcome()
        self._refresh_conversation_list()

    def _select_files(self) -> None:
        supported_types = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
        file_types = [
            ("Desteklenen belgeler", " ".join(f"*{ext}" for ext in sorted(supported_types))),
            ("Tüm dosyalar", "*.*"),
        ]
        selected = filedialog.askopenfilenames(
            title="Sohbete eklenecek dosyaları seç",
            filetypes=file_types,
        )
        if not selected:
            return
        self._add_file_paths(selected)

    def _add_file_paths(self, selected: tuple[str, ...] | list[str]) -> None:
        try:
            for selected_path in selected:
                path = Path(selected_path)
                if any(
                    item["name"].casefold() == path.name.casefold()
                    for item in self.pending_files + self.pending_images
                ):
                    continue
                if path.stat().st_size > MAX_FILE_SIZE:
                    raise ValueError(f"{path.name}: dosya 10 MB sınırını aşıyor.")
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self._add_image_file(path)
                    continue
                if len(self.pending_files) >= 5:
                    raise ValueError("Bir mesajda en fazla 5 belge ekleyebilirsin.")
                content = read_document(path)
                total_chars = sum(len(item["content"]) for item in self.pending_files)
                if total_chars + len(content) > MAX_CHAT_FILE_CHARS:
                    raise ValueError(
                        "Eklenen dosyaların toplam metni 6.000 karakteri aşıyor. "
                        "Daha az veya daha küçük dosya seç."
                    )
                self.pending_files.append({"name": path.name, "content": content})
        except (OSError, ValueError, ImportError) as error:
            messagebox.showerror("Dosya eklenemedi", str(error), parent=self)
        self._update_attachment_status()

    def _add_image_file(self, path: Path) -> None:
        from PIL import Image

        with Image.open(path) as image:
            self._add_image(image.copy(), f"{path.stem}.png")

    def _add_image(self, image: Any, name: str) -> None:
        if len(self.pending_images) >= 5:
            raise ValueError("Bir mesajda en fazla 5 görsel ekleyebilirsin.")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"{name}: görsel 5 MB sınırını aşıyor.")
        if sum(
            len(base64.b64decode(item["content_base64"]))
            for item in self.pending_images
        ) + len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("Eklenen görsellerin toplamı 10 MB sınırını aşıyor.")
        self.pending_images.append(
            {
                "name": name,
                "content_base64": base64.b64encode(image_bytes).decode("ascii"),
            }
        )

    def _paste_clipboard(self, _event: tk.Event[Any]) -> str | None:
        try:
            from PIL import ImageGrab

            clipboard = ImageGrab.grabclipboard()
        except (ImportError, OSError, ValueError):
            return None

        if isinstance(clipboard, list):
            selected = [
                str(path)
                for path in clipboard
                if Path(str(path)).suffix.lower()
                in SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
            ]
            if not selected:
                return None
            self._add_file_paths(selected)
            return "break"
        if clipboard is not None and hasattr(clipboard, "save"):
            try:
                self._add_image(
                    clipboard,
                    f"panodan-gorsel-{len(self.pending_images) + 1}.png",
                )
            except (OSError, ValueError) as error:
                messagebox.showerror("Görsel eklenemedi", str(error), parent=self)
            self._update_attachment_status()
            return "break"
        return None

    def _update_attachment_status(self) -> None:
        files = self.pending_files or self.active_files
        images = self.pending_images or self.active_images
        names = [f"📎 {item['name']}" for item in files]
        names.extend(f"▧ {item['name']}" for item in images)
        self.attachment_status.configure(
            text="   ".join(names) if names else ""
        )

    def _send_from_keyboard(self, _event: tk.Event[Any]) -> str:
        self._send_message()
        return "break"

    def _send_message(
        self,
        suggested_text: str | None = None,
    ) -> None:
        if self.busy:
            return
        prompt = suggested_text
        if prompt is None:
            prompt = self.prompt_box.get("1.0", "end-1c").strip()
            if prompt == PROMPT_PLACEHOLDER:
                prompt = ""
        if not prompt and (self.pending_files or self.pending_images):
            prompt = "Eklediğim dosyaları sırayla incele, analiz et ve bana en uygun sonucu ver."
        if not prompt:
            return
        if suggested_text is None:
            self.prompt_box.delete("1.0", "end")
        if not self.messages:
            for child in self.chat_area.winfo_children():
                child.destroy()

        files = self.pending_files.copy()
        images = self.pending_images.copy()
        if self.conversation_id is None:
            title = prompt.replace("\n", " ").strip()
            if len(title) > 36:
                title = title[:33].rstrip() + "…"
            self.conversation_id = self.store.create_conversation(title or "Dosya analizi")
        elif not self.messages:
            self.store.set_title(self.conversation_id, prompt[:36])
        if files:
            self.active_files = files
            self.store.set_files(self.conversation_id, files)
        if images:
            self.active_images = images
            self.store.set_images(self.conversation_id, images)
        stored_prompt = prompt + "".join(
            f"{FILE_MARKER}{item['name']}]" for item in self.active_files
        )
        stored_prompt += "".join(
            f"\n\n[Ek görsel: {item['name']}]" for item in self.active_images
        )
        self.store.add_message(self.conversation_id, "user", stored_prompt)
        self.messages.append(("user", stored_prompt))
        self._add_message("user", stored_prompt)
        self.pending_files.clear()
        self.pending_images.clear()
        self._update_attachment_status()
        self._refresh_conversation_list()
        request_history = self._recent_history()
        endpoint = self.api_url.get()
        self.busy = True
        self.send_button.configure(state="disabled", text="…")
        self.attach_button.configure(state="disabled")
        self.new_chat_button.configure(state="disabled")
        progress_id = uuid.uuid4().hex
        self._active_progress_id = progress_id
        requested_model = {
            "Otomatik": "auto",
            "Flash": "flash",
            "Reasoning": "reasoning",
            "Coder": "coder",
        }[self.model_choice.get()]
        initial_stage = (
            ("analyzing", "Dosyalar analiz ediliyor")
            if self.active_files or self.active_images
            else ("thinking", "Uygun model ve web araştırması belirleniyor")
        )
        self._set_activity(*initial_stage)
        self.after(
            250,
            lambda active_id=progress_id: self._poll_chat_progress(active_id),
        )
        if self.active_files or self.active_images:
            self.status_text.set("Dosyalar sırayla inceleniyor…")
        else:
            self.status_text.set("Model ve gerekli web araştırması belirleniyor…")
        threading.Thread(
            target=self._request_answer,
            args=(
                prompt,
                request_history,
                endpoint,
                self.active_files.copy(),
                requested_model,
                self.active_images.copy(),
                False,
                progress_id,
            ),
            daemon=True,
        ).start()

    def _request_answer(
        self,
        prompt: str,
        history: list[tuple[str, str]],
        endpoint: str,
        files: list[dict[str, str]],
        model: str,
        images: list[dict[str, str]],
        web_search: bool,
        progress_id: str,
    ) -> None:
        streamed_parts: list[str] = []
        try:
            client = InkAIClient(endpoint)

            def receive_chunk(chunk: str) -> None:
                streamed_parts.append(chunk)
                self.after(
                    0,
                    lambda value=chunk: self._append_stream_chunk(value),
                )

            answer, used_model = client.send_message(
                prompt,
                history,
                files,
                model,
                images,
                web_search,
                progress_id,
                on_chunk=receive_chunk,
            )
        except (
            ValueError,
            RuntimeError,
            ConnectionError,
            TimeoutError,
            OSError,
            URLError,
            KeyError,
            TypeError,
        ) as error:
            error_message = str(error)
            partial_answer = "".join(streamed_parts)
            self.after(
                0,
                lambda: self._finish_request(
                    error=error_message,
                    partial_answer=partial_answer,
                ),
            )
        else:
            if client.last_generated_image is not None:
                image_bytes, filename = client.last_generated_image
                self.after(
                    0,
                    lambda: self._finish_image(image_bytes, filename),
                )
                return
            self.after(
                0,
                lambda: self._finish_request(answer=answer, model=used_model),
            )

    def _poll_chat_progress(self, progress_id: str) -> None:
        if not self.busy or self._active_progress_id != progress_id:
            return
        threading.Thread(
            target=self._read_chat_progress,
            args=(progress_id,),
            daemon=True,
        ).start()

    def _read_chat_progress(self, progress_id: str) -> None:
        try:
            progress = InkAIClient(self.api_url.get()).get_chat_progress(progress_id)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as error:
            self.after(
                0,
                lambda message=str(error): self.status_text.set(
                    f"Durum güncellenemedi: {message}"
                ),
            )
        else:
            self.after(
                0,
                lambda: self._set_activity(
                    str(progress["stage"]),
                    str(progress["message"]),
                ),
            )
        self.after(450, lambda: self._poll_chat_progress(progress_id))

    def _set_activity(self, stage: str, message: str) -> None:
        icons = {
            "preparing": "◌",
            "queued": "◌",
            "collecting": "⌕",
            "analyzing": "◈",
            "thinking": "◉",
            "coding": "⌘",
            "generating": "✦",
            "complete": "✓",
            "error": "!",
        }
        icon = icons.get(stage, "◉")
        self.status_text.set(message)
        self.model_badge.configure(text=f"{icon}  {message}")
        if self._activity_frame is None:
            self._activity_frame = ctk.CTkFrame(
                self.chat_area,
                fg_color=PANEL,
                corner_radius=12,
                border_width=1,
                border_color=BORDER,
            )
            self._activity_frame.grid(
                row=len(self.chat_area.winfo_children()),
                column=0,
                sticky="w",
                padx=8,
                pady=7,
            )
            self._activity_icon = ctk.CTkLabel(
                self._activity_frame,
                text=icon,
                text_color=GREEN_BRIGHT,
                font=ctk.CTkFont(size=17, weight="bold"),
            )
            self._activity_icon.grid(row=0, column=0, padx=(14, 6), pady=9)
            self._activity_message = ctk.CTkLabel(
                self._activity_frame,
                text=message,
                text_color=WHITE,
                font=ctk.CTkFont(size=11, weight="bold"),
            )
            self._activity_message.grid(row=0, column=1, padx=(0, 14), pady=9)
        else:
            assert self._activity_icon is not None
            assert self._activity_message is not None
            self._activity_icon.configure(text=icon)
            self._activity_message.configure(text=message)

    def _clear_activity(self) -> None:
        if self._activity_frame is not None:
            self._activity_frame.destroy()
        self._activity_frame = None
        self._activity_icon = None
        self._activity_message = None
        self._active_progress_id = None
        self.model_badge.configure(text="✦  OTOMATİK UZMAN MODEL")

    def _request_image(self, prompt: str, endpoint: str) -> None:
        try:
            image_bytes, filename = InkAIClient(endpoint).generate_image(prompt)
        except (
            ValueError,
            RuntimeError,
            ConnectionError,
            TimeoutError,
            OSError,
            URLError,
        ) as error:
            self.after(0, lambda message=str(error): self._finish_request(error=message))
        else:
            self.after(0, lambda: self._finish_image(image_bytes, filename))

    def _finish_image(self, image_bytes: bytes, filename: str) -> None:
        from PIL import Image

        self.busy = False
        self._enable_composer()
        self._clear_activity()
        image = ctk.CTkImage(
            light_image=Image.open(BytesIO(image_bytes)),
            dark_image=Image.open(BytesIO(image_bytes)),
            size=(384, 384),
        )
        answer = f"Floppy görseli hazır: {filename}"
        if self.conversation_id is not None:
            self.store.add_message(self.conversation_id, "assistant", answer)
        self.messages.append(("assistant", answer))
        self._add_message("assistant", answer, image=image)
        self._refresh_conversation_list()
        self.status_text.set("Floppy görseli başarıyla oluşturdu")

    def _recent_history(self) -> list[tuple[str, str]]:
        history: list[tuple[str, str]] = []
        total_chars = 0
        for message in reversed(self.messages[:-1]):
            history_content = self._message_preview(message[1])
            if total_chars + len(history_content) > 1_800:
                break
            history.append((message[0], history_content))
            total_chars += len(history_content)
        return list(reversed(history))

    def _finish_request(
        self,
        answer: str | None = None,
        error: str | None = None,
        model: str | None = None,
        partial_answer: str | None = None,
    ) -> None:
        self.busy = False
        self._clear_activity()
        if error is not None:
            partial = ""
            if self._stream_body is not None:
                visible = self._stream_full_text
                if partial_answer and partial_answer.startswith(visible):
                    self._append_stream_chunk(partial_answer[len(visible) :])
                    visible = partial_answer
                partial = visible.strip()
                self._finish_stream_message()
            else:
                self._enable_composer()
                partial = (partial_answer or "").strip()
                if partial:
                    self._add_message("assistant", partial)
            if partial:
                if self.conversation_id is not None:
                    self.store.add_message(
                        self.conversation_id,
                        "assistant",
                        partial,
                    )
                self.messages.append(("assistant", partial))
                self._refresh_conversation_list()
            self._add_message("assistant", error, is_error=True)
            self.status_text.set("İstek başarısız")
            return
        if answer is None:
            self._enable_composer()
            self.status_text.set("Yanıt alınamadı")
            return
        if self.conversation_id is not None:
            self.store.add_message(self.conversation_id, "assistant", answer)
        self.messages.append(("assistant", answer))
        if self._stream_body is not None:
            self._finish_stream_message(answer)
        else:
            self.busy = True
            self.send_button.configure(state="disabled", text="…")
            self.attach_button.configure(state="disabled")
            self.new_chat_button.configure(state="disabled")
            self._add_message("assistant", answer, animate=True)
        self._refresh_conversation_list()
        model_labels = {
            "flash": "Hızlı",
            "reasoning": "Analiz",
            "coder": "Kod",
        }
        label = model_labels.get(model or "", "Yanıtlandı")
        self.status_text.set(f"Yanıtlandı · {label}")

    def _enable_composer(self) -> None:
        self.send_button.configure(state="normal", text="↑")
        self.attach_button.configure(state="normal")
        self.new_chat_button.configure(state="normal")

    def _check_health(self) -> None:
        self.status_text.set("API kontrol ediliyor…")
        endpoint = self.api_url.get()

        def check() -> None:
            try:
                health = InkAIClient(endpoint).check_health()
            except (
                ValueError,
                URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                KeyError,
            ) as error:
                message = f"API çevrimdışı: {error}"
            else:
                model_count = sum(
                    bool(model["available"]) for model in health.get("models", [])
                )
                if health.get("gpu_offload_supported") is False:
                    message = (
                        f"GPU hızlandırma yok · CPU kullanılıyor · "
                        f"{model_count}/4 model"
                    )
                else:
                    message = f"API çevrimiçi · {model_count}/4 model"
            self.after(0, lambda: self.status_text.set(message))

        threading.Thread(target=check, daemon=True).start()


def main() -> None:
    InkAIApp().mainloop()


if __name__ == "__main__":
    main()
