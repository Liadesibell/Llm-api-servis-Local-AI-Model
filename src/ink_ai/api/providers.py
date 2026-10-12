from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

LOGGER = logging.getLogger("ink_ai.api.providers")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


_URL_OPENER = urllib.request.build_opener(_NoRedirectHandler)


class ProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


@dataclass(frozen=True)
class ProviderModel:
    id: str
    provider: str
    capabilities: tuple[str, ...]


class HTTPModelProvider:
    provider_id: str

    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Provider URL must use HTTP or HTTPS.")
        if parsed.username or parsed.password:
            raise ValueError("Provider URL must not contain embedded credentials.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {"Accept": "application/json"}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with _URL_OPENER.open(
                request,
                timeout=self.timeout if timeout is None else timeout,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            LOGGER.warning(
                "Provider %s returned HTTP %s for %s.",
                self.provider_id,
                error.code,
                path,
            )
            raise ProviderError(
                self.provider_id,
                f"Upstream {self.provider_id} request failed.",
                503 if error.code == 429 else 502,
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            LOGGER.warning(
                "Provider %s request failed for %s: %s",
                self.provider_id,
                path,
                type(error).__name__,
            )
            raise ProviderError(
                self.provider_id,
                f"Upstream {self.provider_id} is unavailable.",
                503,
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderError(
                self.provider_id,
                f"Upstream {self.provider_id} returned invalid JSON.",
            ) from error

    def stream(
        self,
        path: str,
        payload: dict[str, object],
        headers: dict[str, str],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[dict[str, object]]:
        body = json.dumps(payload).encode("utf-8")
        request_headers = {
            "Accept": "text/event-stream, application/x-ndjson",
            "Content-Type": "application/json",
            **headers,
        }
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            with _URL_OPENER.open(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError as error:
                        raise ProviderError(
                            self.provider_id,
                            f"Upstream {self.provider_id} returned invalid stream data.",
                        ) from error
                    if not line:
                        continue
                    if line.startswith((":", "event:", "id:", "retry:")):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            return
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ProviderError(
                            self.provider_id,
                            f"Upstream {self.provider_id} returned invalid stream data.",
                        ) from error
                    if not isinstance(item, dict):
                        raise ProviderError(
                            self.provider_id,
                            f"Upstream {self.provider_id} returned an invalid stream event.",
                        )
                    yield item
        except urllib.error.HTTPError as error:
            LOGGER.warning(
                "Provider %s returned HTTP %s for streaming request.",
                self.provider_id,
                error.code,
            )
            raise ProviderError(
                self.provider_id,
                f"Upstream {self.provider_id} request failed.",
                503 if error.code == 429 else 502,
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            LOGGER.warning(
                "Provider %s streaming request failed: %s",
                self.provider_id,
                type(error).__name__,
            )
            raise ProviderError(
                self.provider_id,
                f"Upstream {self.provider_id} is unavailable.",
                503,
            ) from error


class OpenAICompatibleProvider(HTTPModelProvider):
    provider_id = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
    ) -> None:
        super().__init__(base_url, timeout)
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def list_models(self) -> list[ProviderModel]:
        response = self._request(
            "/models",
            headers=self.headers,
            timeout=min(self.timeout, 2),
        )
        models = response.get("data") if isinstance(response, dict) else None
        if not isinstance(models, list):
            raise ProviderError("openai", "Provider returned an invalid model list.")
        return [
            ProviderModel(
                id=f"openai/{item['id']}",
                provider="openai",
                capabilities=("chat", "streaming"),
            )
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    def chat(self, model: str, payload: dict[str, object]) -> Any:
        request_payload = {**payload, "model": model}
        return self._request(
            "/chat/completions",
            request_payload,
            method="POST",
            headers=self.headers,
        )

    def stream_chat(
        self,
        model: str,
        payload: dict[str, object],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[dict[str, object]]:
        request_payload = {**payload, "model": model, "stream": True}
        return self.stream(
            "/chat/completions",
            request_payload,
            self.headers,
            cancel_event,
        )

    def embeddings(self, model: str, inputs: list[str]) -> Any:
        return self._request(
            "/embeddings",
            {"model": model, "input": inputs},
            method="POST",
            headers=self.headers,
        )


class OllamaProvider(HTTPModelProvider):
    provider_id = "ollama"

    @staticmethod
    def _chat_payload(
        model: str,
        payload: dict[str, object],
        *,
        stream: bool,
    ) -> dict[str, object]:
        options: dict[str, object] = {}
        if isinstance(payload.get("temperature"), (float, int)):
            options["temperature"] = payload["temperature"]
        if isinstance(payload.get("max_tokens"), int):
            options["num_predict"] = payload["max_tokens"]
        request_body: dict[str, object] = {
            "model": model,
            "messages": payload.get("messages"),
            "stream": stream,
            "options": options,
        }
        if isinstance(payload.get("tools"), list):
            request_body["tools"] = payload["tools"]
        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                request_body["format"] = "json"
            elif response_format.get("type") == "json_schema":
                schema = response_format.get("json_schema")
                if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
                    request_body["format"] = schema["schema"]
        return request_body

    def list_models(self) -> list[ProviderModel]:
        response = self._request("/api/tags", timeout=min(self.timeout, 2))
        models = response.get("models") if isinstance(response, dict) else None
        if not isinstance(models, list):
            raise ProviderError("ollama", "Ollama returned an invalid model list.")
        return [
            ProviderModel(
                id=f"ollama/{item['name']}",
                provider="ollama",
                capabilities=("chat", "streaming"),
            )
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]

    def chat(self, model: str, payload: dict[str, object]) -> Any:
        response = self._request(
            "/api/chat",
            self._chat_payload(model, payload, stream=False),
            method="POST",
        )
        message = response.get("message") if isinstance(response, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("ollama", "Ollama returned an invalid chat response.")
        tool_calls = message.get("tool_calls")
        normalized_tool_calls = []
        if isinstance(tool_calls, list):
            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = function.get("arguments", {})
                if not isinstance(arguments, (dict, str)):
                    continue
                normalized_tool_calls.append(
                    {
                        "id": f"call_ollama_{index}",
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": (
                                arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, ensure_ascii=False)
                            ),
                        },
                    }
                )
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": message["content"],
        }
        if normalized_tool_calls:
            assistant_message["tool_calls"] = normalized_tool_calls
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        if not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        if not isinstance(completion_tokens, int):
            completion_tokens = 0
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "tool_calls" if normalized_tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_chat(
        self,
        model: str,
        payload: dict[str, object],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[dict[str, object]]:
        request_payload = self._chat_payload(model, payload, stream=True)
        for item in self.stream("/api/chat", request_payload, {}, cancel_event):
            if isinstance(item.get("error"), str):
                raise ProviderError(
                    "ollama",
                    "Ollama failed to generate a response.",
                    503,
                )
            message = item.get("message")
            content = message.get("content", "") if isinstance(message, dict) else ""
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            normalized_tools = []
            if isinstance(tool_calls, list):
                for index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                        continue
                    arguments = function.get("arguments", {})
                    normalized_tools.append(
                        {
                            "index": index,
                            "id": f"call_ollama_{index}",
                            "type": "function",
                            "function": {
                                "name": function["name"],
                                "arguments": (
                                    arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments, ensure_ascii=False)
                                ),
                            },
                        }
                    )
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": content,
                            **({"tool_calls": normalized_tools} if normalized_tools else {}),
                        },
                        "finish_reason": "stop" if item.get("done") else None,
                    }
                ]
            }

    def embeddings(self, model: str, inputs: list[str]) -> Any:
        response = self._request(
            "/api/embed",
            {"model": model, "input": inputs},
            method="POST",
        )
        embeddings = response.get("embeddings") if isinstance(response, dict) else None
        if not isinstance(embeddings, list):
            raise ProviderError("ollama", "Ollama returned invalid embeddings.")
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(embeddings)
            ],
            "model": model,
        }


@dataclass
class ProviderRegistry:
    openai: OpenAICompatibleProvider | None
    ollama: OllamaProvider | None

    @classmethod
    def from_environment(cls) -> ProviderRegistry:
        timeout = float(os.environ.get("INK_AI_PROVIDER_TIMEOUT_SECONDS", "60"))
        if not 1 <= timeout <= 300:
            raise ValueError("Provider timeout must be between 1 and 300 seconds.")
        openai_url = os.environ.get("INK_AI_OPENAI_BASE_URL")
        openai_key = os.environ.get("INK_AI_OPENAI_API_KEY")
        if openai_url is None and openai_key:
            openai_url = "https://api.openai.com/v1"
        ollama_url = os.environ.get("INK_AI_OLLAMA_BASE_URL")
        return cls(
            openai=(
                OpenAICompatibleProvider(openai_url, openai_key, timeout)
                if openai_url
                else None
            ),
            ollama=(
                OllamaProvider(ollama_url, timeout)
                if ollama_url
                else None
            ),
        )

    def provider_for(self, model_id: str) -> tuple[HTTPModelProvider, str]:
        if model_id.startswith("openai/") and self.openai is not None:
            return self.openai, model_id.removeprefix("openai/")
        if model_id.startswith("ollama/") and self.ollama is not None:
            return self.ollama, model_id.removeprefix("ollama/")
        raise ProviderError(
            "registry",
            f"Model '{model_id}' is unavailable or its provider is not configured.",
            404,
        )

    def list_models(self) -> tuple[list[ProviderModel], dict[str, str]]:
        models: list[ProviderModel] = []
        errors: dict[str, str] = {}
        for provider in (self.openai, self.ollama):
            if provider is None:
                continue
            try:
                models.extend(provider.list_models())
            except ProviderError as error:
                errors[provider.provider_id] = str(error)
        return models, errors
