import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from ink_ai.api.providers import ProviderRegistry
from ink_ai.api.routes import router
from ink_ai.brain.image_generation import DEFAULT_IMAGE_MODEL, LocalImageGenerator
from ink_ai.brain.model_router import MODEL_PATHS, LocalModelRouter, SohbetMotoru
from ink_ai.memory.manager import KnowledgeManager

LOGGER = logging.getLogger("ink_ai.api")
DEFAULT_MODEL_PATH = MODEL_PATHS["coder"]


def create_app(
    model_path: Path | None = None,
    engine_factory: Callable[[str], SohbetMotoru] | None = None,
    knowledge_manager_factory: Callable[[], KnowledgeManager] = KnowledgeManager,
) -> FastAPI:
    model_paths = MODEL_PATHS.copy()
    if model_path is not None:
        model_paths = {
            "flash": model_path.parent / "flash.gguf",
            "reasoning": model_path.parent / "reasoning.gguf",
            "coder": model_path,
        }
    else:
        model_paths["coder"] = Path(
            os.environ.get("INK_AI_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        ).expanduser()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        available = [path for path in model_paths.values() if path.is_file()]
        preload = "coder" if model_path is not None and model_path.is_file() else None
        application.state.models = LocalModelRouter(
            model_paths=model_paths,
            engine_factory=engine_factory,
            preload=preload,
        )
        application.state.engine = application.state.models.engine
        application.state.inference_lock = threading.Lock()
        application.state.active_inference = None
        application.state.chat_progress = {}
        application.state.chat_progress_lock = threading.Lock()
        image_model_path = Path(
            os.environ.get("INK_AI_IMAGE_MODEL_PATH", str(DEFAULT_IMAGE_MODEL))
        ).expanduser()
        application.state.image_model_path = image_model_path
        application.state.image_generator = LocalImageGenerator(image_model_path)
        application.state.knowledge = knowledge_manager_factory()
        application.state.knowledge_collections = {}
        application.state.knowledge_collections_lock = threading.Lock()
        application.state.collections_dir = Path(
            os.environ.get(
                "INK_AI_COLLECTIONS_DIR",
                str(Path.home() / ".ink-ai" / "collections"),
            )
        ).expanduser()
        application.state.model_providers = ProviderRegistry.from_environment()
        application.state.api_key = os.environ.get("INK_AI_API_KEY")
        application.state.rate_limit = int(
            os.environ.get("INK_AI_RATE_LIMIT_PER_MINUTE", "60")
        )
        if not 1 <= application.state.rate_limit <= 10_000:
            raise ValueError("INK_AI_RATE_LIMIT_PER_MINUTE must be between 1 and 10000.")
        application.state.rate_limit_buckets = defaultdict(deque)
        application.state.rate_limit_lock = threading.Lock()
        application.state.gateway_metrics = {
            "requests": 0,
            "errors": 0,
            "duration_seconds": 0.0,
        }
        application.state.gateway_metrics_lock = threading.Lock()
        application.state.active_request_cancellations = {}
        application.state.active_request_cancellations_lock = threading.Lock()
        if not available:
            LOGGER.warning(
                "Yerel GGUF modelleri bulunamadı. Sohbet için models klasörüne model "
                "dosyaları koyun veya INK_AI_*_MODEL_PATH değişkenlerini ayarlayın."
            )
        yield
        application.state.models.engine = None
        application.state.models.loaded_model = None
        application.state.image_generator.unload()
        application.state.engine = None
        application.state.image_generator.pipeline = None
        application.state.knowledge = None

    application = FastAPI(
        title="Ink AI Yerel API",
        description="Ink AI'ın yerel dil modeliyle sohbet etmek için kullanılan API.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Ink-AI-Model", "X-Ink-AI-Sources", "X-Request-ID"],
    )

    @application.middleware("http")
    async def gateway_controls(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        if (
            not supplied_request_id
            or len(supplied_request_id) > 64
            or not all(char.isalnum() or char in "-_." for char in supplied_request_id)
        ):
            request_id = uuid.uuid4().hex
        else:
            request_id = supplied_request_id
        request.state.request_id = request_id
        path = request.url.path
        is_gateway_route = path.startswith("/v1/")
        started = time.monotonic()

        def record_early_response(status_code: int) -> None:
            with request.app.state.gateway_metrics_lock:
                request.app.state.gateway_metrics["requests"] += 1
                request.app.state.gateway_metrics["duration_seconds"] += (
                    time.monotonic() - started
                )
                if status_code >= 500:
                    request.app.state.gateway_metrics["errors"] += 1

        if is_gateway_route and request.app.state.api_key:
            authorization = request.headers.get("Authorization", "")
            bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
            api_key = request.headers.get("X-API-Key", "")
            if not (
                secrets_equal(bearer, request.app.state.api_key)
                or secrets_equal(api_key, request.app.state.api_key)
            ):
                result = JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Valid API key required.", "type": "authentication_error"}},
                    headers={"WWW-Authenticate": "Bearer", "X-Request-ID": request_id},
                )
                record_early_response(result.status_code)
                return result

        if is_gateway_route:
            client_id = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with request.app.state.rate_limit_lock:
                buckets = request.app.state.rate_limit_buckets
                if len(buckets) > 4_096:
                    stale_clients = [
                        key
                        for key, previous in buckets.items()
                        if not previous or now - previous[-1] >= 60
                    ]
                    for key in stale_clients:
                        del buckets[key]
                bucket = buckets[client_id]
                while bucket and now - bucket[0] >= 60:
                    bucket.popleft()
                if len(bucket) >= request.app.state.rate_limit:
                    result = JSONResponse(
                        status_code=429,
                        content={"error": {"message": "Rate limit exceeded.", "type": "rate_limit_error"}},
                        headers={
                            "Retry-After": str(max(1, int(60 - (now - bucket[0])))),
                            "X-Request-ID": request_id,
                        },
                    )
                    record_early_response(result.status_code)
                    return result
                bucket.append(now)

        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("Unhandled API error (request_id=%s).", request_id)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error.", "request_id": request_id},
            )
        duration = time.monotonic() - started
        with request.app.state.gateway_metrics_lock:
            request.app.state.gateway_metrics["requests"] += 1
            request.app.state.gateway_metrics["duration_seconds"] += duration
            if response.status_code >= 500:
                request.app.state.gateway_metrics["errors"] += 1
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def metrics_endpoint(request: Request) -> str:
        with request.app.state.gateway_metrics_lock:
            metrics = dict(request.app.state.gateway_metrics)
        return (
            "# HELP ink_ai_api_requests_total Total HTTP requests.\n"
            "# TYPE ink_ai_api_requests_total counter\n"
            f"ink_ai_api_requests_total {metrics['requests']}\n"
            "# HELP ink_ai_api_errors_total HTTP 5xx responses.\n"
            "# TYPE ink_ai_api_errors_total counter\n"
            f"ink_ai_api_errors_total {metrics['errors']}\n"
            "# HELP ink_ai_api_request_duration_seconds_total Total request duration.\n"
            "# TYPE ink_ai_api_request_duration_seconds_total counter\n"
            f"ink_ai_api_request_duration_seconds_total {metrics['duration_seconds']:.6f}\n"
        )

    @application.get("/")
    def read_root() -> dict[str, str]:
        return {"status": "ok", "message": "Ink AI Engine Calisiyor"}

    application.include_router(router)
    return application


def secrets_equal(value: str, expected: str) -> bool:
    import hmac

    return bool(value) and hmac.compare_digest(value, expected)


app = create_app()
