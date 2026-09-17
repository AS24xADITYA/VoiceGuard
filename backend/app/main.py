"""VoiceGuard FastAPI application factory.

Lifespan: initialises database, loads AI components, runs warmup.
Error handling: uniform error taxonomy from 03 §6.
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.db.session import close_db, init_db
from app.logging_config import CorrelationIdMiddleware, configure_logging, get_correlation_id

log = structlog.get_logger()


def _build_error_response(
    code: str, message: str, http_status: int, details: dict[str, Any] | None = None
) -> JSONResponse:
    """Build the uniform error envelope per 03 §6."""
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "correlation_id": get_correlation_id(),
        }
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=http_status, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()

    # Configure logging
    configure_logging(settings.log_level)
    log.info("startup_begin", env=settings.app_env, device=settings.resolve_device())

    # Validate production constraints
    settings.validate_production()

    # Initialize database
    init_db(settings)
    log.info("database_initialized", url=settings.database_url.split("@")[-1])

    # Ensure storage directory exists
    if settings.storage_backend == "local":
        Path(settings.storage_local_root).mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)

    # Initialize AI component registry
    from ai.registry import get_registry

    registry = get_registry()
    registry.set_capacity(settings.max_concurrent_inference)

    # Register AI components — they load lazily but we attempt eager load
    _register_components(registry, settings)

    # Load all components (graceful — failures are logged, not fatal)
    registry.load_all()

    # Warmup in background — don't block startup
    import asyncio

    asyncio.create_task(_warmup_components(registry))

    log.info("startup_complete")
    yield

    # Shutdown
    log.info("shutdown_begin")
    await close_db()
    log.info("shutdown_complete")


def _register_components(registry: Any, settings: Settings) -> None:
    """Register all AI components in the registry.

    Components that cannot be instantiated (missing model artifacts)
    will fail at load time, not at registration.
    """
    try:
        from ai.acoustic.detector import AcousticDetector

        registry.register(
            "acoustic",
            AcousticDetector(
                model_path=settings.acoustic_model_id,
                device=settings.resolve_device(),
                borderline_low=settings.acoustic_borderline_low,
                borderline_high=settings.acoustic_borderline_high,
            ),
        )
    except Exception as e:
        log.warning("acoustic_registration_failed", error=str(e))

    try:
        from ai.linguistic.transcriber import Transcriber

        registry.register(
            "whisper",
            Transcriber(
                model_size=settings.whisper_model_size,
                device=settings.resolve_device(),
                compute_type=settings.whisper_compute_type,
            ),
        )
    except Exception as e:
        log.warning("whisper_registration_failed", error=str(e))

    try:
        from ai.linguistic.scam_classifier import ScamIntentClassifier

        registry.register(
            "scam",
            ScamIntentClassifier(
                model_path=settings.scam_model_id,
                device=settings.resolve_device(),
            ),
        )
    except Exception as e:
        log.warning("scam_registration_failed", error=str(e))

    try:
        from ai.fusion.fuser import FusionEngine

        registry.register(
            "fusion",
            FusionEngine(model_path=settings.fusion_model_path),
        )
    except Exception as e:
        log.warning("fusion_registration_failed", error=str(e))

    try:
        from ai.explain.gradcam import GradCAMExplainer

        acoustic_detector = registry._components.get("acoustic")
        registry.register("explainer", GradCAMExplainer(acoustic_detector=acoustic_detector))
    except Exception as e:
        log.warning("explainer_registration_failed", error=str(e))


async def _warmup_components(registry: Any) -> None:
    """Background warmup task."""
    import asyncio

    await asyncio.sleep(0.5)  # Let startup finish
    log.info("warmup_begin")
    registry.warmup_all()
    log.info("warmup_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="VoiceGuard API",
        description="AI-powered multi-signal deepfake voice and scam call detection",
        version="0.1.0",
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-RateLimit-Limit",
                        "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    # Correlation ID
    app.add_middleware(CorrelationIdMiddleware)

    # ── Exception handlers ───────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        msg = "; ".join(
            f"{'.'.join(str(l) for l in e.get('loc', []))}: {e.get('msg', '')}"
            for e in errors
        )
        return _build_error_response(
            "VALIDATION_ERROR", msg, status.HTTP_400_BAD_REQUEST,
            details={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = "HTTP_ERROR"
        message = str(exc.detail)
        details = None
        if isinstance(exc.detail, dict):
            code = exc.detail.get("code", "HTTP_ERROR")
            message = exc.detail.get("message", str(exc.detail))
            details = exc.detail.get("details")
        return _build_error_response(code, message, exc.status_code, details)

    @app.exception_handler(VoiceGuardError)
    async def voiceguard_error_handler(
        request: Request, exc: VoiceGuardError
    ) -> JSONResponse:
        return _build_error_response(exc.code, exc.message, exc.http_status, exc.details)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Stack traces never reach the client
        log.error(
            "unhandled_exception",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return _build_error_response(
            "INTERNAL_ERROR",
            "An unexpected error occurred.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    from app.api.routes_analyses import router as analyses_router
    from app.api.routes_artifacts import router as artifacts_router
    from app.api.routes_auth import router as auth_router
    from app.api.routes_challenge import router as challenge_router
    from app.api.routes_system import router as system_router

    app.include_router(system_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(analyses_router, prefix="/api/v1")
    app.include_router(challenge_router, prefix="/api/v1")
    app.include_router(artifacts_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


class VoiceGuardError(Exception):
    """Base exception for all application errors per 03 §6."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details
        super().__init__(message)


class ValidationError(VoiceGuardError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("VALIDATION_ERROR", message, status.HTTP_400_BAD_REQUEST, details)


class AuthError(VoiceGuardError):
    def __init__(self, message: str = "Authentication failed."):
        super().__init__("AUTH_ERROR", message, status.HTTP_401_UNAUTHORIZED)


class PermissionError(VoiceGuardError):
    def __init__(self, message: str = "Access denied."):
        super().__init__("PERMISSION_ERROR", message, status.HTTP_403_FORBIDDEN)


class NotFoundError(VoiceGuardError):
    def __init__(self, message: str = "Resource not found."):
        super().__init__("NOT_FOUND", message, status.HTTP_404_NOT_FOUND)


class PayloadTooLargeError(VoiceGuardError):
    def __init__(self, message: str = "File too large."):
        super().__init__("PAYLOAD_TOO_LARGE", message, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


class RateLimitedError(VoiceGuardError):
    def __init__(self, message: str = "Rate limit exceeded.", retry_after: int = 60):
        super().__init__(
            "RATE_LIMITED", message, status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after},
        )


class ModelError(VoiceGuardError):
    def __init__(self, message: str = "A model component is unavailable."):
        super().__init__("MODEL_ERROR", message, status.HTTP_503_SERVICE_UNAVAILABLE)


# Module-level app instance for uvicorn
app = create_app()
