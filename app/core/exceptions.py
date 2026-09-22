"""Custom exceptions + FastAPI handlers."""
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════
class AppException(Exception):
    """
    Base exception — all custom errors inherit from this.

    Attributes:
        status_code:  HTTP status code to return.
        error_code:   Machine-readable code for the frontend.
        retry_after:  Optional seconds to include in Retry-After header.
    """
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"
    retry_after: int | None = None

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# ─── 4xx — Client errors ────────────────────────────────

class BadRequestError(AppException):
    """Generic 400 for malformed requests that aren't validation errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "bad_request"


class AuthenticationError(AppException):
    """Missing / invalid credentials. Client should log in again."""
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "authentication_error"


class InsufficientCreditsError(AppException):
    """
    402 — user exhausted their AI credit budget for the period.
    Details should carry: used, limit, resets_at.
    """
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    error_code = "insufficient_credits"


class AuthorizationError(AppException):
    """Authenticated but not permitted (wrong role, banned, etc.)."""
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "authorization_error"


class NotFoundError(AppException):
    """Resource does not exist."""
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"


class NotAcceptableError(AppException):
    """Requested representation (Accept header) not supported."""
    status_code = status.HTTP_406_NOT_ACCEPTABLE
    error_code = "not_acceptable"


class ConflictError(AppException):
    """Resource already exists or state conflict (e.g. duplicate email)."""
    status_code = status.HTTP_409_CONFLICT
    error_code = "conflict"


class GoneError(AppException):
    """Resource existed but has been permanently deleted."""
    status_code = status.HTTP_410_GONE
    error_code = "gone"


class PayloadTooLargeError(AppException):
    """Request body exceeded a configured limit."""
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "payload_too_large"


class ValidationError(AppException):
    """Semantic validation failure (not Pydantic schema validation)."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "validation_error"


class RateLimitError(AppException):
    """Client has exceeded a rate limit."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: dict | None = None,
        retry_after: int = 60,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


# ─── 5xx — Server / dependency errors ───────────────────

class ExternalServiceError(AppException):
    """
    502 — a downstream dependency returned an error:
    OpenAI rate limited, Ollama offline, scraped site 5xx, etc.
    Distinct from ServiceUnavailableError in that *our* wiring is fine;
    the failure is in the third party.
    """
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "external_service_error"


class ServiceUnavailableError(AppException):
    """
    503 — a required dependency is not currently available:
    monitoring engine not wired, feature flag off, DB pool exhausted.
    Emits a Retry-After header so clients know when to try again.
    """
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "service_unavailable"

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        retry_after: int = 60,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class AIBudgetExceededError(AppException):
    """
    503 — global AI daily cost limit reached. System-wide kill switch.
    Distinct from InsufficientCreditsError (which is per-user).
    """
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "ai_budget_exceeded"

    def __init__(
        self,
        message: str = "AI daily budget exceeded",
        details: dict | None = None,
        retry_after: int = 3600,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


# ─── Special ────────────────────────────────────────────

class HoneypotTriggered(AppException):
    """
    Internal signal raised by honeypot middleware. Never shown as 4xx
    to the client — the handler always returns 404 to hide the trap.
    """
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"


# ═══════════════════════════════════════════════════════
# Handlers
# ═══════════════════════════════════════════════════════
def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_payload(
    code: str,
    message: str,
    details: dict | None = None,
    request_id: str | None = None,
) -> dict:
    payload = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }
    if request_id:
        payload["request_id"] = request_id
    return payload


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        request_id = _get_request_id(request)
        logger.warning(
            "app_exception",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
            method=request.method,
            request_id=request_id,
        )

        headers: dict[str, str] = {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                exc.error_code, exc.message, exc.details, request_id
            ),
            headers=headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ):
        request_id = _get_request_id(request)
        headers: dict[str, str] = {}
        # Propagate any headers FastAPI/Starlette set on the original exc
        if getattr(exc, "headers", None):
            headers.update(exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                "http_error", str(exc.detail), request_id=request_id
            ),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        request_id = _get_request_id(request)
        # Pydantic v2 errors may contain non-JSON-serializable objects
        safe_errors = []
        for err in exc.errors():
            safe_err = {k: v for k, v in err.items() if k != "ctx"}
            if "ctx" in err and isinstance(err["ctx"], dict):
                safe_err["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
            safe_errors.append(safe_err)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "validation_error",
                "Invalid request",
                {"errors": safe_errors},
                request_id,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = _get_request_id(request)
        logger.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            exc_info=True,
        )
        # Never leak internals in production
        detail = str(exc) if settings.debug else "Internal server error"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                "internal_error", detail, request_id=request_id
            ),
        )