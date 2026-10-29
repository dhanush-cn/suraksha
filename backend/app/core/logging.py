from __future__ import annotations
import json, logging, sys, uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Final
from app.core.config import LogLevel, Settings

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
REDACTED: Final[str] = "***redacted***"
_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset({"password","passwd","secret","token","access_token","refresh_token","authorization","api_key","apikey","jwt_secret","auth_token","smtp_password","twilio_auth_token","database_url","cookie","set-cookie"})
_STANDARD_ATTRS: Final[frozenset[str]] = frozenset({"args","asctime","created","exc_info","exc_text","filename","funcName","levelname","levelno","lineno","message","module","msecs","msg","name","pathname","process","processName","relativeCreated","stack_info","taskName","thread","threadName"})

def new_correlation_id() -> str: return uuid.uuid4().hex
def set_correlation_id(value: str | None) -> Token[str | None]: return _correlation_id.set(value)
def reset_correlation_id(token: Token[str | None]) -> None: _correlation_id.reset(token)
def get_correlation_id() -> str | None: return _correlation_id.get()
def set_user_id(value: str | None) -> Token[str | None]: return _user_id.set(value)
def reset_user_id(token: Token[str | None]) -> None: _user_id.reset(token)
def get_user_id() -> str | None: return _user_id.get()

def redact(value: Any, _depth: int = 0) -> Any:
    if _depth > 6: return "<max-depth>"
    if isinstance(value, dict): return {key: (REDACTED if str(key).lower() in _SENSITIVE_KEYS else redact(item, _depth + 1)) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [redact(item, _depth + 1) for item in value]
    return value

class JsonFormatter(logging.Formatter):
    def __init__(self, *, service: str, environment: str, version: str) -> None:
        super().__init__(); self._service = service; self._environment = environment; self._version = version
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(), "level": record.levelname, "logger": record.name, "message": record.getMessage(), "service": self._service, "environment": self._environment, "version": self._version}
        if (cid := get_correlation_id()) is not None: payload["correlation_id"] = cid
        if (uid := get_user_id()) is not None: payload["user_id"] = uid
        extras = {key: value for key, value in record.__dict__.items() if key not in _STANDARD_ATTRS and not key.startswith("_")}
        if extras: payload.update(redact(extras))
        if record.exc_info: payload["exception"] = {"type": record.exc_info[0].__name__ if record.exc_info[0] else None, "stacktrace": self.formatException(record.exc_info)}
        return json.dumps(payload, default=str, ensure_ascii=False)

class ConsoleFormatter(logging.Formatter):
    _FMT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    def format(self, record: logging.LogRecord) -> str:
        base = logging.Formatter(self._FMT, datefmt="%H:%M:%S").format(record)
        if (cid := get_correlation_id()) is not None: base = f"{base}  (cid={cid[:8]})"
        return base

def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json: handler.setFormatter(JsonFormatter(service=settings.project_name, environment=str(settings.environment), version=settings.version))
    else: handler.setFormatter(ConsoleFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers): root.removeHandler(existing)
    root.addHandler(handler); root.setLevel(settings.log_level.value)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name); logger.handlers.clear(); logger.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    if settings.log_level is not LogLevel.DEBUG: logging.getLogger("httpx").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger: return logging.getLogger(name)

__all__ = ["ConsoleFormatter","JsonFormatter","configure_logging","get_correlation_id","get_logger","get_user_id","new_correlation_id","redact","reset_correlation_id","reset_user_id","set_correlation_id","set_user_id"]