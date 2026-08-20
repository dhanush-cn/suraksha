from __future__ import annotations
import json, sys
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Final, Literal
from pydantic import AnyHttpUrl, Field, RedisDsn, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MIN_SECRET_LENGTH: Final[int] = 32

class Environment(StrEnum):
    DEVELOPMENT = "development"; STAGING = "staging"; PRODUCTION = "production"; TEST = "test"
    @property
    def is_production(self) -> bool: return self is Environment.PRODUCTION
    @property
    def is_local(self) -> bool: return self in (Environment.DEVELOPMENT, Environment.TEST)

class LogLevel(StrEnum):
    DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"; CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env",".env.local"), env_file_encoding="utf-8", env_nested_delimiter="__", case_sensitive=False, extra="ignore", frozen=True)
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    project_name: str = "RockfallGuard"
    version: str = "3.0.0"
    api_prefix: str = "/api/v1"
    jwt_secret: SecretStr
    jwt_algorithm: Literal["HS256","HS384","HS512"] = "HS256"
    access_token_ttl_seconds: Annotated[int, Field(ge=60, le=86_400)] = 3_600
    refresh_token_ttl_seconds: Annotated[int, Field(ge=3_600, le=2_592_000)] = 604_800
    jwt_issuer: str = "rockfallguard.api"
    jwt_audience: str = "rockfallguard.client"
    bcrypt_rounds: Annotated[int, Field(ge=10, le=16)] = 12
    cors_origins: Annotated[list[AnyHttpUrl], NoDecode] = Field(default_factory=list)
    cors_allow_credentials: bool = False
    trusted_hosts: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    max_request_body_bytes: Annotated[int, Field(ge=1_024)] = 10 * 1024 * 1024
    max_upload_rows: Annotated[int, Field(ge=1)] = 50_000
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    database_url: SecretStr
    database_pool_size: Annotated[int, Field(ge=1, le=100)] = 10
    database_max_overflow: Annotated[int, Field(ge=0, le=100)] = 20
    database_echo: bool = False
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_socket_timeout_seconds: Annotated[float, Field(gt=0)] = 2.0
    redis_max_connections: Annotated[int, Field(ge=1)] = 50
    weather_cache_ttl_seconds: Annotated[int, Field(ge=0)] = 300
    mine_cache_ttl_seconds: Annotated[int, Field(ge=0)] = 60
    rate_limit_enabled: bool = True
    rate_limit_requests: Annotated[int, Field(ge=1)] = 60
    rate_limit_window_seconds: Annotated[int, Field(ge=1)] = 60
    weather_api_url: AnyHttpUrl = Field(default="https://api.open-meteo.com/v1/forecast")
    weather_api_timeout_seconds: Annotated[float, Field(gt=0)] = 5.0
    smtp_host: str | None = None
    smtp_port: Annotated[int, Field(ge=1, le=65_535)] = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None
    twilio_from_number: str | None = None
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = True
    models_dir: str = "models"
    require_ml_artifacts: bool = True

    # --- LLM / RAG (Step 8) -------------------------------------------------
    # Any OpenAI-chat-compatible endpoint. Point at OpenAI, Anthropic's
    # OpenAI-compat proxy, Groq, Together, or a local Ollama by changing
    # the URL alone -- no code changes needed. Absent -> chat endpoint
    # returns 503 with a config-required message rather than crashing.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_chat_model: str = "gpt-4o-mini"
    llm_embedding_model: str = "text-embedding-3-small"
    # Dimension MUST match llm_embedding_model. text-embedding-3-small = 1536;
    # -large = 3072; Ollama nomic-embed-text = 768. Kept explicit so a
    # model swap can't silently corrupt the pgvector index (which is
    # dimension-typed).
    llm_embedding_dim: Annotated[int, Field(ge=1, le=8_192)] = 1_536
    llm_request_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    rag_top_k: Annotated[int, Field(ge=1, le=50)] = 5
    # When True, /api/chat + the embedder require llm_api_key. When
    # False (dev default), a missing key falls through to a
    # "not configured" 503 without an exception. Turn on in production
    # to fail loud at boot instead.
    llm_required: bool = False

    @property
    def llm_enabled(self) -> bool:
        return self.llm_api_key is not None

    @field_validator("cors_origins", "trusted_hosts", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped: return []
            if stripped.startswith("["):
                try: return json.loads(stripped)
                except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON list: {exc}") from exc
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("api_prefix")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        if not value.startswith("/"): raise ValueError("api_prefix must start with '/'")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _enforce_environment_invariants(self) -> Settings:
        secret = self.jwt_secret.get_secret_value()
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"jwt_secret must be at least {MIN_SECRET_LENGTH} characters (got {len(secret)}). Generate one with: openssl rand -hex 32")
        if not self.environment.is_production: return self
        if self.debug: raise ValueError("debug must be False in production")
        if not self.cors_origins: raise ValueError("cors_origins must be set explicitly in production; an empty list would block the frontend and a wildcard is rejected")
        if "*" in self.trusted_hosts: raise ValueError("trusted_hosts must not contain '*' in production")
        if self.cors_allow_credentials and any(str(o) == "*" for o in self.cors_origins): raise ValueError("cors_allow_credentials cannot be combined with a wildcard origin")
        if self.database_echo: raise ValueError("database_echo must be False in production (leaks query contents)")
        return self

    @property
    def cors_origin_strings(self) -> list[str]: return [str(o).rstrip("/") for o in self.cors_origins]
    @property
    def email_enabled(self) -> bool: return bool(self.smtp_host and self.smtp_user and self.smtp_password)
    @property
    def sms_enabled(self) -> bool: return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_number)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try: return Settings()
    except ValidationError as exc:
        print("FATAL: invalid application configuration\n", file=sys.stderr)
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            print(f"  - {location}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from exc

__all__ = ["Environment", "LogLevel", "Settings", "get_settings"]