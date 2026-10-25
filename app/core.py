"""Core: configuration, structured logging and the error model.

Integration Standard points covered here:
  * 5. Clear errors  - one ApiError type + UPPER_SNAKE_CASE codes.
  * 6. Configuration - every setting comes from the environment (see .env.example).
  * 8. Logs and health - JSON logs carry timestamp/level/job_id/correlation id.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Load settings from a .env file if one is present (so you can keep keys and
# toggles in .env instead of typing them every time). Harmless if the file or
# the python-dotenv package is missing - real environment variables still work.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_ffmpeg(preferred: str) -> str:
    """Return a usable ffmpeg binary.

    Prefers the configured binary; falls back to the imageio-ffmpeg bundle so
    the service can run on machines without a system ffmpeg (e.g. the CI box).
    """
    if preferred and (shutil.which(preferred) or Path(preferred).exists()):
        return preferred
    try:  # pragma: no cover - convenience fallback
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return preferred or "ffmpeg"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    mock_mode: bool = field(default_factory=lambda: _as_bool(os.getenv("MOCK_MODE"), True))
    storage_backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "local"))
    public_base_url: str = field(
        default_factory=lambda: os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
    )
    # S3 / MinIO
    s3_endpoint: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT", "http://minio:9000"))
    s3_access_key: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", "minio"))
    s3_secret_key: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY", "minio123"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "incubrix-broll"))
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "us-east-1"))
    # media
    ffmpeg_binary: str = field(
        default_factory=lambda: resolve_ffmpeg(os.getenv("FFMPEG_BINARY", "ffmpeg"))
    )
    # provider selection
    stt_backend: str = field(default_factory=lambda: os.getenv("STT_BACKEND", "mock"))
    tts_backend: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "mock"))
    embeddings_backend: str = field(default_factory=lambda: os.getenv("EMBEDDINGS_BACKEND", "mock"))
    footage_sources: list = field(
        default_factory=lambda: [
            s.strip() for s in os.getenv("FOOTAGE_SOURCES", "pixabay").split(",") if s.strip()
        ]
    )
    # keys (never logged, never committed)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    pexels_api_key: str = field(default_factory=lambda: os.getenv("PEXELS_API_KEY", ""))
    pixabay_api_key: str = field(default_factory=lambda: os.getenv("PIXABAY_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "whisper-1"))
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "tts-1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    # free / local backends
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    piper_voice: str = field(default_factory=lambda: os.getenv(
        "PIPER_VOICE", "./piper_voices/en_US-lessac-medium.onnx"))
    embed_model: str = field(default_factory=lambda: os.getenv(
        "EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    # runtime
    worker_threads: int = field(default_factory=lambda: int(os.getenv("WORKER_THREADS", "2")))
    max_audio_seconds: int = field(default_factory=lambda: int(os.getenv("MAX_AUDIO_SECONDS", "600")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def ensure_dirs(self) -> None:
        for sub in ("uploads", "outputs", "jobs", "tmp", "footage"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()


# --------------------------------------------------------------------------- #
# Structured JSON logging (Integration Standard point 8)
# --------------------------------------------------------------------------- #


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("job_id", "correlation_id", "path", "method", "status_code"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel((level or settings.log_level).upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Error model (Integration Standard point 5)
# --------------------------------------------------------------------------- #

ERROR_CODES = {
    "VALIDATION_ERROR": "Request body failed validation.",
    "MISSING_INPUT": "Provide exactly one of audio_url, script or scenes.",
    "UNSUPPORTED_ASPECT_RATIO": "Aspect ratio must be one of 16:9, 9:16, 1:1.",
    "UNSUPPORTED_MEDIA_TYPE": "Only mp3, wav and m4a audio inputs are supported.",
    "AUDIO_TOO_LONG": "Audio exceeds the configured maximum length.",
    "JOB_NOT_FOUND": "No job exists with that id.",
    "INVALID_IDEMPOTENCY_KEY": "Idempotency-Key must be a non-empty string.",
    "FOOTAGE_UNAVAILABLE": "No footage source returned a usable clip.",
    "RENDER_FAILED": "Video rendering failed.",
    "INTERNAL_ERROR": "Unexpected server error.",
}


class ApiError(Exception):
    """Raised anywhere in the service; rendered as the standard JSON error body."""

    def __init__(self, code: str, message: str | None = None, *, status_code: int = 400,
                 retryable: bool = False, details: dict | None = None):
        self.code = code
        self.message = message or ERROR_CODES.get(code, code)
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }
