"""HTTP API (FastAPI). Mirrors openapi.yaml.

Routes:
  POST /api/v1/broll/jobs        create a job (202 + job_id)
  GET  /api/v1/jobs/{job_id}     status, progress, result links
  GET  /health, GET /ready       liveness / readiness
  GET  /files/...                download rendered outputs (local storage backend)
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core import ApiError, get_logger, settings, setup_logging
from .models import ASPECT_RATIOS, BrollRequest
from .store import JobStore
from .storage import get_storage
from .worker import Worker

setup_logging()
settings.ensure_dirs()
log = get_logger("api")

store = JobStore(settings.data_dir / "jobs" / "jobs.db")
worker = Worker(store, settings, settings.worker_threads)
_submitted: set[str] = set()

app = FastAPI(title="IncuBrix B-roll Video Engine", version=__version__)


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-Id") or "cid_" + uuid.uuid4().hex[:12]
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = cid
    log.info("request", extra={"method": request.method, "path": request.url.path,
                               "status_code": response.status_code, "correlation_id": cid})
    return response


@app.exception_handler(ApiError)
async def _api_error(_request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError):
    err = ApiError("VALIDATION_ERROR", status_code=422,
                   details={"errors": jsonable_encoder(exc.errors(), custom_encoder={ValueError: str})})
    return JSONResponse(status_code=422, content=err.to_dict())


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception):  # pragma: no cover
    log.exception("unhandled error")
    err = ApiError("INTERNAL_ERROR", str(exc), status_code=500)
    return JSONResponse(status_code=500, content=err.to_dict())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    storage = get_storage(settings)
    return {"status": "ready", "storage": settings.storage_backend,
            "mock_mode": settings.mock_mode, "ffmpeg": settings.ffmpeg_binary,
            "footage_sources": settings.footage_sources,
            "pixabay_key_set": bool(settings.pixabay_api_key)}


@app.post("/api/v1/broll/jobs", status_code=202)
async def create_job(request: Request, body: BrollRequest,
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    if idempotency_key is not None and not idempotency_key.strip():
        raise ApiError("INVALID_IDEMPOTENCY_KEY", status_code=400)
    for ratio in body.aspect_ratios:
        if ratio not in ASPECT_RATIOS:
            raise ApiError("UNSUPPORTED_ASPECT_RATIO", details={"value": ratio,
                                                               "allowed": sorted(ASPECT_RATIOS)})
    job = store.create(body.model_dump(), idempotency_key)
    if job["job_id"] not in _submitted:
        _submitted.add(job["job_id"])
        worker.submit(job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"],
            "links": {"self": f"/api/v1/jobs/{job['job_id']}"}}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise ApiError("JOB_NOT_FOUND", status_code=404)
    return {"job_id": job["job_id"], "status": job["status"], "progress": job["progress"],
            "result": job["result"], "error": job["error"], "warnings": job["warnings"] or []}


# Serve local outputs. With the s3 backend, result links are presigned instead.
app.mount("/files", StaticFiles(directory=str(settings.data_dir / "outputs")), name="files")
