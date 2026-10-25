"""API contract tests against the running app in mock mode."""
import time

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _poll(job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.5)
    raise AssertionError("job did not finish in time")


def test_health_and_ready():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_create_job_runs_to_success():
    body = {"script": "Save money every month. Track your spending.",
            "aspect_ratios": ["16:9"]}
    resp = client.post("/api/v1/broll/jobs", json=body,
                       headers={"Idempotency-Key": "api-test-1"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    done = _poll(job_id)
    assert done["status"] == "succeeded"
    assert done["progress"] == 100
    links = done["result"]["links"]
    assert "video_16x9" in links and "timeline_json" in links and "manifest_json" in links
    assert len(done["result"]["timeline"]["scenes"]) == 2


def test_idempotency_key_returns_same_job():
    body = {"script": "One. Two."}
    first = client.post("/api/v1/broll/jobs", json=body,
                        headers={"Idempotency-Key": "same-key"}).json()
    second = client.post("/api/v1/broll/jobs", json=body,
                         headers={"Idempotency-Key": "same-key"}).json()
    assert first["job_id"] == second["job_id"]


def test_validation_error_when_two_inputs_given():
    resp = client.post("/api/v1/broll/jobs", json={"script": "x", "audio_url": "http://x/a.mp3"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unsupported_aspect_ratio():
    resp = client.post("/api/v1/broll/jobs", json={"script": "x", "aspect_ratios": ["4:3"]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_ASPECT_RATIO"


def test_job_not_found():
    resp = client.get("/api/v1/jobs/job_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_error_body_shape():
    resp = client.get("/api/v1/jobs/nope")
    err = resp.json()["error"]
    assert set(err) >= {"code", "message", "retryable", "details"}
