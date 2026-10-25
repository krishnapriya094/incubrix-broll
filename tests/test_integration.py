"""End-to-end test: full pipeline through the API, all must-have behaviours."""
import time

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _run(body, key):
    resp = client.post("/api/v1/broll/jobs", json=body, headers={"Idempotency-Key": key})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    deadline = time.time() + 180
    while time.time() < deadline:
        data = client.get(f"/api/v1/jobs/{job_id}").json()
        if data["status"] in {"succeeded", "failed"}:
            return data
        time.sleep(0.5)
    raise AssertionError("timeout")


def test_two_aspect_ratios_and_no_repeated_clips():
    done = _run({"script": "Save money every month. Small habits add up over time. "
                           "Track spending and set a goal.",
                 "aspect_ratios": ["16:9", "9:16"]}, "e2e-1")
    assert done["status"] == "succeeded"
    links = done["result"]["links"]
    assert "video_16x9" in links and "video_9x16" in links
    scenes = done["result"]["timeline"]["scenes"]
    clip_ids = [s["clip"]["id"] for s in scenes]
    assert len(clip_ids) == len(set(clip_ids)), "a clip was reused in one video"


def test_every_scene_has_licence_provenance():
    done = _run({"script": "One line here. Another line there."}, "e2e-2")
    for scene in done["result"]["timeline"]["scenes"]:
        clip = scene["clip"]
        assert clip["licence"], "clip missing licence"
        assert clip["source"], "clip missing source"
        assert clip["attribution"], "clip missing attribution"
        assert isinstance(clip["ai_generated"], bool)


def test_manifest_records_providers_and_rights():
    done = _run({"script": "A short clip."}, "e2e-3")
    manifest = done["result"]["manifest"]
    assert "models_and_providers" in manifest
    assert "rights" in manifest and manifest["rights"]["licences"]


def test_scenes_input_is_accepted():
    done = _run({"scenes": [{"text": "Welcome to the channel.",
                             "visual_hint": "friendly host waving"}],
                 "aspect_ratios": ["1:1"]}, "e2e-4")
    assert done["status"] == "succeeded"
    assert "video_1x1" in done["result"]["links"]
