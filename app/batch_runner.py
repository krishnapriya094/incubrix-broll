"""Batch runner (Integration Standard point 10).

Runs a folder or file of test inputs against the API and writes every result to
one output folder. This is the tool reviewers use to test every submission the
same way.

Usage:
    python -m app.batch_runner --input ./testdata --out ./batch_results
    python -m app.batch_runner --input ./testdata/scripts.txt --out ./batch_results

Input formats:
    * a folder of *.json files, each one request body, or
    * a single *.json file (one object or a list of objects), or
    * a *.txt file with one script per non-empty line.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import httpx


def load_inputs(path: Path) -> list[dict]:
    if path.is_dir():
        inputs = []
        for file in sorted(path.glob("*.json")):
            data = json.loads(file.read_text())
            inputs.extend(data if isinstance(data, list) else [data])
        return inputs
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    if path.suffix == ".txt":
        return [{"script": line.strip(), "aspect_ratios": ["16:9"]}
                for line in path.read_text().splitlines() if line.strip()]
    raise SystemExit(f"unsupported input: {path}")


def run(base_url: str, inputs: list[dict], out_dir: Path, poll_interval: float = 2.0,
        timeout_s: float = 900.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for i, body in enumerate(inputs):
            key = f"batch-{uuid.uuid4().hex}"
            resp = client.post("/api/v1/broll/jobs", json=body,
                               headers={"Idempotency-Key": key})
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            deadline = time.time() + timeout_s
            status = {}
            while time.time() < deadline:
                status = client.get(f"/api/v1/jobs/{job_id}").json()
                if status["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(poll_interval)
            (out_dir / f"{i:03d}_{job_id}.json").write_text(json.dumps(status, indent=2))
            summary.append({"input": i, "job_id": job_id, "status": status.get("status"),
                            "progress": status.get("progress")})
            print(f"[{i}] {job_id} -> {status.get('status')}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(summary)} job(s) written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of inputs against the B-roll API.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()
    run(args.base_url, load_inputs(args.input), args.out, args.poll_interval)


if __name__ == "__main__":
    main()
