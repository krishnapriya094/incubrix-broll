# IncuBrix B-roll Video Engine

Turns a voiceover (or a script) into a finished, licence-clean B-roll video,
scene by scene: relevant footage for every line, timed to the narration, with
captions and a full record of where every clip came from.

This is the IncuBrix Engineering Assessment 3 submission skeleton. It runs end
to end in **mock mode with no API keys**, so the whole thing is testable
immediately; real providers (OpenAI speech, Pexels footage) plug in behind the
same adapters by changing configuration only.

## Architecture

```
 client ──POST /api/v1/broll/jobs──►  FastAPI (app/main.py)
                                        │  Idempotency-Key → JobStore (SQLite)
                                        │  202 { job_id }
                                        ▼
                                   Worker threads (app/worker.py)
                                        │
                                        ▼
                                   Pipeline (app/pipeline.py)
   1. narration + timings   ── STT (audio)  |  TTS (text/scenes)
   2. scenes on sentences   ── split, timed to the voice
   3. clip per scene        ── embeddings → FootageSource(s), no repeats, failover
   4. render                ── ffmpeg normalise → concat → mux narration → captions
   5. timeline.json + manifest.json
                                        │
                                        ▼
                             Storage adapter (local | S3/MinIO)
                                        │
 client ──GET /api/v1/jobs/{id}──►  status, progress, result links
```

Every external provider sits behind a small adapter (`app/adapters.py`) selected
by configuration, with a mock implementation so tests need no keys.

## Quick start (under 10 commands)

```bash
python -m venv .venv && source .venv/bin/activate   # 1
pip install -r requirements-dev.txt                 # 2
cp .env.example .env                                # 3
./run_tests.sh                                      # 4  all tests, no API keys
uvicorn app.main:app --port 8000                    # 5  start the service
# in another shell:                                 # 6  submit a job
curl -s -X POST localhost:8000/api/v1/broll/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: demo-1' \
  -d @testdata/request_script.json
curl -s localhost:8000/api/v1/jobs/<job_id>         # 7  poll for result links
```

Or with Docker (service + MinIO):

```bash
docker compose up --build        # api on :8000, MinIO console on :9001
docker compose run --rm api python -m pytest -q   # tests inside the image
```

## Sample output

`samples/` holds one real run in mock mode: `video_16x9.mp4`, `captions.srt`,
`timeline.json` and `manifest.json`. Open the video to see the rendered result,
and read `timeline.json` to see the per-scene clip provenance.

## Batch runner

Runs a folder or file of inputs against the API and writes every result to one
folder (the tool reviewers use to test submissions uniformly):

```bash
python -m app.batch_runner --input ./testdata --out ./batch_results
python -m app.batch_runner --input ./testdata/scripts.txt --out ./batch_results
```

Inputs may be a folder of `*.json` request bodies, a single `*.json` (object or
list), or a `*.txt` with one script per line.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/broll/jobs` | Create a job → `202 { job_id }` |
| GET | `/api/v1/jobs/{job_id}` | status, progress 0–100, result links |
| GET | `/health`, `/ready` | liveness / readiness |

Full contract: [`openapi.yaml`](./openapi.yaml). Errors are always
`{"error": {"code", "message", "retryable", "details"}}` with `UPPER_SNAKE_CASE`
codes listed in `app/core.py`.

## Configuration

All settings come from the environment — see [`.env.example`](./.env.example).
Nothing is hard-coded. The important ones:

| Variable | Meaning |
| --- | --- |
| `MOCK_MODE` | `true` = mock providers, no keys needed |
| `STORAGE_BACKEND` | `local` or `s3` (MinIO/S3-compatible) |
| `STT_BACKEND` / `TTS_BACKEND` / `EMBEDDINGS_BACKEND` | provider per step |
| `FOOTAGE_SOURCES` | ordered list, e.g. `pexels,pixabay`; first working wins |
| `FFMPEG_BINARY` | ffmpeg path; falls back to the bundled imageio-ffmpeg |

Secrets (`OPENAI_API_KEY`, `PEXELS_API_KEY`) live only in the environment and
are never written to logs or the repo.

## How the must-haves are addressed

| Must-have | Where |
| --- | --- |
| 1. Both inputs work | `pipeline.build_scenes` (audio via STT, text via TTS) |
| 2. Scenes by meaning, timed to voice | `pipeline._timed_from_words` / `_timed_from_text` |
| 3. Every clip licence-clean | `timeline.json` clip block; adapters carry licence + attribution |
| 4. Correct, consistent video | `pipeline._normalise` (scale/crop/fps/pix_fmt) + concat + mux |
| 5. No repeats, meaningful matches | `select_clips` (dedupe by id; embeddings ranking) |
| 6. Safe output | stock/synthetic clips only; no real faces; `avoid` filter |
| 7. Resilient jobs | per-scene source failover + synthetic fallback in `select_clips` |
| 8. Integration Standard | job API, idempotency, JSON errors, config, storage, logs, manifest, batch runner |

## Project layout

```
app/
  core.py        settings, JSON logging, error model
  models.py      request/response models
  store.py       SQLite job store
  storage.py     local / S3 storage adapters
  adapters.py    STT, TTS, embeddings, footage sources (mock + real)
  pipeline.py    the 5-step pipeline
  worker.py      background job queue
  main.py        FastAPI app
  batch_runner.py CLI batch tool
tests/           unit, API and end-to-end tests (mock mode)
testdata/        sample inputs
openapi.yaml     API contract
```
