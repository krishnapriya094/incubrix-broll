# SOURCES.md

Every model, service, library, dataset and media source used, with its licence.
**Update this file whenever you add or change a dependency, and verify each
licence allows commercial use before relying on it.**

## Runtime libraries

| Component | Licence | Notes |
| --- | --- | --- |
| FastAPI | MIT | web framework |
| Uvicorn | BSD-3-Clause | ASGI server |
| Pydantic | MIT | request/response validation |
| HTTPX | BSD-3-Clause | outbound HTTP (providers, downloads) |
| python-multipart | Apache-2.0 | file uploads |
| boto3 | Apache-2.0 | S3/MinIO storage backend (optional) |
| imageio-ffmpeg | BSD-2-Clause (bundled ffmpeg binary: see below) | convenience fallback ffmpeg |
| pytest | MIT | test runner (dev only) |

## Media tooling

| Component | Licence | Notes |
| --- | --- | --- |
| FFmpeg | LGPL-2.1+ / GPL (depends on build) | Called as a separate process. Debian's `ffmpeg` package is built with GPL components. **Verify** that your distribution's ffmpeg build is acceptable for your commercial use; calling it as a subprocess rather than linking it is the usual mitigation, but confirm with counsel. |

## Providers (enabled only when `MOCK_MODE=false`)

| Provider | Use | Terms / licence |
| --- | --- | --- |
| OpenAI (Whisper `whisper-1`) | speech-to-text with word timings | OpenAI API terms — commercial use permitted; output usable by you |
| OpenAI (`tts-1`) | text-to-speech narration | OpenAI API terms |
| Pexels Video API | stock footage | Pexels License — free for commercial use, no attribution required (attribution recorded anyway) |
| (Pixabay) | optional second footage source | Pixabay Content License — verify before enabling |

## Embeddings

| Component | Licence | Notes |
| --- | --- | --- |
| Mock hash embeddings | in-house (this repo) | default; deterministic, no external calls |
| sentence-transformers `all-MiniLM-L6-v2` | Apache-2.0 | optional real embeddings, not enabled by default |

## Media / data used in tests

| Source | Licence | Notes |
| --- | --- | --- |
| Mock footage clips | generated in-repo (CC0-style synthetic placeholders) | produced by ffmpeg colour sources; contain no real people or brands |
| `testdata/*.txt`, `testdata/*.json` | in-house | written for this submission |
| Test narration audio | generated via TTS (mock tone, or OpenAI TTS) | no real person's voice is used |

## What is deliberately NOT used

- No real person's voice, face or likeness without explicit consent.
- No research-only / non-commercial (e.g. CC BY-NC) models, data or media in the
  default path.
- No AGPL code in the shipped service.
