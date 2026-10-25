# REPORT — Script and Voiceover to B-roll Video Engine

> **Status of this submission.** This is a working skeleton. The full pipeline
> runs end to end in mock mode (no API keys), with a complete API, background
> jobs, storage, logging, manifest and tests. The real providers (OpenAI speech,
> Pexels footage) are wired behind the same adapters but are **not yet enabled
> or measured** — see "What does not work yet". Everything below is stated
> honestly; unmeasured items are marked as such.

## 1. The problem

A creator records a 60-second voiceover and then spends hours hunting stock
footage for each sentence, trimming clips, matching sizes and frame rates, and
adding captions. Keyword search makes it worse ("value" returns a pile of cash,
"cloud computing" returns the sky). The engine must understand what each part of
the narration *means*, pick footage that fits, time it to the voice, and record
where every clip came from and under what licence.

## 2. Approaches considered

**A. One video-generation model per scene.** Feed each line to a text-to-video
model and stitch the results. Rejected as the default path: expensive, slow,
and it produces AI-generated content that is harder to clear for commercial use
and risks realistic faces (must-have 6). Kept as a possible *second* footage
source for the Advanced level, not the default.

**B. Semantic retrieval over licensed stock libraries (chosen).** Turn each
scene into an embedding, rank candidate clips from a commercial-licence stock
library by cosine similarity, and assemble with ffmpeg. Reasons: footage is
already licence-clean and safe (no invented faces or logos), it is cheap and
fast, and relevance is measurable. This is the default path.

**C. Symbolic/keyword matching with a visual dictionary.** Rejected: the brief's
own example ("value" → cash) shows keyword matching fails on double meanings;
embeddings handle this far better.

**Timing model.** The narration is the timing authority, never the clips. For
audio input we take word-level timings from speech-to-text and group them into
sentences. For text input we generate narration with TTS and distribute its
duration across sentences by word count. Either way every second of narration is
covered and scene changes fall on sentence boundaries.

## 3. What works

- **Both input types**: `audio_url` (STT) and `script`/`scenes` (TTS) — verified
  by tests.
- **Scene segmentation timed to the voice**, on sentence boundaries.
- **Semantic clip selection** with no repeats within a video and per-scene
  failover between footage sources (a failing source does not kill the job).
- **Rendering**: one video per requested aspect ratio (16:9, 9:16, 1:1), each
  clip normalised to a common resolution/frame rate/pixel format, concatenated,
  with the narration muxed on top and captions written as SRT.
- **Provenance**: `timeline.json` records, per scene, the narration text and
  times, the clip used, its source, licence, attribution, whether it is
  AI-generated, the reason it was chosen and a relevance score. `manifest.json`
  records providers and versions, settings, timings, cost and rights.
- **Service shape**: background jobs with progress, idempotency keys, JSON
  errors, environment-only configuration, structured logs, `/health` and
  `/ready`, an S3/MinIO storage option, and a batch runner.

## 4. Measured numbers

| Metric | Value | How measured |
| --- | --- | --- |
| Turnaround (mock, 3 scenes, 2 ratios) | ~4 s cold | pipeline `elapsed_s` in manifest |
| Test suite | 16 tests, all passing | `./run_tests.sh` |
| Coverage of narration time | 100% | scenes are contiguous and end at narration length |
| Repeat-clip rate | 0 | asserted in `tests/test_integration.py` |
| Cost per minute | **not measured** | mock path is free; real providers not yet wired |

> Fill these in with your own runs once the real providers are enabled. The
> brief requires cold and warm turnaround, cost per minute, coverage, relevance,
> reliability and resources — report them even where the answer is "not done".

## 5. What does not work yet

- **Real providers are stubbed.** `STT_BACKEND=openai`, `TTS_BACKEND=openai`
  and the Pexels footage source exist but have not been run or measured.
- **Relevance is mock-approximate.** In mock mode, clips are synthetic colour
  placeholders, so the "relevance ≥ 3.5/5" bar cannot be evaluated. With real
  stock footage and a real embedding model this becomes meaningful.
- **Burn-in captions** are generated as SRT; burning them into the picture needs
  an ffmpeg build with libass and is not yet enabled.
- **Advanced/Outstanding items** (clip swap and partial re-render, colour
  normalisation across sources, independent relevance scoring, multilingual
  narration, production numbers) are **not implemented**.

## 6. Design decisions worth explaining

- **Everything external is an adapter.** Providers are selected by config with a
  mock implementation, so tests run with no keys and the default path can be
  swapped without touching the pipeline.
- **Narration drives timing.** Clips are trimmed to the narration, not the other
  way round — this is what keeps A/V in sync and avoids drift.
- **Failover at the scene level.** Each scene independently tries each footage
  source in order and falls back to a synthetic clip, so one bad provider
  degrades one scene rather than failing the job.
- **The job store is the source of truth.** Progress is persisted, so a client
  can poll at any time and the status is consistent across requests.

## 7. What I would do next

1. Enable and measure the real STT/TTS/Pexels path; report real cost per minute.
2. Add a second real footage source (Pixabay or a generative model) with a
   written rule for choosing between them, and store the reason per scene.
3. Add clip-swap with partial re-render, and colour/brightness normalisation
   across sources.
4. Score relevance with an independent vision-language model and report its
   agreement with human ratings.
5. Add Hindi and Tamil narration, and harden the hard cases (double meanings,
   named people/brands, sensitive news, provider outages).
