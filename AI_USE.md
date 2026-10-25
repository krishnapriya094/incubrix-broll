# AI_USE.md

The brief asks for every AI coding tool used, and what it was used for. Keep
this file accurate — the live session will ask you to explain and modify the
code, so list only tools you actually used.

## How AI was used

| Tool | Used for | What was verified by hand |
| --- | --- | --- |
| _<add each AI assistant you used, e.g. ChatGPT / Claude / Copilot / Cursor>_ | _<what it helped with, e.g. scaffolding the FastAPI app, writing tests, ffmpeg command lines>_ | _<what you personally checked/ran/understood>_ |

> Replace the row above with your real usage before submitting. Do not claim a
> tool you did not use, and do not leave out one you did.

## AI used *inside* the product (runtime)

| Component | Default | Notes |
| --- | --- | --- |
| Speech-to-text | mock (deterministic) | real path uses OpenAI Whisper when `STT_BACKEND=openai` |
| Text-to-speech | mock (tone) | real path uses OpenAI `tts-1` when `TTS_BACKEND=openai` |
| Embeddings | mock hash vectors | real path can use sentence-transformers |
| Footage | mock synthetic clips | real path can use the Pexels Video API |
| Timeline logic | **none — plain deterministic Python** | an AI model never builds the timeline directly |

## Reminder

- In the live session you must be able to explain and change this code yourself.
- Any AI-generated media the engine produces must be flagged as AI-generated in
  `manifest.json` (the pipeline already does this per clip).
