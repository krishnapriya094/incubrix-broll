"""The B-roll pipeline: transcript/script -> scenes -> clips -> finished video.

Steps, in order:
  1. obtain narration audio + word timings (STT for audio, TTS for text/scenes)
  2. split into scenes on sentence boundaries, timed to the narration
  3. pick a licence-clean clip per scene (semantic match, no repeats, failover)
  4. normalise + render one video per aspect ratio, with captions
  5. write timeline.json and manifest.json
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path

import httpx

from .adapters import (Clip, EmbeddingsAdapter, MockFootageSource, STTAdapter, TTSAdapter,
                       build_footage_sources, cosine, probe_duration)
from .core import ApiError, get_logger
from .storage import get_storage

log = get_logger("pipeline")

RATIO_SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Words we drop before sending a scene to a keyword-based footage search.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "at",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "as", "your", "you",
    "we", "our", "they", "their", "i", "me", "my", "so", "then", "than", "too", "very",
    "can", "will", "just", "also", "how", "what", "when", "where", "who", "why", "not",
    "no", "do", "does", "did", "have", "has", "had", "here", "there",
}


def _keywords(text: str, limit: int = 6) -> str:
    """Reduce a scene to a few salient words for a keyword-based footage search."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    keep = [w for w in words if w not in STOPWORDS and len(w) > 2]
    out = " ".join(keep[:limit])
    return out or text.strip()[:100]


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _timed_from_words(words, max_scene_s: float = 4.0, max_scene_words: int = 14,
                     pause_s: float = 0.7) -> list[dict]:
    """Group word timings into scenes, breaking on a sentence end, a long pause,
    or when a scene grows too long - so transcripts without punctuation still split."""
    scenes: list = []
    current: list = []
    prev_end = None
    for word in words:
        if current:
            too_long = ((word.end - current[0].start) > max_scene_s
                        or len(current) >= max_scene_words)
            paused = prev_end is not None and (word.start - prev_end) > pause_s
            if too_long or paused:
                scenes.append(current)
                current = []
        current.append(word)
        if word.word.strip().endswith((".", "!", "?")):
            scenes.append(current)
            current = []
        prev_end = word.end
    if current:
        scenes.append(current)
    out = []
    for i, group in enumerate(scenes):
        text = " ".join(w.word for w in group).strip()
        out.append({"index": i, "text": text, "start": round(group[0].start, 3),
                    "end": round(group[-1].end, 3), "visual_hint": None})
    return out


def _timed_from_text(parts: list[str], total_duration: float, hints: list | None = None) -> list[dict]:
    """Distribute a known duration across scenes by word count."""
    counts = [max(1, len(p.split())) for p in parts]
    total_words = sum(counts)
    scenes, cursor = [], 0.0
    for i, (part, count) in enumerate(zip(parts, counts)):
        span = total_duration * count / total_words
        scenes.append({"index": i, "text": part, "start": round(cursor, 3),
                       "end": round(cursor + span, 3),
                       "visual_hint": (hints[i] if hints else None)})
        cursor += span
    if scenes:
        scenes[-1]["end"] = round(total_duration, 3)
    return scenes


# --------------------------------------------------------------------------- #
# step 1 + 2: narration and scenes
# --------------------------------------------------------------------------- #
def _download_audio(url: str, dest: Path) -> Path:
    if url.startswith("http://") or url.startswith("https://"):
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
    else:
        shutil.copyfile(url, dest)
    return dest


def build_scenes(request: dict, work: Path, settings) -> tuple[list[dict], Path]:
    stt = STTAdapter(settings)
    tts = TTSAdapter(settings)
    avoid_ext = {"mp3", "wav", "m4a"}

    if request.get("audio_url"):
        url = request["audio_url"]
        ext = Path(url.split("?")[0]).suffix.lstrip(".").lower()
        if ext and ext not in avoid_ext:
            raise ApiError("UNSUPPORTED_MEDIA_TYPE", details={"extension": ext})
        narration = _download_audio(url, work / f"voiceover.{ext or 'mp3'}")
        duration = probe_duration(settings.ffmpeg_binary, narration)
        if duration and duration > settings.max_audio_seconds:
            raise ApiError("AUDIO_TOO_LONG", details={"seconds": duration,
                                                     "max": settings.max_audio_seconds})
        transcript = stt.transcribe(narration)
        return _timed_from_words(transcript.words), narration

    if request.get("scenes"):
        parts = [s["text"] for s in request["scenes"]]
        hints = [s.get("visual_hint") for s in request["scenes"]]
        narration = work / "narration.wav"
        duration = tts.synthesize(" ".join(parts), request.get("voice", "en-IN-female"), narration)
        return _timed_from_text(parts, duration, hints), narration

    script = request.get("script") or ""
    narration = work / "narration.wav"
    duration = tts.synthesize(script, request.get("voice", "en-IN-female"), narration)
    return _timed_from_text(split_sentences(script), duration), narration


# --------------------------------------------------------------------------- #
# step 3: clip selection with failover and no repeats
# --------------------------------------------------------------------------- #
def select_clips(scenes: list[dict], request: dict, work: Path, settings) -> tuple[list[Clip], list[str]]:
    embed = EmbeddingsAdapter(settings)
    sources = build_footage_sources(settings)
    avoid = request.get("avoid") or []
    style = request.get("style") or ""
    used: set[str] = set()
    chosen: list[Clip] = []
    warnings: list[str] = []

    for scene in scenes:
        scene_text = " ".join(x for x in [scene["text"], scene.get("visual_hint") or ""] if x)
        # keyword-based query for the footage API; the full sentence is used for ranking
        query = _keywords(" ".join([scene_text, style]))
        qv = embed.embed([scene_text])[0]
        picked = None
        for source in sources:
            try:
                candidates = source.search(query, k=8, avoid=avoid)
            except Exception as exc:  # a provider failing must not kill the job
                warnings.append(f"source '{source.name}' failed: {exc}")
                log.warning("footage source failed", extra={"job_id": request.get("job_id", "")})
                continue
            available = [c for c in candidates if c.id not in used]
            if not available:
                continue
            # rank the candidates by how well they match the scene's MEANING
            vectors = embed.embed([c.description or query for c in available])
            sims = [cosine(qv, v) for v in vectors]
            best = max(range(len(available)), key=lambda i: sims[i])
            picked = available[best]
            picked.score = round(sims[best], 4)
            break
        if picked is None:
            # last-resort fallback: synthesise a placeholder clip so the job still
            # finishes, instead of failing the whole submission
            clip = Clip(id="fallback_" + uuid.uuid4().hex[:8], description=query,
                        source="fallback", licence="CC0 (synthetic placeholder)",
                        attribution="IncuBrix fallback", ai_generated=True,
                        color="334155", score=0.0)
            MockFootageSource(settings).fetch(clip, work / f"clip_{scene['index']:03d}.mp4",
                                              max(1.0, scene["end"] - scene["start"]))
            used.add(clip.id)
            chosen.append(clip)
            warnings.append(f"scene {scene['index']}: no footage found, used a synthetic fallback clip")
            continue
        used.add(picked.id)
        chosen.append(picked)
    return chosen, warnings


# --------------------------------------------------------------------------- #
# step 4: render
# --------------------------------------------------------------------------- #
def _run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg and, on failure, raise with its own stderr so the cause is visible."""
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise ApiError("RENDER_FAILED",
                       f"ffmpeg exited {proc.returncode}: " + " | ".join(tail),
                       status_code=500)


def _normalise(ffmpeg: str, src: Path, dest: Path, size: tuple[int, int], duration: float) -> None:
    w, h = size
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},fps=30,format=yuv420p")
    _run_ffmpeg([ffmpeg, "-y", "-stream_loop", "-1", "-i", str(src), "-t",
                 f"{max(0.5, duration):.3f}", "-vf", vf, "-an", "-c:v", "libx264",
                 "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dest)])


def write_srt(scenes: list[dict], path: Path) -> None:
    def ts(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, scene in enumerate(scenes, start=1):
        lines.append(f"{i}\n{ts(scene['start'])} --> {ts(scene['end'])}\n{scene['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_videos(scenes: list[dict], clips: list[Clip], narration: Path, ratios: list[str],
                  request: dict, work: Path, settings, storage, job_id: str) -> tuple[dict, list[str]]:
    import subprocess

    ffmpeg = settings.ffmpeg_binary
    warnings: list[str] = []
    clip_files = []
    for scene, clip in zip(scenes, clips):
        path = work / f"clip_{scene['index']:03d}.mp4"
        if not path.exists():
            duration = max(1.0, scene["end"] - scene["start"])
            if clip.download_url:
                src = next((s for s in build_footage_sources(settings)
                            if s.name == clip.source), None)
                if src is None:
                    src = build_footage_sources(settings)[-1]
                src.fetch(clip, path, duration)
            else:
                # synthetic / fallback clip: render it locally, no download
                MockFootageSource(settings).fetch(clip, path, duration)
        clip_files.append(path)

    srt_path = work / "captions.srt"
    write_srt(scenes, srt_path)
    storage.save_file(f"{job_id}/captions.srt", srt_path)

    links: dict[str, str] = {"captions_srt": storage.url(f"{job_id}/captions.srt")}

    for ratio in ratios:
        size = RATIO_SIZES[ratio]
        norm_dir = work / f"norm_{ratio.replace(':', 'x')}"
        norm_dir.mkdir(exist_ok=True)
        norm_paths = []
        for scene, clip_file in zip(scenes, clip_files):
            dest = norm_dir / f"{scene['index']:03d}.mp4"
            _normalise(ffmpeg, clip_file, dest, size, scene["end"] - scene["start"])
            norm_paths.append(dest)
        # Join the clips with the concat demuxer. Two things matter for Windows:
        #  * absolute paths in the list file (relative paths are resolved against
        #    the list file's own folder, not the project root -> "file not found")
        #  * LF line endings (newline="\n"), or the stray CR breaks the path
        # Re-encoding (instead of -c copy) keeps it robust across mixed sources and
        # streams one file at a time, so memory stays low.
        silent = norm_dir / "silent.mp4"
        concat_list = norm_dir / "list.txt"
        with open(concat_list, "w", encoding="utf-8", newline="\n") as fh:
            for path in norm_paths:
                fh.write(f"file '{path.resolve().as_posix()}'\n")
        _run_ffmpeg([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                     "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                     "-an", str(silent)])
        out = work / f"video_{ratio.replace(':', 'x')}.mp4"
        try:
            _run_ffmpeg([ffmpeg, "-y", "-i", str(silent), "-i", str(narration),
                         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)])
        except ApiError:
            # narration may be missing/short - retry with silent audio
            _run_ffmpeg([ffmpeg, "-y", "-i", str(silent), "-f", "lavfi", "-i",
                         "anullsrc=r=48000:cl=mono", "-c:v", "copy", "-c:a", "aac",
                         "-shortest", str(out)])
            warnings.append(f"{ratio}: narration audio was unusable, rendered silent")
        key = f"{job_id}/video_{ratio.replace(':', 'x')}.mp4"
        storage.save_file(key, out)
        links[f"video_{ratio.replace(':', 'x')}"] = storage.url(key)

    return links, warnings


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(job_id: str, request: dict, store, settings) -> None:
    start = time.time()
    storage = get_storage(settings)
    work = settings.data_dir / "tmp" / job_id
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        store.update(job_id, status="running", progress=10)
        scenes, narration = build_scenes(request, work, settings)

        store.update(job_id, progress=35)
        clips, clip_warnings = select_clips(scenes, {**request, "job_id": job_id}, work, settings)
        warnings += clip_warnings

        store.update(job_id, progress=60)
        ratios = request.get("aspect_ratios") or ["16:9"]
        links, render_warnings = render_videos(scenes, clips, narration, ratios, request,
                                               work, settings, storage, job_id)
        warnings += render_warnings

        store.update(job_id, progress=90)
        timeline = build_timeline(scenes, clips, request)
        manifest = build_manifest(job_id, request, settings, scenes, clips, warnings, time.time() - start)

        tl_path = work / "timeline.json"
        mf_path = work / "manifest.json"
        tl_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        links["timeline_json"] = storage.save_file(f"{job_id}/timeline.json", tl_path)
        links["manifest_json"] = storage.save_file(f"{job_id}/manifest.json", mf_path)

        store.update(job_id, status="succeeded", progress=100,
                     result={"links": links, "timeline": timeline, "manifest": manifest},
                     warnings=warnings)
    except ApiError as exc:
        store.update(job_id, status="failed", progress=100, error=exc.to_dict(), warnings=warnings)
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("pipeline failed", extra={"job_id": job_id})
        err = ApiError("INTERNAL_ERROR", str(exc), status_code=500)
        store.update(job_id, status="failed", progress=100, error=err.to_dict(), warnings=warnings)


def build_timeline(scenes: list[dict], clips: list[Clip], request: dict) -> dict:
    out_scenes = []
    for scene, clip in zip(scenes, clips):
        out_scenes.append({
            "index": scene["index"],
            "narration_text": scene["text"],
            "start_s": scene["start"],
            "end_s": scene["end"],
            "clip": {
                "id": clip.id,
                "description": clip.description,
                "source": clip.source,
                "licence": clip.licence,
                "attribution": clip.attribution,
                "ai_generated": clip.ai_generated,
                "url": clip.download_url,
            },
            "reason": f"semantic match score {clip.score}",
            "relevance_score": round(min(5.0, 1.0 + clip.score * 4.0), 2),
        })
    return {"aspect_ratios": request.get("aspect_ratios") or ["16:9"], "scenes": out_scenes}


def build_manifest(job_id: str, request: dict, settings, scenes: list[dict], clips: list[Clip],
                   warnings: list[str], elapsed: float) -> dict:
    providers = {
        "stt": "mock" if settings.mock_mode else settings.stt_backend,
        "tts": "mock" if settings.mock_mode else settings.tts_backend,
        "embeddings": "mock" if settings.mock_mode else settings.embeddings_backend,
        "footage_sources": [c.source for c in clips],
    }
    return {
        "job_id": job_id,
        "models_and_providers": providers,
        "settings": {
            "aspect_ratios": request.get("aspect_ratios") or ["16:9"],
            "voice": request.get("voice"),
            "style": request.get("style"),
            "avoid": request.get("avoid") or [],
            "captions": request.get("captions") or {},
        },
        "seed": request.get("seed"),
        "scene_count": len(scenes),
        "elapsed_s": round(elapsed, 2),
        "cost_estimate_usd": 0.0 if settings.mock_mode else None,
        "warnings": warnings,
        "rights": {
            "licences": sorted({c.licence for c in clips}),
            "ai_generated_clips": sorted({c.id for c in clips if c.ai_generated}),
            "note": "Every clip records its source, licence and attribution in timeline.json.",
        },
    }
