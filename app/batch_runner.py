"""External providers, each behind a small adapter (rule 3: swappable providers).

Every provider has a mock implementation so the whole pipeline runs end to end
with no API keys (Integration Standard point 1), plus a real implementation
selected by configuration.

  STTAdapter        - speech to text with word timings
  TTSAdapter        - text to speech (narration)
  EmbeddingsAdapter - text embeddings for semantic clip matching
  FootageSource     - one footage provider (mock, pexels, pixabay)
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx


# --------------------------------------------------------------------------- #
# small ffmpeg helpers
# --------------------------------------------------------------------------- #
def probe_duration(ffmpeg: str, path: str | Path) -> float:
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not match:
        return 0.0
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def make_tone(ffmpeg: str, path: str | Path, duration_s: float, freq: int = 220) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration_s:.3f}",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, check=True,
    )


def make_color_clip(ffmpeg: str, path: str | Path, duration_s: float, color: str,
                    size: str = "1280x720", fps: int = 30) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x{color}:s={size}:d={duration_s:.3f}:r={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)],
        capture_output=True, check=True,
    )


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _hash_vector(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingsAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.embeddings_backend
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "fastembed":  # free, local, no torch
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.settings.embed_model)
            out = []
            for vec in self._model.embed(texts):
                v = [float(x) for x in vec]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
            return out
        if self.backend == "sentence_transformers":  # pragma: no cover - optional heavy dep
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.embed_model)
            return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]
        return [_hash_vector(t) for t in texts]


# --------------------------------------------------------------------------- #
# speech to text
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    segments: list = field(default_factory=list)


class STTAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.stt_backend
        self._whisper = None

    def transcribe(self, audio_path: str | Path) -> Transcript:
        if self.backend == "faster_whisper":  # free, local
            return self._faster_whisper(audio_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(audio_path)
        return self._mock(audio_path)

    def _faster_whisper(self, audio_path: str | Path) -> Transcript:
        import subprocess
        import wave

        import numpy as np
        from faster_whisper import WhisperModel

        # Decode to 16 kHz mono with ffmpeg and hand the samples straight to
        # Whisper, which avoids the fragile PyAV dependency entirely.
        wav16 = str(audio_path) + ".16k.wav"
        subprocess.run([self.settings.ffmpeg_binary, "-y", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16],
                       capture_output=True, check=True)
        with wave.open(wav16) as wf:
            audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                     .astype("float32") / 32768.0)
        if self._whisper is None:
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu",
                                         compute_type="int8")
        segments, info = self._whisper.transcribe(audio, word_timestamps=True)
        words, parts, segs = [], [], []
        for seg in segments:
            parts.append(seg.text)
            segs.append({"text": seg.text.strip(), "start": float(seg.start),
                         "end": float(seg.end)})
            for w in (seg.words or []):
                words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        return Transcript(text=" ".join(parts).strip(), words=words,
                          language=getattr(info, "language", "en"), segments=segs)

    def _mock(self, audio_path: str | Path) -> Transcript:
        duration = probe_duration(self.settings.ffmpeg_binary, audio_path) or 8.0
        text = ("In this short video we explain how to save money each month. "
                "Small habits add up over time and build a strong safety net. "
                "Track your spending and set a clear goal for the year.")
        words = text.split()
        per = duration / max(1, len(words))
        timed = [Word(w, round(i * per, 3), round((i + 1) * per, 3)) for i, w in enumerate(words)]
        return Transcript(text=text, words=timed)

    def _openai(self, audio_path: str | Path) -> Transcript:  # pragma: no cover
        with open(audio_path, "rb") as fh:
            resp = httpx.post(
                f"{self.settings.openai_base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                data={"model": self.settings.stt_model, "response_format": "verbose_json",
                      "timestamp_granularities[]": "word"},
                files={"file": fh}, timeout=180,
            )
        resp.raise_for_status()
        data = resp.json()
        words = [Word(w["word"], w["start"], w["end"]) for w in data.get("words", [])]
        return Transcript(text=data.get("text", ""), words=words, language=data.get("language", "en"))


# --------------------------------------------------------------------------- #
# text to speech
# --------------------------------------------------------------------------- #
class TTSAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.tts_backend
        self._piper = None

    def synthesize(self, text: str, voice: str, out_path: str | Path) -> float:
        if self.backend == "piper":  # free, local, MIT
            return self._piper_tts(text, out_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(text, voice, out_path)
        return self._mock(text, out_path)

    def _piper_tts(self, text: str, out_path: str | Path) -> float:
        import wave

        from piper import PiperVoice

        if self._piper is None:
            self._piper = PiperVoice.load(self.settings.piper_voice)
        with wave.open(str(out_path), "wb") as wf:
            self._piper.synthesize_wav(text, wf)
        return probe_duration(self.settings.ffmpeg_binary, out_path)

    def _mock(self, text: str, out_path: str | Path) -> float:
        # ~2.6 words per second is a reasonable speaking rate.
        duration = max(1.0, len(text.split()) / 2.6)
        make_tone(self.settings.ffmpeg_binary, out_path, duration)
        return duration

    def _openai(self, text: str, voice: str, out_path: str | Path) -> float:  # pragma: no cover
        resp = httpx.post(
            f"{self.settings.openai_base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={"model": self.settings.tts_model, "voice": "alloy", "input": text,
                  "response_format": "wav"},
            timeout=180,
        )
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)
        return probe_duration(self.settings.ffmpeg_binary, out_path)


# --------------------------------------------------------------------------- #
# footage sources
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    id: str
    description: str
    source: str
    licence: str
    attribution: str
    ai_generated: bool
    download_url: str | None = None
    color: str = "1e3a8a"
    score: float = 0.0


_MOCK_LIBRARY = [
    ("mk_city", "city street traffic cars busy road commute", "1e3a8a"),
    ("mk_money", "money savings coins finance budget growth", "065f46"),
    ("mk_cloud", "cloud computing servers data center network", "1e40af"),
    ("mk_people", "people working together office team collaboration", "7c2d12"),
    ("mk_nature", "nature landscape trees outdoors calm", "14532d"),
    ("mk_tech", "technology device screen typing laptop", "312e81"),
    ("mk_chart", "charts growth statistics business graph report", "831843"),
    ("mk_food", "food cooking kitchen healthy meal", "9a3412"),
]


class FootageSource:
    name = "base"

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:  # pragma: no cover
        raise NotImplementedError

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:  # pragma: no cover
        raise NotImplementedError


class MockFootageSource(FootageSource):
    name = "mock"

    def __init__(self, settings):
        self.settings = settings
        self.embed = EmbeddingsAdapter(settings)

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        descriptions = [d for _, d, _ in _MOCK_LIBRARY]
        qv = self.embed.embed([query])[0]
        dvs = self.embed.embed(descriptions)
        clips = []
        for (cid, desc, color), dv in zip(_MOCK_LIBRARY, dvs):
            if any(a.lower() in desc.lower() for a in avoid):
                continue
            clips.append(Clip(id=cid, description=desc, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color=color, score=round(cosine(qv, dv), 4)))
        clips.sort(key=lambda c: c.score, reverse=True)
        # If we need more than the library holds, synthesise unique clips so that
        # "no clip is used twice" always holds.
        while len(clips) < k:
            uid = "mk_" + uuid.uuid4().hex[:8]
            clips.append(Clip(id=uid, description=query, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color="334155", score=0.5))
        return clips[:k]

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        make_color_clip(self.settings.ffmpeg_binary, dest_path, duration_s, clip.color)


class PexelsFootageSource(FootageSource):
    name = "pexels"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pexels_api_key:
            raise RuntimeError("PEXELS_API_KEY is not configured")
        resp = httpx.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": self.settings.pexels_api_key},
            params={"query": query, "per_page": k, "orientation": "landscape"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for video in resp.json().get("videos", []):
            files = sorted(video.get("video_files", []), key=lambda f: f.get("width") or 0)
            best = next((f for f in files if (f.get("width") or 0) >= 1280), files[-1] if files else None)
            if not best:
                continue
            clips.append(Clip(
                id=f"pexels_{video['id']}",
                description=(video.get("url") or query),
                source="pexels",
                licence="Pexels License (free for commercial use)",
                attribution=f"Pexels / {video.get('user', {}).get('name', 'unknown')}",
                ai_generated=False,
                download_url=best.get("link"),
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


class PixabayFootageSource(FootageSource):
    """Free footage via the Pixabay API (Pixabay Content License, commercial OK).

    Pexels has paused new API keys, so Pixabay is the default footage source.
    """
    name = "pixabay"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pixabay_api_key:
            raise RuntimeError("PIXABAY_API_KEY is not configured")
        resp = httpx.get(
            "https://pixabay.com/api/videos/",
            params={"key": self.settings.pixabay_api_key, "q": query[:100],
                    "per_page": max(3, min(k, 20)), "safesearch": "true", "lang": "en"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for hit in resp.json().get("hits", []):
            tags = hit.get("tags", "") or ""
            if any(a.lower() in tags.lower() for a in avoid):
                continue
            vids = hit.get("videos", {}) or {}
            best = (vids.get("large") if (vids.get("large") or {}).get("url") else None) \
                or vids.get("medium") or vids.get("small") or vids.get("tiny")
            if not best or not best.get("url"):
                continue
            clips.append(Clip(
                id=f"pixabay_{hit['id']}",
                description=tags or query,
                source="pixabay",
                licence="Pixabay Content License (free for commercial use)",
                attribution=f"Pixabay / {hit.get('user', 'unknown')}",
                ai_generated=False,
                download_url=best["url"],
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


def build_footage_sources(settings) -> list[FootageSource]:
    sources: list[FootageSource] = []
    names = ["mock"] if settings.mock_mode else settings.footage_sources
    for name in names:
        if name == "mock":
            sources.append(MockFootageSource(settings))
        elif name == "pexels":
            sources.append(PexelsFootageSource(settings))
        elif name == "pixabay":
            sources.append(PixabayFootageSource(settings))
    if not sources:
        sources.append(MockFootageSource(settings))
    return sources
"""External providers, each behind a small adapter (rule 3: swappable providers).

Every provider has a mock implementation so the whole pipeline runs end to end
with no API keys (Integration Standard point 1), plus a real implementation
selected by configuration.

  STTAdapter        - speech to text with word timings
  TTSAdapter        - text to speech (narration)
  EmbeddingsAdapter - text embeddings for semantic clip matching
  FootageSource     - one footage provider (mock, pexels, pixabay)
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx


# --------------------------------------------------------------------------- #
# small ffmpeg helpers
# --------------------------------------------------------------------------- #
def probe_duration(ffmpeg: str, path: str | Path) -> float:
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not match:
        return 0.0
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def make_tone(ffmpeg: str, path: str | Path, duration_s: float, freq: int = 220) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration_s:.3f}",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, check=True,
    )


def make_color_clip(ffmpeg: str, path: str | Path, duration_s: float, color: str,
                    size: str = "1280x720", fps: int = 30) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x{color}:s={size}:d={duration_s:.3f}:r={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)],
        capture_output=True, check=True,
    )


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _hash_vector(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingsAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.embeddings_backend
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "fastembed":  # free, local, no torch
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.settings.embed_model)
            out = []
            for vec in self._model.embed(texts):
                v = [float(x) for x in vec]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
            return out
        if self.backend == "sentence_transformers":  # pragma: no cover - optional heavy dep
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.embed_model)
            return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]
        return [_hash_vector(t) for t in texts]


# --------------------------------------------------------------------------- #
# speech to text
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    segments: list = field(default_factory=list)


class STTAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.stt_backend
        self._whisper = None

    def transcribe(self, audio_path: str | Path) -> Transcript:
        if self.backend == "faster_whisper":  # free, local
            return self._faster_whisper(audio_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(audio_path)
        return self._mock(audio_path)

    def _faster_whisper(self, audio_path: str | Path) -> Transcript:
        import subprocess
        import wave

        import numpy as np
        from faster_whisper import WhisperModel

        # Decode to 16 kHz mono with ffmpeg and hand the samples straight to
        # Whisper, which avoids the fragile PyAV dependency entirely.
        wav16 = str(audio_path) + ".16k.wav"
        subprocess.run([self.settings.ffmpeg_binary, "-y", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16],
                       capture_output=True, check=True)
        with wave.open(wav16) as wf:
            audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                     .astype("float32") / 32768.0)
        if self._whisper is None:
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu",
                                         compute_type="int8")
        segments, info = self._whisper.transcribe(audio, word_timestamps=True)
        words, parts, segs = [], [], []
        for seg in segments:
            parts.append(seg.text)
            segs.append({"text": seg.text.strip(), "start": float(seg.start),
                         "end": float(seg.end)})
            for w in (seg.words or []):
                words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        return Transcript(text=" ".join(parts).strip(), words=words,
                          language=getattr(info, "language", "en"), segments=segs)

    def _mock(self, audio_path: str | Path) -> Transcript:
        duration = probe_duration(self.settings.ffmpeg_binary, audio_path) or 8.0
        text = ("In this short video we explain how to save money each month. "
                "Small habits add up over time and build a strong safety net. "
                "Track your spending and set a clear goal for the year.")
        words = text.split()
        per = duration / max(1, len(words))
        timed = [Word(w, round(i * per, 3), round((i + 1) * per, 3)) for i, w in enumerate(words)]
        return Transcript(text=text, words=timed)

    def _openai(self, audio_path: str | Path) -> Transcript:  # pragma: no cover
        with open(audio_path, "rb") as fh:
            resp = httpx.post(
                f"{self.settings.openai_base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                data={"model": self.settings.stt_model, "response_format": "verbose_json",
                      "timestamp_granularities[]": "word"},
                files={"file": fh}, timeout=180,
            )
        resp.raise_for_status()
        data = resp.json()
        words = [Word(w["word"], w["start"], w["end"]) for w in data.get("words", [])]
        return Transcript(text=data.get("text", ""), words=words, language=data.get("language", "en"))


# --------------------------------------------------------------------------- #
# text to speech
# --------------------------------------------------------------------------- #
class TTSAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.tts_backend
        self._piper = None

    def synthesize(self, text: str, voice: str, out_path: str | Path) -> float:
        if self.backend == "piper":  # free, local, MIT
            return self._piper_tts(text, out_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(text, voice, out_path)
        return self._mock(text, out_path)

    def _piper_tts(self, text: str, out_path: str | Path) -> float:
        import wave

        from piper import PiperVoice

        if self._piper is None:
            self._piper = PiperVoice.load(self.settings.piper_voice)
        with wave.open(str(out_path), "wb") as wf:
            self._piper.synthesize_wav(text, wf)
        return probe_duration(self.settings.ffmpeg_binary, out_path)

    def _mock(self, text: str, out_path: str | Path) -> float:
        # ~2.6 words per second is a reasonable speaking rate.
        duration = max(1.0, len(text.split()) / 2.6)
        make_tone(self.settings.ffmpeg_binary, out_path, duration)
        return duration

    def _openai(self, text: str, voice: str, out_path: str | Path) -> float:  # pragma: no cover
        resp = httpx.post(
            f"{self.settings.openai_base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={"model": self.settings.tts_model, "voice": "alloy", "input": text,
                  "response_format": "wav"},
            timeout=180,
        )
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)
        return probe_duration(self.settings.ffmpeg_binary, out_path)


# --------------------------------------------------------------------------- #
# footage sources
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    id: str
    description: str
    source: str
    licence: str
    attribution: str
    ai_generated: bool
    download_url: str | None = None
    color: str = "1e3a8a"
    score: float = 0.0


_MOCK_LIBRARY = [
    ("mk_city", "city street traffic cars busy road commute", "1e3a8a"),
    ("mk_money", "money savings coins finance budget growth", "065f46"),
    ("mk_cloud", "cloud computing servers data center network", "1e40af"),
    ("mk_people", "people working together office team collaboration", "7c2d12"),
    ("mk_nature", "nature landscape trees outdoors calm", "14532d"),
    ("mk_tech", "technology device screen typing laptop", "312e81"),
    ("mk_chart", "charts growth statistics business graph report", "831843"),
    ("mk_food", "food cooking kitchen healthy meal", "9a3412"),
]


class FootageSource:
    name = "base"

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:  # pragma: no cover
        raise NotImplementedError

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:  # pragma: no cover
        raise NotImplementedError


class MockFootageSource(FootageSource):
    name = "mock"

    def __init__(self, settings):
        self.settings = settings
        self.embed = EmbeddingsAdapter(settings)

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        descriptions = [d for _, d, _ in _MOCK_LIBRARY]
        qv = self.embed.embed([query])[0]
        dvs = self.embed.embed(descriptions)
        clips = []
        for (cid, desc, color), dv in zip(_MOCK_LIBRARY, dvs):
            if any(a.lower() in desc.lower() for a in avoid):
                continue
            clips.append(Clip(id=cid, description=desc, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color=color, score=round(cosine(qv, dv), 4)))
        clips.sort(key=lambda c: c.score, reverse=True)
        # If we need more than the library holds, synthesise unique clips so that
        # "no clip is used twice" always holds.
        while len(clips) < k:
            uid = "mk_" + uuid.uuid4().hex[:8]
            clips.append(Clip(id=uid, description=query, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color="334155", score=0.5))
        return clips[:k]

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        make_color_clip(self.settings.ffmpeg_binary, dest_path, duration_s, clip.color)


class PexelsFootageSource(FootageSource):
    name = "pexels"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pexels_api_key:
            raise RuntimeError("PEXELS_API_KEY is not configured")
        resp = httpx.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": self.settings.pexels_api_key},
            params={"query": query, "per_page": k, "orientation": "landscape"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for video in resp.json().get("videos", []):
            files = sorted(video.get("video_files", []), key=lambda f: f.get("width") or 0)
            best = next((f for f in files if (f.get("width") or 0) >= 1280), files[-1] if files else None)
            if not best:
                continue
            clips.append(Clip(
                id=f"pexels_{video['id']}",
                description=(video.get("url") or query),
                source="pexels",
                licence="Pexels License (free for commercial use)",
                attribution=f"Pexels / {video.get('user', {}).get('name', 'unknown')}",
                ai_generated=False,
                download_url=best.get("link"),
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


class PixabayFootageSource(FootageSource):
    """Free footage via the Pixabay API (Pixabay Content License, commercial OK).

    Pexels has paused new API keys, so Pixabay is the default footage source.
    """
    name = "pixabay"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pixabay_api_key:
            raise RuntimeError("PIXABAY_API_KEY is not configured")
        resp = httpx.get(
            "https://pixabay.com/api/videos/",
            params={"key": self.settings.pixabay_api_key, "q": query[:100],
                    "per_page": max(3, min(k, 20)), "safesearch": "true", "lang": "en"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for hit in resp.json().get("hits", []):
            tags = hit.get("tags", "") or ""
            if any(a.lower() in tags.lower() for a in avoid):
                continue
            vids = hit.get("videos", {}) or {}
            best = (vids.get("large") if (vids.get("large") or {}).get("url") else None) \
                or vids.get("medium") or vids.get("small") or vids.get("tiny")
            if not best or not best.get("url"):
                continue
            clips.append(Clip(
                id=f"pixabay_{hit['id']}",
                description=tags or query,
                source="pixabay",
                licence="Pixabay Content License (free for commercial use)",
                attribution=f"Pixabay / {hit.get('user', 'unknown')}",
                ai_generated=False,
                download_url=best["url"],
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


def build_footage_sources(settings) -> list[FootageSource]:
    sources: list[FootageSource] = []
    names = ["mock"] if settings.mock_mode else settings.footage_sources
    for name in names:
        if name == "mock":
            sources.append(MockFootageSource(settings))
        elif name == "pexels":
            sources.append(PexelsFootageSource(settings))
        elif name == "pixabay":
            sources.append(PixabayFootageSource(settings))
    if not sources:
        sources.append(MockFootageSource(settings))
    return sources
"""External providers, each behind a small adapter (rule 3: swappable providers).

Every provider has a mock implementation so the whole pipeline runs end to end
with no API keys (Integration Standard point 1), plus a real implementation
selected by configuration.

  STTAdapter        - speech to text with word timings
  TTSAdapter        - text to speech (narration)
  EmbeddingsAdapter - text embeddings for semantic clip matching
  FootageSource     - one footage provider (mock, pexels, pixabay)
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx


# --------------------------------------------------------------------------- #
# small ffmpeg helpers
# --------------------------------------------------------------------------- #
def probe_duration(ffmpeg: str, path: str | Path) -> float:
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not match:
        return 0.0
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def make_tone(ffmpeg: str, path: str | Path, duration_s: float, freq: int = 220) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration_s:.3f}",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, check=True,
    )


def make_color_clip(ffmpeg: str, path: str | Path, duration_s: float, color: str,
                    size: str = "1280x720", fps: int = 30) -> None:
    duration_s = max(0.5, duration_s)
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x{color}:s={size}:d={duration_s:.3f}:r={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)],
        capture_output=True, check=True,
    )


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _hash_vector(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingsAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.embeddings_backend
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "fastembed":  # free, local, no torch
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.settings.embed_model)
            out = []
            for vec in self._model.embed(texts):
                v = [float(x) for x in vec]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
            return out
        if self.backend == "sentence_transformers":  # pragma: no cover - optional heavy dep
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.embed_model)
            return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]
        return [_hash_vector(t) for t in texts]


# --------------------------------------------------------------------------- #
# speech to text
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    segments: list = field(default_factory=list)


class STTAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.stt_backend
        self._whisper = None

    def transcribe(self, audio_path: str | Path) -> Transcript:
        if self.backend == "faster_whisper":  # free, local
            return self._faster_whisper(audio_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(audio_path)
        return self._mock(audio_path)

    def _faster_whisper(self, audio_path: str | Path) -> Transcript:
        import subprocess
        import wave

        import numpy as np
        from faster_whisper import WhisperModel

        # Decode to 16 kHz mono with ffmpeg and hand the samples straight to
        # Whisper, which avoids the fragile PyAV dependency entirely.
        wav16 = str(audio_path) + ".16k.wav"
        subprocess.run([self.settings.ffmpeg_binary, "-y", "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16],
                       capture_output=True, check=True)
        with wave.open(wav16) as wf:
            audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                     .astype("float32") / 32768.0)
        if self._whisper is None:
            self._whisper = WhisperModel(self.settings.whisper_model, device="cpu",
                                         compute_type="int8")
        segments, info = self._whisper.transcribe(audio, word_timestamps=True)
        words, parts, segs = [], [], []
        for seg in segments:
            parts.append(seg.text)
            segs.append({"text": seg.text.strip(), "start": float(seg.start),
                         "end": float(seg.end)})
            for w in (seg.words or []):
                words.append(Word(w.word.strip(), float(w.start), float(w.end)))
        return Transcript(text=" ".join(parts).strip(), words=words,
                          language=getattr(info, "language", "en"), segments=segs)

    def _mock(self, audio_path: str | Path) -> Transcript:
        duration = probe_duration(self.settings.ffmpeg_binary, audio_path) or 8.0
        text = ("In this short video we explain how to save money each month. "
                "Small habits add up over time and build a strong safety net. "
                "Track your spending and set a clear goal for the year.")
        words = text.split()
        per = duration / max(1, len(words))
        timed = [Word(w, round(i * per, 3), round((i + 1) * per, 3)) for i, w in enumerate(words)]
        return Transcript(text=text, words=timed)

    def _openai(self, audio_path: str | Path) -> Transcript:  # pragma: no cover
        with open(audio_path, "rb") as fh:
            resp = httpx.post(
                f"{self.settings.openai_base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                data={"model": self.settings.stt_model, "response_format": "verbose_json",
                      "timestamp_granularities[]": "word"},
                files={"file": fh}, timeout=180,
            )
        resp.raise_for_status()
        data = resp.json()
        words = [Word(w["word"], w["start"], w["end"]) for w in data.get("words", [])]
        return Transcript(text=data.get("text", ""), words=words, language=data.get("language", "en"))


# --------------------------------------------------------------------------- #
# text to speech
# --------------------------------------------------------------------------- #
class TTSAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock" if settings.mock_mode else settings.tts_backend
        self._piper = None

    def synthesize(self, text: str, voice: str, out_path: str | Path) -> float:
        if self.backend == "piper":  # free, local, MIT
            return self._piper_tts(text, out_path)
        if self.backend == "openai":  # pragma: no cover - needs a key
            return self._openai(text, voice, out_path)
        return self._mock(text, out_path)

    def _piper_tts(self, text: str, out_path: str | Path) -> float:
        import wave

        from piper import PiperVoice

        if self._piper is None:
            self._piper = PiperVoice.load(self.settings.piper_voice)
        with wave.open(str(out_path), "wb") as wf:
            self._piper.synthesize_wav(text, wf)
        return probe_duration(self.settings.ffmpeg_binary, out_path)

    def _mock(self, text: str, out_path: str | Path) -> float:
        # ~2.6 words per second is a reasonable speaking rate.
        duration = max(1.0, len(text.split()) / 2.6)
        make_tone(self.settings.ffmpeg_binary, out_path, duration)
        return duration

    def _openai(self, text: str, voice: str, out_path: str | Path) -> float:  # pragma: no cover
        resp = httpx.post(
            f"{self.settings.openai_base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={"model": self.settings.tts_model, "voice": "alloy", "input": text,
                  "response_format": "wav"},
            timeout=180,
        )
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)
        return probe_duration(self.settings.ffmpeg_binary, out_path)


# --------------------------------------------------------------------------- #
# footage sources
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    id: str
    description: str
    source: str
    licence: str
    attribution: str
    ai_generated: bool
    download_url: str | None = None
    color: str = "1e3a8a"
    score: float = 0.0


_MOCK_LIBRARY = [
    ("mk_city", "city street traffic cars busy road commute", "1e3a8a"),
    ("mk_money", "money savings coins finance budget growth", "065f46"),
    ("mk_cloud", "cloud computing servers data center network", "1e40af"),
    ("mk_people", "people working together office team collaboration", "7c2d12"),
    ("mk_nature", "nature landscape trees outdoors calm", "14532d"),
    ("mk_tech", "technology device screen typing laptop", "312e81"),
    ("mk_chart", "charts growth statistics business graph report", "831843"),
    ("mk_food", "food cooking kitchen healthy meal", "9a3412"),
]


class FootageSource:
    name = "base"

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:  # pragma: no cover
        raise NotImplementedError

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:  # pragma: no cover
        raise NotImplementedError


class MockFootageSource(FootageSource):
    name = "mock"

    def __init__(self, settings):
        self.settings = settings
        self.embed = EmbeddingsAdapter(settings)

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        descriptions = [d for _, d, _ in _MOCK_LIBRARY]
        qv = self.embed.embed([query])[0]
        dvs = self.embed.embed(descriptions)
        clips = []
        for (cid, desc, color), dv in zip(_MOCK_LIBRARY, dvs):
            if any(a.lower() in desc.lower() for a in avoid):
                continue
            clips.append(Clip(id=cid, description=desc, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color=color, score=round(cosine(qv, dv), 4)))
        clips.sort(key=lambda c: c.score, reverse=True)
        # If we need more than the library holds, synthesise unique clips so that
        # "no clip is used twice" always holds.
        while len(clips) < k:
            uid = "mk_" + uuid.uuid4().hex[:8]
            clips.append(Clip(id=uid, description=query, source="mock:generator",
                              licence="CC0 (synthetic placeholder)", attribution="IncuBrix mock",
                              ai_generated=True, color="334155", score=0.5))
        return clips[:k]

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        make_color_clip(self.settings.ffmpeg_binary, dest_path, duration_s, clip.color)


class PexelsFootageSource(FootageSource):
    name = "pexels"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pexels_api_key:
            raise RuntimeError("PEXELS_API_KEY is not configured")
        resp = httpx.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": self.settings.pexels_api_key},
            params={"query": query, "per_page": k, "orientation": "landscape"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for video in resp.json().get("videos", []):
            files = sorted(video.get("video_files", []), key=lambda f: f.get("width") or 0)
            best = next((f for f in files if (f.get("width") or 0) >= 1280), files[-1] if files else None)
            if not best:
                continue
            clips.append(Clip(
                id=f"pexels_{video['id']}",
                description=(video.get("url") or query),
                source="pexels",
                licence="Pexels License (free for commercial use)",
                attribution=f"Pexels / {video.get('user', {}).get('name', 'unknown')}",
                ai_generated=False,
                download_url=best.get("link"),
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


class PixabayFootageSource(FootageSource):
    """Free footage via the Pixabay API (Pixabay Content License, commercial OK).

    Pexels has paused new API keys, so Pixabay is the default footage source.
    """
    name = "pixabay"

    def __init__(self, settings):
        self.settings = settings

    def search(self, query: str, k: int, avoid: list[str]) -> list[Clip]:
        if not self.settings.pixabay_api_key:
            raise RuntimeError("PIXABAY_API_KEY is not configured")
        resp = httpx.get(
            "https://pixabay.com/api/videos/",
            params={"key": self.settings.pixabay_api_key, "q": query[:100],
                    "per_page": max(3, min(k, 20)), "safesearch": "true", "lang": "en"},
            timeout=30,
        )
        resp.raise_for_status()
        clips = []
        for hit in resp.json().get("hits", []):
            tags = hit.get("tags", "") or ""
            if any(a.lower() in tags.lower() for a in avoid):
                continue
            vids = hit.get("videos", {}) or {}
            best = (vids.get("large") if (vids.get("large") or {}).get("url") else None) \
                or vids.get("medium") or vids.get("small") or vids.get("tiny")
            if not best or not best.get("url"):
                continue
            clips.append(Clip(
                id=f"pixabay_{hit['id']}",
                description=tags or query,
                source="pixabay",
                licence="Pixabay Content License (free for commercial use)",
                attribution=f"Pixabay / {hit.get('user', 'unknown')}",
                ai_generated=False,
                download_url=best["url"],
                score=1.0,
            ))
        return clips

    def fetch(self, clip: Clip, dest_path: str | Path, duration_s: float) -> None:
        if not clip.download_url:
            raise RuntimeError("clip has no download url")
        with httpx.stream("GET", clip.download_url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


def build_footage_sources(settings) -> list[FootageSource]:
    sources: list[FootageSource] = []
    names = ["mock"] if settings.mock_mode else settings.footage_sources
    for name in names:
        if name == "mock":
            sources.append(MockFootageSource(settings))
        elif name == "pexels":
            sources.append(PexelsFootageSource(settings))
        elif name == "pixabay":
            sources.append(PixabayFootageSource(settings))
    if not sources:
        sources.append(MockFootageSource(settings))
    return sources
