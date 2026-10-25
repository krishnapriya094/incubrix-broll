FROM python:3.12-slim

# real ffmpeg in the image; the app also works without it via imageio-ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt requirements-free.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir -r requirements-free.txt

# Bake the Piper voice model into the image so the container can narrate on
# first run without downloading anything.
RUN python -m piper.download_voices en_US-lessac-medium --data-dir /app/piper_voices

COPY app ./app
COPY tests ./tests
COPY testdata ./testdata
COPY openapi.yaml README.md ./

# Sensible defaults for a hosted deploy. Override in the host's dashboard.
# WHISPER_MODEL=tiny keeps memory low on small instances.
ENV DATA_DIR=/data \
    MOCK_MODE=false \
    STORAGE_BACKEND=local \
    STT_BACKEND=faster_whisper \
    TTS_BACKEND=piper \
    EMBEDDINGS_BACKEND=fastembed \
    FOOTAGE_SOURCES=pixabay \
    WHISPER_MODEL=tiny \
    PIPER_VOICE=/app/piper_voices/en_US-lessac-medium.onnx

EXPOSE 8000
# Hosts like Render inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
