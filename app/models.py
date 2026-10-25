"""Request / response models (the API contract, mirrored in openapi.yaml)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ASPECT_RATIOS = {"16:9", "9:16", "1:1"}
ALLOWED_AUDIO_EXT = {"mp3", "wav", "m4a"}


class Scene(BaseModel):
    text: str
    visual_hint: str | None = None


class CaptionsConfig(BaseModel):
    burn_in: bool = True
    srt: bool = True


class BrollRequest(BaseModel):
    audio_url: str | None = None
    script: str | None = None
    scenes: list[Scene] | None = None
    voice: str = "en-IN-female"
    aspect_ratios: list[str] = Field(default_factory=lambda: ["16:9"])
    style: str | None = None
    avoid: list[str] = Field(default_factory=list)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _exactly_one_input(self):
        provided = [bool(self.audio_url), bool(self.script), bool(self.scenes)]
        if sum(provided) != 1:
            raise ValueError("provide exactly one of audio_url, script or scenes")
        return self


class JobCreated(BaseModel):
    job_id: str
    status: str
    links: dict[str, str] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
