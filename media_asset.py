from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def infer_media_type(location: str | Path) -> str:
    suffix = Path(location).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    return "file"


@dataclass(frozen=True, slots=True)
class MediaAsset:
    location: str
    role: str = "reference"
    media_type: str = "image"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        role: str = "reference",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MediaAsset":
        return cls(
            location=str(path),
            role=role,
            media_type=infer_media_type(path),
            label=label,
            metadata=metadata or {},
        )

    @property
    def path(self) -> Path:
        return Path(self.location)

    def exists(self) -> bool:
        return self.path.exists()

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "role": self.role,
            "media_type": self.media_type,
            "label": self.label,
            "metadata": dict(self.metadata),
        }
