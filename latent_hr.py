from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


class VAECodec(Protocol):
    def encode(self, image: Any) -> Any: ...

    def decode(self, latent: Any) -> Any: ...


Interpolator = Callable[[Any, float, str], Any]


@dataclass(slots=True)
class LatentHRConfig:
    scale: float = 2.0
    interpolation: str = "bilinear"

    def validate(self) -> None:
        if self.scale <= 1.0:
            raise ValueError("scale must be greater than 1.0")


def latent_upscale(
    image: Any,
    *,
    vae: VAECodec,
    scale: float = 2.0,
    interpolation: str = "bilinear",
    interpolator: Interpolator | None = None,
) -> Any:
    config = LatentHRConfig(scale=scale, interpolation=interpolation)
    config.validate()
    latent = vae.encode(image)
    scaled_latent = (interpolator or _default_interpolator)(
        latent,
        config.scale,
        config.interpolation,
    )
    return vae.decode(scaled_latent)


def apply_hr_upscaler(
    image: Any,
    *,
    hr_upscaler: str | None,
    vae: VAECodec,
    scale: float = 2.0,
    interpolation: str = "bilinear",
    interpolator: Interpolator | None = None,
) -> Any:
    if hr_upscaler != "latent":
        return image
    return latent_upscale(
        image,
        vae=vae,
        scale=scale,
        interpolation=interpolation,
        interpolator=interpolator,
    )


def _default_interpolator(latent: Any, scale: float, _: str) -> Any:
    if isinstance(latent, Sequence) and latent and isinstance(latent[0], Sequence):
        return _resize_matrix(latent, scale)
    raise TypeError("Provide a custom interpolator for non-matrix latent types")


def _resize_matrix(latent: Sequence[Sequence[float]], scale: float) -> list[list[float]]:
    source_rows = len(latent)
    source_cols = len(latent[0]) if source_rows else 0
    target_rows = max(1, round(source_rows * scale))
    target_cols = max(1, round(source_cols * scale))
    result: list[list[float]] = []
    for row in range(target_rows):
        source_row = min(source_rows - 1, int(row / scale))
        out_row: list[float] = []
        for col in range(target_cols):
            source_col = min(source_cols - 1, int(col / scale))
            out_row.append(float(latent[source_row][source_col]))
        result.append(out_row)
    return result
