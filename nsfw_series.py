from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from media_asset import MediaAsset
from workflow import CharacterLock, PromptBuilder

VariationFactory = Callable[[int, "NSFWSeriesConfig"], Sequence[str]]


@dataclass(slots=True)
class NSFWSeriesConfig:
    prompt_goal: str
    base_scene: str
    total_variations: int = 4
    duration_seconds: int = 15
    model: str | None = None
    pov: str | None = None
    continuity_anchors: tuple[str, ...] = field(default_factory=tuple)
    performance_notes: tuple[str, ...] = field(default_factory=tuple)
    negative_constraints: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.total_variations <= 0:
            raise ValueError("total_variations must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(slots=True)
class SeriesPrompt:
    index: int
    variation_notes: tuple[str, ...]
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SeriesGenerator:
    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder | None = None,
        variation_factories: Mapping[str, VariationFactory] | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.variation_factories = dict(variation_factories or {})

    def register_factory(self, name: str, factory: VariationFactory) -> None:
        self.variation_factories[name] = factory

    def build_series(
        self,
        config: NSFWSeriesConfig,
        *,
        assets: Sequence[MediaAsset] = (),
        character_locks: Sequence[CharacterLock] = (),
        timeline_factory: Callable[[int, NSFWSeriesConfig], Sequence[Mapping[str, Any]]] | None = None,
        offscreen_partner_notes: Sequence[str] = (),
        sound_design: Sequence[str] = (),
    ) -> list[SeriesPrompt]:
        config.validate()
        prompts: list[SeriesPrompt] = []
        for index in range(config.total_variations):
            variation_notes = tuple(self._collect_variations(index, config))
            timeline = timeline_factory(index, config) if timeline_factory else ()
            prompt = self.prompt_builder.build(
                prompt_goal=config.prompt_goal,
                scene=config.base_scene,
                duration_seconds=config.duration_seconds,
                model=config.model,
                pov=config.pov,
                assets=assets,
                character_locks=character_locks,
                continuity_anchors=config.continuity_anchors,
                timeline=timeline,
                performance_notes=[*config.performance_notes, *variation_notes],
                offscreen_partner_notes=offscreen_partner_notes,
                sound_design=sound_design,
                negative_constraints=config.negative_constraints,
                model_notes=variation_notes,
            )
            prompts.append(
                SeriesPrompt(
                    index=index,
                    variation_notes=variation_notes,
                    prompt=prompt,
                    metadata={"variation_index": index, **config.metadata},
                )
            )
        return prompts

    def _collect_variations(
        self,
        index: int,
        config: NSFWSeriesConfig,
    ) -> Iterable[str]:
        for factory in self.variation_factories.values():
            yield from factory(index, config)


def cycle_variations(name: str, values: Sequence[str]) -> VariationFactory:
    if not values:
        raise ValueError(f"{name} requires at least one value")

    def factory(index: int, _: NSFWSeriesConfig) -> Sequence[str]:
        return (f"{name}: {values[index % len(values)]}",)

    return factory


def paired_variations(name: str, pairs: Sequence[tuple[str, str]]) -> VariationFactory:
    if not pairs:
        raise ValueError(f"{name} requires at least one pair")

    def factory(index: int, _: NSFWSeriesConfig) -> Sequence[str]:
        left, right = pairs[index % len(pairs)]
        return (f"{name}: {left}", f"{name} secondary: {right}")

    return factory
