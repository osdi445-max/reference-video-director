from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from media_asset import MediaAsset
from workflow import CharacterLock, LLMProvider, PromptBuilder, export_bundle

GeneratorCallable = Callable[["GenerationConfig", str], Sequence[str | Path | MediaAsset]]


@dataclass(slots=True)
class GenerationConfig:
    prompt: str = ""
    negative_prompt: str = ""
    output_dir: str | Path = "outputs"
    width: int = 832
    height: int = 1216
    num_outputs: int = 1
    steps: int = 30
    seed: int | None = None
    model: str | None = None
    duration_seconds: int | None = None
    hr_upscaler: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_output_dir(self) -> Path:
        return Path(self.output_dir)

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")


@dataclass(slots=True)
class GenerationResult:
    config: GenerationConfig
    prompt: str
    negative_prompt: str
    planned_outputs: list[str] = field(default_factory=list)
    assets: list[MediaAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    exported_to: Path | None = None

    @property
    def success(self) -> bool:
        return not self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "prompt": self.config.prompt,
                "negative_prompt": self.config.negative_prompt,
                "output_dir": str(self.config.output_dir),
                "width": self.config.width,
                "height": self.config.height,
                "num_outputs": self.config.num_outputs,
                "steps": self.config.steps,
                "seed": self.config.seed,
                "model": self.config.model,
                "duration_seconds": self.config.duration_seconds,
                "hr_upscaler": self.config.hr_upscaler,
                "metadata": dict(self.config.metadata),
            },
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "planned_outputs": list(self.planned_outputs),
            "assets": [asset.as_dict() for asset in self.assets],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "exported_to": str(self.exported_to) if self.exported_to else None,
        }


class MinimalPipeline:
    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder | None = None,
        llm_provider: LLMProvider | None = None,
        generator: GeneratorCallable | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_provider = llm_provider
        self.generator = generator

    def prepare_prompt(
        self,
        config: GenerationConfig,
        *,
        prompt_goal: str,
        scene: str,
        pov: str | None = None,
        assets: Sequence[MediaAsset] = (),
        character_locks: Sequence[CharacterLock] = (),
        continuity_anchors: Sequence[str] = (),
        timeline: Sequence[Mapping[str, Any]] = (),
        performance_notes: Sequence[str] = (),
        offscreen_partner_notes: Sequence[str] = (),
        sound_design: Sequence[str] = (),
        negative_constraints: Sequence[str] = (),
        model_notes: Sequence[str] = (),
        use_llm: bool = False,
    ) -> str:
        prompt = config.prompt.strip() or self.prompt_builder.build(
            prompt_goal=prompt_goal,
            scene=scene,
            duration_seconds=config.duration_seconds,
            model=config.model,
            pov=pov,
            assets=assets,
            character_locks=character_locks,
            continuity_anchors=continuity_anchors,
            timeline=timeline,
            performance_notes=performance_notes,
            offscreen_partner_notes=offscreen_partner_notes,
            sound_design=sound_design,
            negative_constraints=negative_constraints,
            model_notes=model_notes,
        )

        if use_llm:
            if self.llm_provider is None:
                raise ValueError("An LLM provider is required when use_llm=True")
            prompt = self.llm_provider.complete(
                system_prompt="Refine the prompt without changing its intent.",
                user_prompt=prompt,
                metadata={"model": config.model, "duration_seconds": config.duration_seconds},
            )

        return prompt

    def run(
        self,
        config: GenerationConfig,
        *,
        prompt_goal: str,
        scene: str,
        pov: str | None = None,
        assets: Sequence[MediaAsset] = (),
        character_locks: Sequence[CharacterLock] = (),
        continuity_anchors: Sequence[str] = (),
        timeline: Sequence[Mapping[str, Any]] = (),
        performance_notes: Sequence[str] = (),
        offscreen_partner_notes: Sequence[str] = (),
        sound_design: Sequence[str] = (),
        negative_constraints: Sequence[str] = (),
        model_notes: Sequence[str] = (),
        export_path: str | Path | None = None,
        use_llm: bool = False,
    ) -> GenerationResult:
        config.validate()
        prompt = self.prepare_prompt(
            config,
            prompt_goal=prompt_goal,
            scene=scene,
            pov=pov,
            assets=assets,
            character_locks=character_locks,
            continuity_anchors=continuity_anchors,
            timeline=timeline,
            performance_notes=performance_notes,
            offscreen_partner_notes=offscreen_partner_notes,
            sound_design=sound_design,
            negative_constraints=negative_constraints,
            model_notes=model_notes,
            use_llm=use_llm,
        )
        planned_outputs, generated_assets = self._plan_outputs(config, prompt)
        result = GenerationResult(
            config=config,
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            planned_outputs=planned_outputs,
            assets=generated_assets,
            metadata={"model": config.model, "hr_upscaler": config.hr_upscaler},
        )

        if export_path is not None:
            bundle = self.prompt_builder.build_bundle(
                prompt_goal=prompt_goal,
                scene=scene,
                duration_seconds=config.duration_seconds,
                model=config.model,
                pov=pov,
                assets=assets,
                character_locks=character_locks,
                continuity_anchors=continuity_anchors,
                timeline=timeline,
                performance_notes=performance_notes,
                offscreen_partner_notes=offscreen_partner_notes,
                sound_design=sound_design,
                negative_constraints=negative_constraints,
                model_notes=model_notes,
                metadata=result.as_dict(),
            )
            result.exported_to = export_bundle(bundle, export_path)

        return result

    def _plan_outputs(
        self,
        config: GenerationConfig,
        prompt: str,
    ) -> tuple[list[str], list[MediaAsset]]:
        if self.generator is None:
            suffix = ".latent.png" if config.hr_upscaler == "latent" else ".png"
            planned = [
                str(config.normalized_output_dir() / f"generation_{index + 1}{suffix}")
                for index in range(config.num_outputs)
            ]
            return planned, []

        raw_outputs = self.generator(config, prompt)
        planned_outputs: list[str] = []
        assets: list[MediaAsset] = []
        for item in raw_outputs:
            if isinstance(item, MediaAsset):
                assets.append(item)
                planned_outputs.append(item.location)
            else:
                planned_outputs.append(str(item))
        return planned_outputs, assets
