from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from media_asset import MediaAsset


class LLMProvider(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        metadata: Mapping[str, Any] | None = None,
    ) -> str: ...


@dataclass(slots=True)
class CharacterLock:
    name: str
    reference_role: str = "character"
    locked_attributes: tuple[str, ...] = field(default_factory=tuple)
    required_traits: tuple[str, ...] = field(default_factory=tuple)
    negative_constraints: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        parts: list[str] = [self.name]
        if self.locked_attributes:
            parts.append(f"lock {', '.join(self.locked_attributes)}")
        if self.required_traits:
            parts.append(f"preserve {', '.join(self.required_traits)}")
        if self.negative_constraints:
            parts.append(f"avoid {', '.join(self.negative_constraints)}")
        return " — ".join(parts)


@dataclass(slots=True)
class PromptBundle:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "metadata": dict(self.metadata)}


class PromptBuilder:
    def build(
        self,
        *,
        prompt_goal: str,
        scene: str,
        duration_seconds: int | None = None,
        model: str | None = None,
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
    ) -> str:
        sections: list[str] = []
        summary_bits = [prompt_goal]
        if duration_seconds:
            summary_bits.append(f"{duration_seconds}s")
        if pov:
            summary_bits.append(pov)
        if model:
            summary_bits.append(model)
        sections.append("## Prompt Goal\n" + " | ".join(summary_bits))

        if assets:
            lines = [
                f"- {asset.role}: {asset.label or asset.location} ({asset.media_type})"
                for asset in assets
            ]
            sections.append("## Reference Mapping\n" + "\n".join(lines))

        if character_locks:
            sections.append(
                "## Character Consistency\n"
                + "\n".join(f"- {lock.render()}" for lock in character_locks)
            )

        sections.append("## Scene\n" + scene.strip())

        if pov:
            sections.append("## Camera / POV\n- " + pov)

        if continuity_anchors:
            sections.append(
                "## Continuity Anchors\n"
                + "\n".join(f"- {anchor}" for anchor in continuity_anchors)
            )

        if timeline:
            timeline_lines: list[str] = []
            for index, beat in enumerate(timeline, start=1):
                label = beat.get("time") or f"Beat {index}"
                timeline_lines.append(f"### {label}")
                for key in ("action", "micro_expression", "eye_line", "dialogue", "pause", "camera"):
                    if beat.get(key):
                        timeline_lines.append(f"- {key.replace('_', ' ')}: {beat[key]}")
            sections.append("## Timeline\n" + "\n".join(timeline_lines))

        if performance_notes:
            sections.append(
                "## Performance Notes\n"
                + "\n".join(f"- {note}" for note in performance_notes)
            )

        if offscreen_partner_notes:
            sections.append(
                "## Off-screen Partner Performance\n"
                + "\n".join(f"- {note}" for note in offscreen_partner_notes)
            )

        if sound_design:
            sections.append(
                "## Sound Design\n" + "\n".join(f"- {note}" for note in sound_design)
            )

        if negative_constraints:
            sections.append(
                "## Negative Constraints\n"
                + "\n".join(f"- {note}" for note in negative_constraints)
            )

        if model_notes:
            sections.append(
                "## Model-specific Notes\n"
                + "\n".join(f"- {note}" for note in model_notes)
            )

        return "\n\n".join(section for section in sections if section.strip())

    def build_bundle(self, **kwargs: Any) -> PromptBundle:
        metadata = kwargs.pop("metadata", {})
        prompt = self.build(**kwargs)
        return PromptBundle(prompt=prompt, metadata=dict(metadata))


def refine_prompt_with_llm(
    provider: LLMProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    return provider.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        metadata=metadata,
    )


def export_bundle(
    bundle: PromptBundle,
    destination: str | Path,
    *,
    format: str = "json",
) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_path.write_text(
            json.dumps(bundle.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    if format == "md":
        metadata = (
            "```json\n"
            + json.dumps(bundle.metadata, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            if bundle.metadata
            else ""
        )
        output_path.write_text(metadata + bundle.prompt + "\n", encoding="utf-8")
        return output_path

    raise ValueError(f"Unsupported export format: {format}")
