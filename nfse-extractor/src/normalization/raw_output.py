"""Normalization of raw engine artifacts into a shared intermediate format."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.core import ExtractedElement


class NormalizedRawArtifact(BaseModel):
    """Engine-agnostic intermediate representation of raw OCR output."""

    source_element_id: str
    source_engine: str
    source_element_type: str
    raw_text: str
    confidence: float | None = None
    page_number: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    block_hint: int | None = None
    line_hint: int | None = None
    word_hint: int | None = None
    label_hint: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


def normalize_raw_elements(
    elements: list[ExtractedElement],
) -> list[NormalizedRawArtifact]:
    """Convert engine-specific raw elements into a shared intermediate format."""
    normalized: list[NormalizedRawArtifact] = []
    for element in elements:
        metadata = element.metadata
        normalized.append(
            NormalizedRawArtifact(
                source_element_id=element.element_id,
                source_engine=str(metadata.get("source_engine") or "unknown"),
                source_element_type=element.element_type,
                raw_text=element.text,
                confidence=element.confidence,
                page_number=element.page_number,
                bounding_box=element.bounding_box,
                block_hint=_optional_int(metadata.get("block_num")),
                line_hint=_optional_int(metadata.get("line_num")),
                word_hint=_optional_int(metadata.get("word_num")),
                label_hint=_optional_str(metadata.get("raw_label")),
                trace={
                    "element_type": element.element_type,
                    "source_metadata": dict(metadata),
                },
            )
        )
    return normalized


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
