"""Helpers for serializing and deserializing raw extraction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.core import ExtractedElement


def serialize_extracted_elements(elements: list[ExtractedElement]) -> list[dict]:
    """Serialize extracted elements into JSON-friendly dictionaries."""
    return [element.model_dump(mode="json") for element in elements]


def write_extracted_elements_json(
    elements: list[ExtractedElement],
    output_path: str | Path,
) -> Path:
    """Write serialized raw extraction artifacts to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_extracted_elements(elements), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_extracted_elements_json(input_path: str | Path) -> list[ExtractedElement]:
    """Load raw extraction artifacts from a JSON file written by write_extracted_elements_json.

    Each record in the JSON array is validated against ExtractedElement so the
    returned objects are identical to what the adapter would have produced.
    """
    path = Path(input_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ExtractedElement.model_validate(record) for record in data]


def write_text_log(content: str, output_path: str | Path) -> Path:
    """Write a plain-text log to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
