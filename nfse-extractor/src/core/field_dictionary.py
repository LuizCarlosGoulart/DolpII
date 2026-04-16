"""Config-driven loader for the canonical NFS-e field dictionary."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class CanonicalFieldDefinition(BaseModel):
    """Canonical metadata for one structured NFS-e field."""

    internal_name: str
    category: str
    description: str
    type: str
    required: bool
    expected_format: str
    aliases: list[str] = Field(default_factory=list)
    example_values: list[str] = Field(default_factory=list)
    validation_hints: list[str] = Field(default_factory=list)
    context_hints: list[str] = Field(default_factory=list)


class FieldDictionary(BaseModel):
    """Canonical field dictionary loaded from configuration."""

    version: str
    fields: list[CanonicalFieldDefinition]

    def by_internal_name(self) -> dict[str, CanonicalFieldDefinition]:
        return {field.internal_name: field for field in self.fields}

    def alias_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for field in self.fields:
            index[field.internal_name] = field.internal_name
            for alias in field.aliases:
                index[alias] = field.internal_name
        return index


def default_field_dictionary_path() -> Path:
    """Return the project-default field dictionary config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "field_dictionary.yaml"


def load_field_dictionary(path: str | Path | None = None) -> FieldDictionary:
    """Load the canonical field dictionary from YAML configuration."""
    config_path = Path(path) if path is not None else default_field_dictionary_path()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return FieldDictionary.model_validate(payload)
