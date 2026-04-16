"""Structural validation for project configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core import load_field_dictionary


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def validate_config_integrity(config_dir: str | Path | None = None) -> list[str]:
    """Return human-readable integrity issues for the configured YAML files."""
    root = Path(config_dir) if config_dir is not None else default_config_dir()
    dictionary = load_field_dictionary(root / "field_dictionary.yaml")
    field_names = set(dictionary.by_internal_name())

    issues: list[str] = []

    aliases_payload = _load_yaml(root / "field_aliases.yaml").get("aliases", {})
    seen_aliases: dict[str, str] = {}
    for field_name, aliases in aliases_payload.items():
        if field_name not in field_names:
            issues.append(f"Unknown alias field: {field_name}")
            continue
        for alias in aliases:
            previous = seen_aliases.get(alias)
            if previous is not None and previous != field_name:
                issues.append(f"Alias collision for '{alias}': {previous} vs {field_name}")
            seen_aliases[alias] = field_name

    patterns_payload = _load_yaml(root / "field_patterns.yaml").get("patterns", {})
    for field_name in patterns_payload:
        if field_name not in field_names:
            issues.append(f"Unknown pattern field: {field_name}")

    critical_fields = _load_yaml(root / "critical_fields.yaml").get("critical_fields", [])
    for field_name in critical_fields:
        if field_name not in field_names:
            issues.append(f"Unknown critical field: {field_name}")

    scoring_payload = _load_yaml(root / "scoring_rules.yaml")
    for section_name in ("evidence_weights", "confidence_weights", "penalties"):
        for key, value in scoring_payload.get(section_name, {}).items():
            if not isinstance(value, (int, float)) or value < 0:
                issues.append(f"Invalid scoring weight for {section_name}.{key}")

    thresholds = _load_yaml(root / "decision_thresholds.yaml").get("thresholds", {})
    bounded_thresholds = (
        "minimum_field_confidence",
        "minimum_decision_score",
        "minimum_critical_field_coverage",
    )
    for key in bounded_thresholds:
        value = thresholds.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            issues.append(f"Invalid bounded threshold: {key}")

    for key in ("maximum_error_issues", "maximum_warning_issues"):
        value = thresholds.get(key)
        if not isinstance(value, int) or value < 0:
            issues.append(f"Invalid issue threshold: {key}")

    return issues
