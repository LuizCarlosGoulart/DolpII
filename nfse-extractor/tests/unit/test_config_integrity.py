from pathlib import Path

from src.validation import validate_config_integrity


def test_default_configs_pass_integrity_validation() -> None:
    assert validate_config_integrity() == []


def test_integrity_validator_reports_unknown_critical_field(tmp_path: Path) -> None:
    source_dir = Path(__file__).resolve().parents[2] / "configs"
    for source in source_dir.glob("*.yaml"):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    critical_fields_path = tmp_path / "critical_fields.yaml"
    content = critical_fields_path.read_text(encoding="utf-8")
    critical_fields_path.write_text(
        content + "  - unknown_field\n",
        encoding="utf-8",
    )

    issues = validate_config_integrity(tmp_path)

    assert "Unknown critical field: unknown_field" in issues
