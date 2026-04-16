"""Filesystem-friendly JSON persistence for extraction experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue
from src.normalization.raw_output import NormalizedRawArtifact


def serialize_jsonable(value: Any) -> Any:
    """Convert supported contract objects into JSON-friendly values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_jsonable(item) for item in value]
    return value

def persist_processing_bundle(
    *,
    document: Document,
    normalized_artifacts: list[NormalizedRawArtifact],
    resolved_fields: list[ResolvedField],
    validation_issues: list[ValidationIssue],
    decision_result: DecisionResult,
    output_root: str | Path,
) -> dict[str, Path]:
    """Persist one document processing bundle to a configurable filesystem root."""
    bundle_dir = Path(output_root) / _safe_name(document.document_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "document": _write_json(bundle_dir / "document.json", serialize_jsonable(document)),
        "normalized_artifacts": _write_json(
            bundle_dir / "normalized_artifacts.json",
            serialize_jsonable(normalized_artifacts),
        ),
        "resolved_fields": _write_json(
            bundle_dir / "resolved_fields.json",
            serialize_jsonable(resolved_fields),
        ),
        "validation_issues": _write_json(
            bundle_dir / "validation_issues.json",
            serialize_jsonable(validation_issues),
        ),
        "decision_result": _write_json(
            bundle_dir / "decision_result.json",
            serialize_jsonable(decision_result),
        ),
        "summary": _write_json(
            bundle_dir / "summary.json",
            {
                "document_id": document.document_id,
                "selected_source": decision_result.selected_source,
                "decision_status": decision_result.decision_status,
                "score": decision_result.score,
                "validation_issue_count": len(validation_issues),
                "resolved_field_count": len(resolved_fields),
                "normalized_artifact_count": len(normalized_artifacts),
            },
        ),
    }

    files["manifest"] = _write_json(
        bundle_dir / "manifest.json",
        {
            "bundle_version": "1.0",
            "document_id": document.document_id,
            "bundle_dir": str(bundle_dir),
            "files": {name: str(path) for name, path in files.items()},
        },
    )
    return files


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "document"
