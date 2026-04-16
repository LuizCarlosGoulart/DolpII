"""Minimal manual review artifacts and correction helpers for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue


def build_manual_review_artifact(
    *,
    document: Document,
    resolved_fields: list[ResolvedField],
    validation_issues: list[ValidationIssue],
    decision_result: DecisionResult | None = None,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """Build a compact review artifact for notebook or file-based correction."""
    low_confidence_fields = [
        _field_review_entry(field)
        for field in resolved_fields
        if field.status == "resolved"
        and field.value is not None
        and field.confidence is not None
        and field.confidence < confidence_threshold
    ]
    conflicts = [
        _field_review_entry(field)
        for field in resolved_fields
        if field.status == "conflict"
    ]
    review_candidates = sorted(
        {
            *[entry["field_name"] for entry in low_confidence_fields],
            *[entry["field_name"] for entry in conflicts],
            *[issue.field_name for issue in validation_issues if issue.field_name],
        }
    )
    review_required = bool(review_candidates)

    return {
        "document_id": document.document_id,
        "source_uri": document.source_uri,
        "document_metadata": document.metadata,
        "decision_status": decision_result.decision_status if decision_result is not None else None,
        "review_required": review_required,
        "review_summary": {
            "low_confidence_field_count": len(low_confidence_fields),
            "conflict_count": len(conflicts),
            "validation_issue_count": len(validation_issues),
        },
        "low_confidence_fields": low_confidence_fields,
        "conflicts": conflicts,
        "validation_issues": [issue.model_dump(mode="json") for issue in validation_issues],
        "review_candidates": review_candidates,
        "corrections_template": {
            "document_id": document.document_id,
            "review_metadata": {
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
            },
            "corrections": [
                {
                    "field_name": field_name,
                    "corrected_value": "",
                    "notes": "",
                }
                for field_name in review_candidates
            ],
        },
    }


def write_manual_review_files(
    *,
    document: Document,
    resolved_fields: list[ResolvedField],
    validation_issues: list[ValidationIssue],
    output_root: str | Path,
    decision_result: DecisionResult | None = None,
    confidence_threshold: float = 0.60,
) -> dict[str, Path]:
    """Write review artifact and correction template to disk."""
    artifact = build_manual_review_artifact(
        document=document,
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        decision_result=decision_result,
        confidence_threshold=confidence_threshold,
    )
    review_dir = Path(output_root) / _safe_name(document.document_id)
    review_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = _write_json(review_dir / "manual_review.json", artifact)
    corrections_path = _write_json(
        review_dir / "manual_corrections.template.json",
        artifact["corrections_template"],
    )
    return {
        "review_dir": review_dir,
        "artifact_path": artifact_path,
        "corrections_template_path": corrections_path,
    }


def apply_manual_corrections(
    resolved_fields: list[ResolvedField],
    corrections_payload: dict[str, Any],
) -> list[ResolvedField]:
    """Apply structured manual corrections to resolved fields."""
    corrections_by_field = {
        item["field_name"]: item
        for item in corrections_payload.get("corrections", [])
        if item.get("field_name")
    }
    field_index = {field.field_name: field for field in resolved_fields}
    updated: list[ResolvedField] = []
    review_metadata = corrections_payload.get("review_metadata", {})

    for field in resolved_fields:
        correction = corrections_by_field.get(field.field_name)
        if correction is None or str(correction.get("corrected_value", "")).strip() == "":
            updated.append(field)
            continue

        metadata = dict(field.metadata)
        metadata["manual_review"] = {
            "corrected": True,
            "notes": correction.get("notes", ""),
            "previous_value": field.value,
            "review_metadata": {
                "reviewer": review_metadata.get("reviewer", ""),
                "reviewed_at": review_metadata.get("reviewed_at", ""),
                "review_notes": review_metadata.get("review_notes", ""),
            },
        }
        updated.append(
            field.model_copy(
                update={
                    "value": str(correction["corrected_value"]).strip(),
                    "status": "resolved",
                    "confidence": None,
                    "metadata": metadata,
                }
            )
        )

    for field_name, correction in corrections_by_field.items():
        if field_name in field_index or str(correction.get("corrected_value", "")).strip() == "":
            continue
        updated.append(
            ResolvedField(
                field_name=field_name,
                value=str(correction["corrected_value"]).strip(),
                status="resolved",
                confidence=None,
                metadata={
                    "manual_review": {
                        "corrected": True,
                        "notes": correction.get("notes", ""),
                        "previous_value": None,
                        "review_metadata": {
                            "reviewer": review_metadata.get("reviewer", ""),
                            "reviewed_at": review_metadata.get("reviewed_at", ""),
                            "review_notes": review_metadata.get("review_notes", ""),
                        },
                    }
                },
            )
        )

    return sorted(updated, key=lambda field: field.field_name)


def _field_review_entry(field: ResolvedField) -> dict[str, Any]:
    metadata = field.metadata or {}
    return {
        "field_name": field.field_name,
        "value": field.value,
        "status": field.status,
        "confidence": field.confidence,
        "suggested_candidates": metadata.get("alternatives", []),
        "selected_candidate_id": metadata.get("selected_candidate_id"),
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value) or "document"
