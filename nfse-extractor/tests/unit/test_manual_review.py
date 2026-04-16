import json

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue
from src.export import (
    apply_manual_corrections,
    build_manual_review_artifact,
    write_manual_review_files,
)


def test_build_manual_review_artifact_collects_low_confidence_conflicts_and_issues() -> None:
    document = Document(document_id="doc-1", source_uri="file:///tmp/doc-1.png", metadata={"origin": "colab"})
    resolved_fields = [
        ResolvedField(
            field_name="nfse_number",
            value="123",
            status="resolved",
            confidence=0.55,
            metadata={
                "alternatives": [{"candidate_id": "cand-1", "value": "123", "confidence": 0.55}],
                "selected_candidate_id": "cand-1",
            },
        ),
        ResolvedField(
            field_name="provider_document",
            value=None,
            status="conflict",
            confidence=0.51,
            metadata={
                "alternatives": [
                    {"candidate_id": "cand-a", "value": "12.345.678/0001-90", "confidence": 0.51},
                    {"candidate_id": "cand-b", "value": "98.765.432/0001-10", "confidence": 0.49},
                ]
            },
        ),
    ]
    validation_issues = [
        ValidationIssue(code="invalid_document_id", message="warning", severity="warning", field_name="provider_document")
    ]

    artifact = build_manual_review_artifact(
        document=document,
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        decision_result=DecisionResult(document_id="doc-1", decision_status="manual_review_required"),
    )

    assert artifact["document_id"] == "doc-1"
    assert artifact["decision_status"] == "manual_review_required"
    assert artifact["review_required"] is True
    assert artifact["review_summary"]["conflict_count"] == 1
    assert artifact["low_confidence_fields"][0]["field_name"] == "nfse_number"
    assert artifact["conflicts"][0]["field_name"] == "provider_document"
    assert artifact["conflicts"][0]["suggested_candidates"][0]["candidate_id"] == "cand-a"
    assert artifact["validation_issues"][0]["field_name"] == "provider_document"
    assert "provider_document" in artifact["review_candidates"]


def test_write_manual_review_files_and_apply_corrections(tmp_path) -> None:
    document = Document(document_id="doc/1")
    resolved_fields = [
        ResolvedField(field_name="provider_document", value=None, status="conflict"),
        ResolvedField(field_name="nfse_number", value="123", status="resolved", confidence=0.95),
    ]
    validation_issues = [ValidationIssue(code="field_conflict", message="conflict", severity="error", field_name="provider_document")]

    files = write_manual_review_files(
        document=document,
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        output_root=tmp_path / "review",
    )

    assert files["artifact_path"].exists()
    assert files["corrections_template_path"].exists()

    template_payload = json.loads(files["corrections_template_path"].read_text(encoding="utf-8"))
    template_payload["review_metadata"]["reviewer"] = "analyst@example.com"
    template_payload["review_metadata"]["reviewed_at"] = "2026-04-16T10:00:00Z"
    template_payload["corrections"][0]["corrected_value"] = "12.345.678/0001-90"
    template_payload["corrections"][0]["notes"] = "Checked manually"

    updated_fields = apply_manual_corrections(resolved_fields, template_payload)
    updated_provider = next(field for field in updated_fields if field.field_name == "provider_document")

    assert updated_provider.value == "12.345.678/0001-90"
    assert updated_provider.status == "resolved"
    assert updated_provider.confidence is None
    assert updated_provider.metadata["manual_review"]["corrected"] is True
    assert updated_provider.metadata["manual_review"]["review_metadata"]["reviewer"] == "analyst@example.com"
