import json

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue
from src.observability import PipelineObserver, format_structured_log


def test_pipeline_observer_collects_stage_timings_and_summary() -> None:
    document = Document(document_id="doc-1")
    observer = PipelineObserver(engine_id="tesseract", document=document)

    with observer.measure_stage("ingestion"):
        pass
    with observer.measure_stage("validation"):
        pass

    summary = observer.build_summary(
        resolved_fields=[
            ResolvedField(field_name="nfse_number", value="123", status="resolved"),
            ResolvedField(field_name="provider_document", value=None, status="conflict"),
        ],
        validation_issues=[
            ValidationIssue(code="warning", message="warn", severity="warning"),
            ValidationIssue(code="error", message="err", severity="error"),
        ],
        decision_result=DecisionResult(document_id="doc-1", decision_status="manual_review_required"),
    )

    assert summary["document_id"] == "doc-1"
    assert summary["engine_id"] == "tesseract"
    assert summary["document_status"] == "manual_review_required"
    assert summary["manual_review_required_count"] == 1
    assert summary["conflict_count"] == 1
    assert summary["validation_issue_counts"] == {"info": 0, "warning": 1, "error": 1}
    assert set(summary["stage_timings_ms"]) == {"ingestion", "validation"}


def test_format_structured_log_returns_single_json_line() -> None:
    line = format_structured_log("document_processed", {"document_id": "doc-1", "engine_id": "dolphin"})
    payload = json.loads(line)

    assert payload["event"] == "document_processed"
    assert payload["document_id"] == "doc-1"
    assert payload["engine_id"] == "dolphin"
