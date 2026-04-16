import json

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue
from src.export import persist_processing_bundle, serialize_jsonable
from src.normalization import NormalizedRawArtifact


def test_serialize_jsonable_preserves_document_and_decision_payloads() -> None:
    document = Document(document_id="doc-1", metadata={"origin": "colab"})
    result = DecisionResult(document_id="doc-1", decision_status="approved_with_warning")

    payload = {
        "document": serialize_jsonable(document),
        "decision_result": serialize_jsonable(result),
    }

    assert payload["document"]["metadata"]["origin"] == "colab"
    assert payload["decision_result"]["decision_status"] == "approved_with_warning"


def test_persist_processing_bundle_writes_all_expected_json_files(tmp_path) -> None:
    document = Document(document_id="doc/1", source_uri="file:///tmp/sample.pdf", metadata={"origin": "local"})
    normalized_artifacts = [
        NormalizedRawArtifact(
            source_element_id="el-1",
            source_engine="tesseract",
            source_element_type="word",
            raw_text="NFSE",
            confidence=0.93,
            trace={"source_metadata": {"page": 1}},
        )
    ]
    resolved_fields = [ResolvedField(field_name="nfse_number", value="123", confidence=0.95)]
    validation_issues = [ValidationIssue(code="invalid_service_code", message="warning", severity="warning")]
    decision_result = DecisionResult(
        document_id="doc/1",
        decision_status="approved_with_warning",
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        metadata={"critical_field_coverage": 1.0},
    )

    files = persist_processing_bundle(
        document=document,
        normalized_artifacts=normalized_artifacts,
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        decision_result=decision_result,
        output_root=tmp_path / "exports",
    )

    assert set(files) == {
        "document",
        "normalized_artifacts",
        "resolved_fields",
        "validation_issues",
        "decision_result",
        "summary",
        "manifest",
    }
    assert files["document"].parent.name == "doc_1"

    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    saved_document = json.loads(files["document"].read_text(encoding="utf-8"))
    saved_decision = json.loads(files["decision_result"].read_text(encoding="utf-8"))
    saved_summary = json.loads(files["summary"].read_text(encoding="utf-8"))

    assert manifest["bundle_version"] == "1.0"
    assert manifest["document_id"] == "doc/1"
    assert saved_document["metadata"]["origin"] == "local"
    assert saved_decision["decision_status"] == "approved_with_warning"
    assert saved_summary["decision_status"] == "approved_with_warning"
    assert saved_summary["normalized_artifact_count"] == 1
