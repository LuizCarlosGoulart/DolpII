import json

from src.core import (
    DecisionEngine,
    DecisionResult,
    Document,
    ExperimentComparisonRunner,
    ExtractedElement,
    FieldCandidate,
    FieldResolver,
    OutputNormalizer,
    ResolvedField,
    ValidationIssue,
    Validator,
)
from src.preprocessing import PreprocessedDocument, PreprocessedPage


class _StubEngine:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.seen_preprocessed_ids: list[int] = []

    def extract_preprocessed(self, preprocessed_document: PreprocessedDocument) -> list[ExtractedElement]:
        self.seen_preprocessed_ids.append(id(preprocessed_document))
        file_name = preprocessed_document.document.metadata["file_name"]
        return [
            ExtractedElement(
                element_id=f"{preprocessed_document.document.document_id}:raw:0",
                element_type="text",
                text=self.outputs[file_name],
                metadata={"source_engine": "stub"},
            )
        ]


class _StubNormalizer(OutputNormalizer):
    def normalize(self, document: Document, elements: list[ExtractedElement]) -> list[FieldCandidate]:
        return [
            FieldCandidate(
                candidate_id=f"{document.document_id}:cand:0",
                field_name="nfse_number",
                value=elements[0].text,
                source_element_ids=[elements[0].element_id],
                confidence=0.95,
            )
        ]


class _StubResolver(FieldResolver):
    def resolve(self, document: Document, candidates: list[FieldCandidate]) -> list[ResolvedField]:
        value = candidates[0].value
        if value == "CONFLICT":
            return [ResolvedField(field_name="nfse_number", value=None, status="conflict", confidence=0.4)]
        return [ResolvedField(field_name="nfse_number", value=value, status="resolved", confidence=0.95)]


class _StubValidator(Validator):
    def validate(self, document: Document, fields: list[ResolvedField]) -> list[ValidationIssue]:
        if fields[0].value == "WARN":
            return [ValidationIssue(code="warn", message="warning", severity="warning", field_name="nfse_number")]
        return []


class _StubDecisionEngine(DecisionEngine):
    def decide(
        self,
        document: Document,
        fields: list[ResolvedField],
        issues: list[ValidationIssue],
    ) -> DecisionResult:
        if any(issue.severity == "error" for issue in issues):
            status = "rejected"
        elif any(field.status == "conflict" for field in fields):
            status = "manual_review_required"
        elif any(issue.severity == "warning" for issue in issues):
            status = "approved_with_warning"
        else:
            status = "auto_approved"
        return DecisionResult(
            document_id=document.document_id,
            decision_status=status,
            resolved_fields=fields,
            validation_issues=issues,
            score=0.9,
        )


def test_experiment_runner_compares_engines_through_shared_pipeline(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    doc_a = dataset_dir / "a.png"
    doc_b = dataset_dir / "b.png"
    doc_a.write_bytes(b"")
    doc_b.write_bytes(b"")

    preprocessed_ids: list[int] = []

    def preprocessor(document: Document) -> PreprocessedDocument:
        result = PreprocessedDocument(
            document=document,
            pages=[PreprocessedPage(page_number=1, image=object())],
            metadata={"source": "shared"},
        )
        preprocessed_ids.append(id(result))
        return result

    tesseract = _StubEngine({"a.png": "OK", "b.png": "WARN"})
    dolphin = _StubEngine({"a.png": "OK", "b.png": "CONFLICT"})

    def correctness_hook(field_name, resolved_field, expected_value, document):
        if field_name != "nfse_number" or resolved_field is None or resolved_field.value is None:
            return None
        return resolved_field.value == ("OK" if document.metadata["file_name"] == "a.png" else "WARN")

    runner = ExperimentComparisonRunner(
        engines={"tesseract": tesseract, "dolphin": dolphin},
        output_normalizer=_StubNormalizer(),
        field_resolver=_StubResolver(),
        validator=_StubValidator(),
        decision_engine=_StubDecisionEngine(),
        output_root=tmp_path / "outputs",
        preprocessor=preprocessor,
        field_correctness_hook=correctness_hook,
    )

    result = runner.run([doc_a, doc_b], experiment_name="comparison")

    assert result["experiment_summary"]["total_documents"] == 2
    assert result["experiment_summary"]["engine_summaries"]["dolphin"]["manual_review_rate"] == 0.5
    assert result["experiment_summary"]["engine_summaries"]["tesseract"]["status_counts"]["approved_with_warning"] == 1

    assert len(preprocessed_ids) == 2
    assert tesseract.seen_preprocessed_ids == preprocessed_ids
    assert dolphin.seen_preprocessed_ids == preprocessed_ids

    document_metrics = result["document_metrics"]
    assert [item["engine_id"] for item in document_metrics[:2]] == ["dolphin", "tesseract"]
    assert any(item["engine_id"] == "dolphin" and item["document_status"] == "manual_review_required" for item in document_metrics)
    assert any(item["engine_id"] == "tesseract" and item["document_status"] == "approved_with_warning" for item in document_metrics)
    assert all("shared_preprocessing_time_ms" in item for item in document_metrics)
    assert all("engine_processing_time_ms" in item for item in document_metrics)
    assert all("sample_index" in item for item in document_metrics)

    field_metrics = result["field_metrics"]
    tesseract_nfse = next(item for item in field_metrics if item["engine_id"] == "tesseract" and item["field_name"] == "nfse_number")
    dolphin_nfse = next(item for item in field_metrics if item["engine_id"] == "dolphin" and item["field_name"] == "nfse_number")
    assert tesseract_nfse["fill_rate"] == 1.0
    assert dolphin_nfse["conflict_rate"] == 0.5
    assert tesseract_nfse["correctness_evaluated_count"] == 2

    summary_payload = json.loads(result["experiment_summary_path"].read_text(encoding="utf-8"))
    manifest_payload = json.loads(result["experiment_manifest_path"].read_text(encoding="utf-8"))
    assert summary_payload["engines"] == ["dolphin", "tesseract"]
    assert manifest_payload["engine_ids"] == ["dolphin", "tesseract"]
    assert manifest_payload["dataset_paths"] == [str(doc_a), str(doc_b)]
    assert result["document_metrics_path"].exists()
    assert result["field_metrics_path"].exists()
