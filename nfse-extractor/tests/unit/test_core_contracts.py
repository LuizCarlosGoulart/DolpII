from src.core import (
    DecisionEngine,
    DecisionResult,
    Document,
    Exporter,
    ExtractedElement,
    ExtractionEngine,
    FieldCandidate,
    FieldResolver,
    OutputNormalizer,
    ResolvedField,
    ValidationIssue,
    Validator,
)


def test_core_models_support_minimal_instantiation_and_independent_defaults() -> None:
    document = Document(document_id="doc-1")
    element = ExtractedElement(
        element_id="el-1",
        element_type="text",
        text="NFSE",
    )
    candidate = FieldCandidate(
        candidate_id="cand-1",
        field_name="issuer_name",
        value="Acme Ltda",
    )
    resolved = ResolvedField(field_name="issuer_name", value="Acme Ltda")
    issue = ValidationIssue(
        code="missing-tax-id",
        message="Tax id not found",
        severity="warning",
    )
    result = DecisionResult(document_id=document.document_id)

    other_document = Document(document_id="doc-2")
    document.metadata["source"] = "fixture"

    assert element.page_number is None
    assert candidate.source_element_ids == []
    assert resolved.status == "resolved"
    assert issue.field_name is None
    assert result.decision_status == "manual_review_required"
    assert result.resolved_fields == []
    assert other_document.metadata == {}


def test_core_interfaces_can_be_implemented_with_contract_shapes() -> None:
    class StubExtractionEngine(ExtractionEngine):
        def extract(self, document: Document) -> list[ExtractedElement]:
            return [
                ExtractedElement(
                    element_id="el-1",
                    element_type="text",
                    text=document.document_id,
                )
            ]

    class StubNormalizer(OutputNormalizer):
        def normalize(
            self,
            document: Document,
            elements: list[ExtractedElement],
        ) -> list[FieldCandidate]:
            return [
                FieldCandidate(
                    candidate_id="cand-1",
                    field_name="document_id",
                    value=elements[0].text,
                    source_element_ids=[elements[0].element_id],
                )
            ]

    class StubResolver(FieldResolver):
        def resolve(
            self,
            document: Document,
            candidates: list[FieldCandidate],
        ) -> list[ResolvedField]:
            return [
                ResolvedField(
                    field_name=candidates[0].field_name,
                    value=candidates[0].value,
                    source_candidate_ids=[candidates[0].candidate_id],
                )
            ]

    class StubValidator(Validator):
        def validate(
            self,
            document: Document,
            fields: list[ResolvedField],
        ) -> list[ValidationIssue]:
            return []

    class StubDecisionEngine(DecisionEngine):
        def decide(
            self,
            document: Document,
            fields: list[ResolvedField],
            issues: list[ValidationIssue],
        ) -> DecisionResult:
            return DecisionResult(
                document_id=document.document_id,
                selected_source="stub",
                resolved_fields=fields,
                validation_issues=issues,
            )

    class StubExporter(Exporter):
        def export(
            self,
            document: Document,
            result: DecisionResult,
        ) -> dict[str, str]:
            return {
                "document_id": document.document_id,
                "selected_source": result.selected_source or "",
            }

    document = Document(document_id="doc-1")
    elements = StubExtractionEngine().extract(document)
    candidates = StubNormalizer().normalize(document, elements)
    fields = StubResolver().resolve(document, candidates)
    issues = StubValidator().validate(document, fields)
    result = StubDecisionEngine().decide(document, fields, issues)
    exported = StubExporter().export(document, result)

    assert elements[0].text == "doc-1"
    assert candidates[0].candidate_id == "cand-1"
    assert candidates[0].field_name == "document_id"
    assert fields[0].value == "doc-1"
    assert fields[0].source_candidate_ids == ["cand-1"]
    assert result.selected_source == "stub"
    assert exported["document_id"] == "doc-1"
