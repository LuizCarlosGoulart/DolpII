from src.core import Document, ResolvedField, ValidationIssue
from src.decision import ConfigDrivenDecisionEngine


def _base_fields() -> list[ResolvedField]:
    return [
        ResolvedField(field_name="nfse_number", value="123", confidence=0.95),
        ResolvedField(field_name="verification_code", value="AB12CD34", confidence=0.94),
        ResolvedField(field_name="issue_date", value="15/04/2026", confidence=0.92),
        ResolvedField(field_name="provider_name", value="ACME LTDA", confidence=0.95),
        ResolvedField(field_name="provider_document", value="12.345.678/0001-95", confidence=0.93),
        ResolvedField(field_name="recipient_name", value="CLIENTE LTDA", confidence=0.91),
        ResolvedField(field_name="service_description", value="Consultoria", confidence=0.94),
        ResolvedField(field_name="gross_amount", value="1.000,00", confidence=0.93),
        ResolvedField(field_name="net_amount", value="950,00", confidence=0.92),
    ]


def test_decision_engine_auto_approves_clean_high_confidence_document() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-auto", metadata={"selected_source": "tesseract", "completeness_score": 0.95})

    result = engine.decide(document, _base_fields(), [])

    assert result.decision_status == "auto_approved"
    assert result.selected_source == "tesseract"
    assert result.score is not None and result.score >= 0.70
    assert result.metadata["triggered_rule"] == "all_thresholds_satisfied"


def test_decision_engine_approves_with_warning_when_only_non_blocking_issue_exists() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-warning", metadata={"completeness_score": 0.95})
    issues = [ValidationIssue(code="invalid_service_code", message="warning", severity="warning", field_name="service_code")]

    result = engine.decide(document, _base_fields(), issues)

    assert result.decision_status == "approved_with_warning"


def test_decision_engine_requires_manual_review_for_unresolved_conflict() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-manual", metadata={"completeness_score": 0.95})
    fields = _base_fields()
    fields[4] = ResolvedField(field_name="provider_document", value=None, status="conflict", confidence=0.89)

    result = engine.decide(document, fields, [])

    assert result.decision_status == "manual_review_required"
    assert "provider_document" in result.metadata["unresolved_conflicts"]


def test_decision_engine_rejects_when_blocking_error_exists() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-rejected", metadata={"completeness_score": 0.95})
    issues = [ValidationIssue(code="required_field_missing", message="missing", severity="error", field_name="gross_amount")]

    result = engine.decide(document, _base_fields(), issues)

    assert result.decision_status == "rejected"
    assert result.metadata["triggered_rule"] == "blocking_validation_issues"


def test_decision_engine_uses_dedicated_completeness_threshold_and_keeps_traceability() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-completeness", metadata={"completeness_score": 0.50})

    result = engine.decide(document, _base_fields(), [])

    assert result.decision_status == "manual_review_required"
    assert result.metadata["triggered_rule"] == "completeness_below_minimum"
    assert result.metadata["thresholds_used"]["minimum_completeness_score"] == 0.80


def test_decision_engine_requires_manual_review_when_warning_volume_exceeds_threshold() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-warning-volume", metadata={"completeness_score": 0.95})
    issues = [
        ValidationIssue(code=f"warning-{index}", message="warning", severity="warning", field_name="service_code")
        for index in range(4)
    ]

    result = engine.decide(document, _base_fields(), issues)

    assert result.decision_status == "manual_review_required"
    assert result.metadata["triggered_rule"] == "warning_volume_above_tolerance"


def test_decision_engine_does_not_treat_missing_confidence_as_low_confidence() -> None:
    engine = ConfigDrivenDecisionEngine()
    document = Document(document_id="doc-manual-corrected", metadata={"completeness_score": 0.95})
    fields = [
        ResolvedField(field_name="nfse_number", value="123", confidence=None),
        ResolvedField(field_name="verification_code", value="AB12CD34", confidence=None),
        ResolvedField(field_name="issue_date", value="15/04/2026", confidence=None),
        ResolvedField(field_name="provider_name", value="ACME LTDA", confidence=None),
        ResolvedField(field_name="provider_document", value="12.345.678/0001-95", confidence=None),
        ResolvedField(field_name="recipient_name", value="CLIENTE LTDA", confidence=None),
        ResolvedField(field_name="service_description", value="Consultoria", confidence=None),
        ResolvedField(field_name="gross_amount", value="1.000,00", confidence=None),
        ResolvedField(field_name="net_amount", value="950,00", confidence=None),
    ]

    result = engine.decide(document, fields, [])

    assert result.decision_status == "auto_approved"
    assert result.metadata["average_field_confidence"] is None
