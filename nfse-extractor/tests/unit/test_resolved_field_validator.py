from datetime import UTC, datetime, timedelta

from src.core import Document, ResolvedField
from src.validation import ConfigDrivenValidator


def test_validator_reports_required_field_missing() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-required")

    issues = validator.validate(document, [])

    assert any(issue.code == "required_field_missing" and issue.field_name == "nfse_number" for issue in issues)
    assert any(issue.code == "required_field_missing" and issue.field_name == "gross_amount" for issue in issues)


def test_validator_reports_syntactic_issues_for_document_email_phone_and_uf() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-syntax")
    fields = [
        ResolvedField(field_name="provider_document", value="11.111.111/1111-11"),
        ResolvedField(field_name="contact_email", value="invalid-email"),
        ResolvedField(field_name="contact_phone", value="1234"),
        ResolvedField(field_name="service_uf", value="XX"),
    ]

    issues = validator.validate(document, fields)
    issue_codes = {issue.code for issue in issues}

    assert "invalid_document_id" in issue_codes
    assert "invalid_email" in issue_codes
    assert "invalid_phone" in issue_codes
    assert "invalid_uf" in issue_codes


def test_validator_reports_semantic_and_relational_issues() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-semantic")
    future_date = (datetime.now(UTC).date() + timedelta(days=5)).strftime("%d/%m/%Y")
    fields = [
        ResolvedField(field_name="nfse_number", value="123"),
        ResolvedField(field_name="verification_code", value="AB12CD34"),
        ResolvedField(field_name="issue_date", value=future_date),
        ResolvedField(field_name="provider_name", value="ACME LTDA"),
        ResolvedField(field_name="provider_document", value="12.345.678/0001-90"),
        ResolvedField(field_name="recipient_name", value="CLIENTE LTDA"),
        ResolvedField(field_name="service_description", value="Consultoria"),
        ResolvedField(field_name="gross_amount", value="1.000,00"),
        ResolvedField(field_name="unconditional_discount", value="100,00"),
        ResolvedField(field_name="iss_withheld_amount", value="50,00"),
        ResolvedField(field_name="net_amount", value="900,00"),
        ResolvedField(field_name="service_code", value="ABC"),
    ]

    issues = validator.validate(document, fields)
    issue_codes = {issue.code for issue in issues}

    assert "implausible_future_date" in issue_codes
    assert "invalid_service_code" in issue_codes
    assert "net_amount_inconsistent" in issue_codes


def test_validator_checks_iss_consistency_without_escalating_to_error() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-iss")
    fields = [
        ResolvedField(field_name="taxable_amount", value="1.000,00"),
        ResolvedField(field_name="iss_rate", value="5%"),
        ResolvedField(field_name="iss_amount", value="70,00"),
    ]

    issues = validator.validate(document, fields)

    iss_issue = next(issue for issue in issues if issue.code == "iss_amount_inconsistent")
    assert iss_issue.severity == "warning"


def test_validator_warns_when_provider_and_recipient_documents_match() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-duplicate-docs")
    fields = [
        ResolvedField(field_name="provider_document", value="12.345.678/0001-95"),
        ResolvedField(field_name="recipient_document", value="12.345.678/0001-95"),
    ]

    issues = validator.validate(document, fields)

    duplicate_issue = next(issue for issue in issues if issue.code == "provider_recipient_document_equal")
    assert duplicate_issue.severity == "warning"
    assert duplicate_issue.field_name == "recipient_document"


def test_validator_avoids_duplicate_required_field_state_issues() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-required-conflict")
    fields = [ResolvedField(field_name="gross_amount", value=None, status="missing")]

    issues = validator.validate(document, fields)
    gross_amount_issues = [issue for issue in issues if issue.field_name == "gross_amount"]

    assert [issue.code for issue in gross_amount_issues] == ["required_field_unresolved"]


def test_validator_skips_relational_check_when_component_value_is_invalid() -> None:
    validator = ConfigDrivenValidator()
    document = Document(document_id="doc-invalid-component")
    fields = [
        ResolvedField(field_name="nfse_number", value="123"),
        ResolvedField(field_name="verification_code", value="AB12CD34"),
        ResolvedField(field_name="issue_date", value="15/04/2026"),
        ResolvedField(field_name="provider_name", value="ACME LTDA"),
        ResolvedField(field_name="provider_document", value="12.345.678/0001-95"),
        ResolvedField(field_name="recipient_name", value="CLIENTE LTDA"),
        ResolvedField(field_name="service_description", value="Consultoria"),
        ResolvedField(field_name="gross_amount", value="1.000,00"),
        ResolvedField(field_name="unconditional_discount", value="abc"),
        ResolvedField(field_name="net_amount", value="900,00"),
    ]

    issues = validator.validate(document, fields)
    issue_codes = {issue.code for issue in issues}

    assert "invalid_monetary_value" in issue_codes
    assert "net_amount_inconsistent" not in issue_codes
