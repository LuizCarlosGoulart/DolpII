from src.core import Document, FieldCandidate
from src.resolver import ConfigDrivenFieldResolver


def test_field_resolver_distinguishes_provider_and_recipient_by_context() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-1")
    candidates = [
        FieldCandidate(
            candidate_id="cand-provider",
            field_name="documento",
            value="12.345.678/0001-90",
            confidence=0.85,
            metadata={
                "label_text": "CNPJ Prestador",
                "section_name": "Prestador de Servicos",
                "same_block_as_label": True,
            },
        ),
        FieldCandidate(
            candidate_id="cand-recipient",
            field_name="documento",
            value="98.765.432/0001-10",
            confidence=0.83,
            metadata={
                "label_text": "CNPJ Tomador",
                "section_name": "Tomador de Servicos",
                "same_block_as_label": True,
            },
        ),
    ]

    resolved = {field.field_name: field for field in resolver.resolve(document, candidates)}

    assert resolved["provider_document"].value == "12.345.678/0001-90"
    assert resolved["recipient_document"].value == "98.765.432/0001-10"


def test_field_resolver_preserves_alternatives_for_multiple_cnpj_like_candidates() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-2")
    candidates = [
        FieldCandidate(
            candidate_id="cand-1",
            field_name="cnpj_prestador",
            value="12.345.678/0001-90",
            confidence=0.78,
            metadata={"section_name": "Prestador", "same_block_as_label": True},
        ),
        FieldCandidate(
            candidate_id="cand-2",
            field_name="documento_prestador",
            value="22.222.222/0001-22",
            confidence=0.74,
            metadata={"section_name": "Prestador", "same_block_as_label": True},
        ),
    ]

    resolved = {field.field_name: field for field in resolver.resolve(document, candidates)}
    provider_document = resolved["provider_document"]

    assert provider_document.status == "conflict"
    assert provider_document.value is None
    assert provider_document.source_candidate_ids == ["cand-1", "cand-2"]
    assert len(provider_document.metadata["alternatives"]) == 2


def test_field_resolver_uses_pattern_and_context_when_labels_are_missing() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-3")
    candidates = [
        FieldCandidate(
            candidate_id="cand-date",
            field_name="unknown",
            value="15/04/2026",
            confidence=0.70,
            metadata={
                "context_text": "Emitida em 15/04/2026",
                "section_name": "Dados da Nota",
                "label_distance": 12,
            },
        )
    ]

    resolved = resolver.resolve(document, candidates)

    issue_date = next(field for field in resolved if field.field_name == "issue_date")
    assert issue_date.status == "resolved"
    assert issue_date.value == "15/04/2026"


def test_field_resolver_marks_conflicting_value_candidates_without_inventing_value() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-4")
    candidates = [
        FieldCandidate(
            candidate_id="cand-gross-1",
            field_name="valor_servicos",
            value="1.250,00",
            confidence=0.76,
            metadata={"section_name": "Valores", "same_block_as_label": True},
        ),
        FieldCandidate(
            candidate_id="cand-gross-2",
            field_name="valor_bruto",
            value="1.350,00",
            confidence=0.75,
            metadata={"section_name": "Valores", "same_block_as_label": True},
        ),
    ]

    resolved = {field.field_name: field for field in resolver.resolve(document, candidates)}
    gross_amount = resolved["gross_amount"]

    assert gross_amount.status == "conflict"
    assert gross_amount.value is None
    assert gross_amount.metadata["ambiguity_detected"] is True
    assert {alternative["value"] for alternative in gross_amount.metadata["alternatives"]} == {
        "1.250,00",
        "1.350,00",
    }


def test_field_resolver_treats_same_block_as_probabilistic_evidence() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-5")
    candidates = [
        FieldCandidate(
            candidate_id="cand-weak-block",
            field_name="unknown",
            value="12.345.678/0001-90",
            confidence=0.0,
            metadata={
                "section_name": "Prestador",
                "same_block_as_label": True,
            },
        )
    ]

    resolved = {field.field_name: field for field in resolver.resolve(document, candidates)}
    provider_document = resolved["provider_document"]

    assert provider_document.status == "conflict"
    assert provider_document.value is None
    assert provider_document.confidence < 0.60


def test_field_resolver_uses_configured_ambiguity_delta() -> None:
    resolver = ConfigDrivenFieldResolver()
    document = Document(document_id="doc-6")
    candidates = [
        FieldCandidate(
            candidate_id="cand-a",
            field_name="cnpj_prestador",
            value="12.345.678/0001-90",
            confidence=0.80,
            metadata={"section_name": "Prestador", "same_block_as_label": True},
        ),
        FieldCandidate(
            candidate_id="cand-b",
            field_name="documento_prestador",
            value="22.222.222/0001-22",
            confidence=0.69,
            metadata={"section_name": "Prestador", "same_block_as_label": True},
        ),
    ]

    resolved = {field.field_name: field for field in resolver.resolve(document, candidates)}

    assert resolved["provider_document"].status == "resolved"
    assert resolved["provider_document"].value == "12.345.678/0001-90"
