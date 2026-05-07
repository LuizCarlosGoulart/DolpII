from src.core import Document, ExtractedElement
from src.normalization import ConfigDrivenOutputNormalizer


def _line(document_id: str, block_num: int, line_num: int, text: str, *, y: float) -> list[ExtractedElement]:
    elements: list[ExtractedElement] = []
    x = 10.0
    for word_index, word in enumerate(text.split(), start=1):
        width = max(float(len(word) * 8), 8.0)
        elements.append(
            ExtractedElement(
                element_id=f"{document_id}:{block_num}:{line_num}:{word_index}",
                element_type="text",
                text=word,
                page_number=1,
                bounding_box=(x, y, width, 12.0),
                confidence=0.90,
                metadata={
                    "source_engine": "tesseract",
                    "block_num": block_num,
                    "line_num": line_num,
                    "word_num": word_index,
                },
            )
        )
        x += width + 8.0
    return elements


def _candidate_values(candidates):
    return {candidate.field_name: candidate.value for candidate in candidates}


def _values_for(candidates, field_name: str) -> list[str]:
    return [candidate.value for candidate in candidates if candidate.field_name == field_name]


def test_output_normalizer_extracts_header_fields_from_labels_and_next_line() -> None:
    document = Document(document_id="doc-header")
    elements = [
        *_line(document.document_id, 1, 1, "Numero da Nota Fiscal", y=10.0),
        *_line(document.document_id, 1, 2, "1933", y=26.0),
        *_line(document.document_id, 1, 3, "Serie E", y=42.0),
        *_line(document.document_id, 1, 4, "Data Emissao 25/08/2023", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["nfse_number"] == "1933"
    assert values["nfse_series"] == "E"
    assert values["issue_date"] == "25/08/2023"


def test_output_normalizer_separates_nfse_number_from_rps_and_generic_nfse_text() -> None:
    document = Document(document_id="doc-header-realistic")
    elements = [
        *_line(document.document_id, 1, 1, "Numero do RPS Numero da nota", y=10.0),
        *_line(document.document_id, 1, 2, "PREFEITURA MUNICIPAL 1.794.323 1792029", y=26.0),
        *_line(document.document_id, 1, 3, "Nota Fiscal Eletronica de Prestacao de Servicos NFS-e Data do fato gerador", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    nfse_values = _values_for(candidates, "nfse_number")

    assert "1792029" in nfse_values
    assert "1.794.323" not in nfse_values
    assert all("NFS" not in value.upper() for value in nfse_values)


def test_output_normalizer_accepts_short_nfse_number_near_header_label() -> None:
    document = Document(document_id="doc-short-number")
    elements = [
        *_line(document.document_id, 1, 1, "PREFEITURA MUNICIPAL Número da Nota Fiscal", y=10.0),
        *_line(document.document_id, 1, 2, "16", y=26.0),
        *_line(document.document_id, 1, 3, "Data Emissao 27/05/2014", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "nfse_number") == ["16"]


def test_output_normalizer_finds_issue_date_near_label_without_using_label_tail() -> None:
    document = Document(document_id="doc-date-nearby")
    elements = [
        *_line(document.document_id, 1, 1, "Data da emissao da nota", y=10.0),
        *_line(document.document_id, 1, 2, "SECRETARIA MUNICIPAL DA RECEITA", y=26.0),
        *_line(document.document_id, 1, 3, "14/11/2021 20:10:00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "issue_date") == ["14/11/2021"]


def test_output_normalizer_keeps_generation_date_out_of_issue_date_candidates() -> None:
    document = Document(document_id="doc-dates")
    elements = [
        *_line(document.document_id, 1, 1, "Data Emissao 27/05/2014", y=10.0),
        *_line(document.document_id, 1, 2, "Mes de Competencia 05/2014 Local do Recolhimento BALNEARIO CAMBORIU", y=26.0),
        *_line(document.document_id, 1, 3, "Data Geracao 28/05/2014 17:11:00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "issue_date") == ["27/05/2014"]
    assert _values_for(candidates, "competence_date") == ["05/2014"]


def test_output_normalizer_rejects_empty_municipal_registration_before_next_label() -> None:
    document = Document(document_id="doc-empty-registration")
    elements = [
        *_line(document.document_id, 1, 1, "DADOS DO TOMADOR", y=10.0),
        *_line(document.document_id, 1, 2, "CNPJ/CPF: 290.772.229-87 Insc. Municipal: Insc. Estadual:", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "recipient_document") == ["290.772.229-87"]
    assert _values_for(candidates, "recipient_municipal_registration") == []


def test_output_normalizer_keeps_provider_and_recipient_contexts_separate() -> None:
    document = Document(document_id="doc-parties")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR", y=10.0),
        *_line(document.document_id, 1, 2, "Nome/Razao Social: JAMES ROBERITAN SILVEIRA", y=26.0),
        *_line(document.document_id, 1, 3, "CNPJ/CPF: 13.101.735/0001-53", y=42.0),
        *_line(document.document_id, 1, 4, "Endereco: HERMANN TRIBESS N 984", y=58.0),
        *_line(document.document_id, 1, 5, "UF: SC CEP: 89057-300", y=74.0),
        *_line(document.document_id, 1, 6, "Telefone: 4730374700", y=90.0),
        *_line(document.document_id, 2, 1, "DADOS DO TOMADOR", y=120.0),
        *_line(document.document_id, 2, 2, "Nome/Razao Social: HAGI PIZZAS E SUPERMERCADO LTDA", y=136.0),
        *_line(document.document_id, 2, 3, "CNPJ/CPF: 02.876.218/0006-44", y=152.0),
        *_line(document.document_id, 2, 4, "UF: PR", y=168.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["provider_name"] == "JAMES ROBERITAN SILVEIRA"
    assert values["provider_document"] == "13.101.735/0001-53"
    assert values["provider_address"] == "HERMANN TRIBESS N 984"
    assert values["provider_uf"] == "SC"
    assert values["provider_phone"] == "4730374700"
    assert values["recipient_name"] == "HAGI PIZZAS E SUPERMERCADO LTDA"
    assert values["recipient_document"] == "02.876.218/0006-44"
    assert values["recipient_uf"] == "PR"


def test_output_normalizer_avoids_section_headings_as_party_names() -> None:
    document = Document(document_id="doc-section-heading")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Nome/Razao social: ORSEGUPS MONITORAMENTO ELETRONICO LTDA.", y=26.0),
        *_line(document.document_id, 2, 1, "TOMADOR DE SERVICOS", y=58.0),
        *_line(document.document_id, 2, 2, "Nome/Razao social: SC COMEX ASSESSORIA EM COMERCIO EXTERIOR EIRELI", y=74.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_name") == ["ORSEGUPS MONITORAMENTO ELETRONICO LTDA."]
    assert _values_for(candidates, "recipient_name") == ["SC COMEX ASSESSORIA EM COMERCIO EXTERIOR EIRELI"]


def test_output_normalizer_does_not_treat_phone_as_document_and_handles_compact_uf() -> None:
    document = Document(document_id="doc-compact")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR", y=10.0),
        *_line(document.document_id, 1, 2, "CPF/CNPJ 08.491.597/0001-26 Inscricao municipal: 9014419 Telefone: 4020441 1/08006486600", y=26.0),
        *_line(document.document_id, 2, 1, "TOMADOR", y=58.0),
        *_line(document.document_id, 2, 2, "Municipio BALNEARIO CAMBOR UF:SC", y=74.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["08.491.597/0001-26"]
    assert _values_for(candidates, "provider_phone") == ["4020441 08006486600"]
    assert _values_for(candidates, "recipient_uf") == ["SC"]


def test_output_normalizer_extracts_service_and_tax_fields_from_aliases() -> None:
    document = Document(document_id="doc-values")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DOS SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Codigo do Servico 01.05.00", y=26.0),
        *_line(document.document_id, 1, 3, "Natureza da Operacao Tributacao no municipio", y=42.0),
        *_line(document.document_id, 1, 4, "Municipio de Incidencia BLUMENAU", y=58.0),
        *_line(document.document_id, 2, 1, "VALORES", y=90.0),
        *_line(document.document_id, 2, 2, "Valor Total do Servico 1.000,00", y=106.0),
        *_line(document.document_id, 2, 3, "Base de Calculo 1.000,00", y=122.0),
        *_line(document.document_id, 2, 4, "Aliquota ISS 2,00%", y=138.0),
        *_line(document.document_id, 2, 5, "Valor ISS 20,00", y=154.0),
        *_line(document.document_id, 2, 6, "ISS Retido 20,00", y=170.0),
        *_line(document.document_id, 2, 7, "PRRF 15,00", y=186.0),
        *_line(document.document_id, 2, 8, "Valor liquido da NFSE 965,00", y=202.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["service_code"] == "01.05.00"
    assert values["operation_nature"] == "Tributacao no municipio"
    assert values["service_city"] == "BLUMENAU"
    assert values["gross_amount"] == "1.000,00"
    assert values["taxable_amount"] == "1.000,00"
    assert values["iss_rate"] == "2,00%"
    assert values["iss_amount"] == "20,00"
    assert values["iss_withheld_amount"] == "20,00"
    assert values["ir_withheld_amount"] == "15,00"
    assert values["net_amount"] == "965,00"


def test_output_normalizer_rejects_service_description_header_without_content() -> None:
    document = Document(document_id="doc-service-header")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DO SERVICO", y=10.0),
        *_line(document.document_id, 1, 2, "Valor Total dos Servicos 230,00", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_description") == []


def test_output_normalizer_extracts_service_description_from_lines_after_header() -> None:
    document = Document(document_id="doc-service-lines")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DO SERVICO", y=10.0),
        *_line(document.document_id, 1, 2, "Consulta medica especializada em angiologia vascular", y=26.0),
        *_line(document.document_id, 1, 3, "VALORES", y=42.0),
        *_line(document.document_id, 1, 4, "Valor Total dos Servicos 230,00", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_description") == [
        "Consulta medica especializada em angiologia vascular"
    ]


def test_output_normalizer_corrects_common_ocr_email_at_symbol_with_low_confidence() -> None:
    document = Document(document_id="doc-ocr-email")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR", y=10.0),
        *_line(document.document_id, 1, 2, "E-mail: fiscalObarbicontabil.com.br Telefone: 4734050730", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    provider_email = next(candidate for candidate in candidates if candidate.field_name == "provider_email")

    assert provider_email.value == "fiscal@barbicontabil.com.br"
    assert provider_email.confidence is not None and provider_email.confidence < 0.75
    assert provider_email.metadata["value_source"] == "ocr_corrected_email"
    assert provider_email.metadata["ocr_correction_applied"] is True


def test_output_normalizer_rejects_implausible_integer_iss_rate_from_noisy_line() -> None:
    document = Document(document_id="doc-noisy-rate")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Base de Calculo: R$230,00 Aliquota: AMO [tel do ISS:", y=26.0),
        *_line(document.document_id, 1, 3, "R$230,00 23", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "taxable_amount") == ["R$230,00"]
    assert _values_for(candidates, "iss_rate") == []


def test_output_normalizer_does_not_assign_incomplete_financial_table_values() -> None:
    document = Document(document_id="doc-incomplete-table")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            "Valor Total das Deducoes Desconto Incondicionado Base de Calculo Aliquota",
            y=26.0,
        ),
        *_line(document.document_id, 1, 3, "R$ 230,00", y=42.0),
        *_line(document.document_id, 1, 4, "VALOR LIQUIDO DA NOTA R$ 230,00", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "deductions_amount") == []
    assert _values_for(candidates, "unconditional_discount") == []
    assert _values_for(candidates, "taxable_amount") == []
    assert _values_for(candidates, "net_amount") == ["R$ 230,00"]


def test_output_normalizer_rejects_generic_nonzero_deductions_from_table_mapping() -> None:
    document = Document(document_id="doc-generic-deductions")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            "Valor Total das Deducoes Desconto Incondicionado Base de Calculo Valor ISS",
            y=26.0,
        ),
        *_line(document.document_id, 1, 3, "R$ 230,00 R$ 0,00 R$ 230,00 R$ 0,00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "deductions_amount") == []
    assert _values_for(candidates, "unconditional_discount") == ["R$ 0,00"]
    assert _values_for(candidates, "taxable_amount") == ["R$ 230,00"]


def test_output_normalizer_requires_percent_for_table_mapped_iss_rate() -> None:
    document = Document(document_id="doc-table-rate")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Base de Calculo Aliquota Valor ISS", y=26.0),
        *_line(document.document_id, 1, 3, "R$ 230,00 0,00 R$ 0,00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "taxable_amount") == ["R$ 230,00"]
    assert _values_for(candidates, "iss_rate") == []


def test_output_normalizer_maps_value_table_columns_to_fields() -> None:
    document = Document(document_id="doc-table")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Desc. condicionado(R$) Desc. incondicionado(R$) Deducoes(R$) Base de calculo(R$) Valor ISS(R$)", y=26.0),
        *_line(document.document_id, 1, 3, "0,00 0,00 0,00 160,87 4,83", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["conditional_discount"] == "0,00"
    assert values["unconditional_discount"] == "0,00"
    assert values["deductions_amount"] == "0,00"
    assert values["taxable_amount"] == "160,87"
    assert values["iss_amount"] == "4,83"


def test_output_normalizer_preserves_candidate_traceability_metadata() -> None:
    document = Document(document_id="doc-trace")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR", y=10.0),
        *_line(document.document_id, 1, 2, "CNPJ/CPF: 13.101.735/0001-53", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    provider_document = next(candidate for candidate in candidates if candidate.field_name == "provider_document")

    assert provider_document.source_element_ids
    assert provider_document.source_name == "config-driven-output-normalizer"
    assert provider_document.metadata["section_name"] == "provider"
    assert provider_document.metadata["label_text"] in {"cnpj cpf", "document pattern"}
