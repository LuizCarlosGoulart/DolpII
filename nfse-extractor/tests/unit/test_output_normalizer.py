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
