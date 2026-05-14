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


def test_output_normalizer_splits_series_number_and_issue_date_from_same_header_line() -> None:
    document = Document(document_id="doc-header-composite")
    elements = [
        *_line(document.document_id, 1, 1, "Serie: E Nota No.: 5852528 Emissao: 05/09/2021", y=10.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["nfse_series"] == "E"
    assert values["nfse_number"] == "5852528"
    assert values["issue_date"] == "05/09/2021"


def test_output_normalizer_sanitizes_long_rps_series_context() -> None:
    document = Document(document_id="doc-series-rps")
    elements = [
        *_line(document.document_id, 1, 1, "RPS No 16822 Serie 100 emitida em 10/08/2021 JWIP-PGQB", y=10.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "nfse_series") == ["100"]


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


def test_output_normalizer_extracts_nfem_number_from_next_line() -> None:
    document = Document(document_id="doc-nfem-number")
    elements = [
        *_line(document.document_id, 1, 1, "Numero da NF-em", y=10.0),
        *_line(document.document_id, 1, 2, "14792", y=26.0),
        *_line(document.document_id, 1, 3, "Data e Hora de Emissao", y=42.0),
        *_line(document.document_id, 1, 4, "02/07/2021 09:43", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "nfse_number") == ["14792"]


def test_output_normalizer_finds_issue_date_near_label_without_using_label_tail() -> None:
    document = Document(document_id="doc-date-nearby")
    elements = [
        *_line(document.document_id, 1, 1, "Data da emissao da nota", y=10.0),
        *_line(document.document_id, 1, 2, "SECRETARIA MUNICIPAL DA RECEITA", y=26.0),
        *_line(document.document_id, 1, 3, "14/11/2021 20:10:00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "issue_date") == ["14/11/2021"]


def test_output_normalizer_extracts_hyphenated_verification_code_from_generic_alias() -> None:
    document = Document(document_id="doc-verification-code")
    elements = [
        *_line(document.document_id, 1, 1, "Certificacao: 7BBC2-2060F", y=10.0),
        *_line(document.document_id, 1, 2, "Numero da Nota Fiscal", y=26.0),
        *_line(document.document_id, 1, 3, "16", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "verification_code") == ["7BBC2-2060F"]


def test_output_normalizer_joins_split_verification_code_lines() -> None:
    document = Document(document_id="doc-verification-split")
    elements = [
        *_line(document.document_id, 1, 1, "Codigo de Verificacao", y=10.0),
        *_line(document.document_id, 1, 2, "60614BB7-2144-0FA7-", y=26.0),
        *_line(document.document_id, 1, 3, "5D1E-BC858573B392", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "verification_code") == ["60614BB7-2144-0FA7-5D1E-BC858573B392"]


def test_output_normalizer_accepts_short_alpha_hyphen_verification_code() -> None:
    document = Document(document_id="doc-verification-alpha-hyphen")
    elements = [
        *_line(document.document_id, 1, 1, "Codigo de Verificacao JWIP-PGQB", y=10.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "verification_code") == ["JWIP-PGQB"]


def test_output_normalizer_rejects_boilerplate_words_as_verification_code() -> None:
    document = Document(document_id="doc-verification-boilerplate")
    elements = [
        *_line(document.document_id, 1, 1, "Certificacao", y=10.0),
        *_line(document.document_id, 1, 2, "ASSINATURA DIGITAL", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "verification_code") == []


def test_output_normalizer_rejects_administrative_words_as_verification_code() -> None:
    document = Document(document_id="doc-verification-admin-words")
    elements = [
        *_line(document.document_id, 1, 1, "Codigo de Verificacao", y=10.0),
        *_line(document.document_id, 1, 2, "PRESTADOR DE SERVICOS", y=26.0),
        *_line(document.document_id, 1, 3, "Autenticidade", y=42.0),
        *_line(document.document_id, 1, 4, "MODELO FISCAL", y=58.0),
        *_line(document.document_id, 1, 5, "ELETRONICA-NFSE", y=74.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "verification_code") == []


def test_output_normalizer_prefers_issue_date_over_authorization_date() -> None:
    document = Document(document_id="doc-date-ranking")
    elements = [
        *_line(document.document_id, 1, 1, "Data/Hora Emissao", y=10.0),
        *_line(document.document_id, 1, 2, "05/08/2021", y=26.0),
        *_line(document.document_id, 1, 3, "Autorizacao para emissao de Nota Fiscal de Servico Eletronica 334/2020 de 27/07/2020", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "issue_date") == ["05/08/2021"]


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


def test_output_normalizer_trims_state_registration_from_municipal_registration() -> None:
    document = Document(document_id="doc-registration-trim")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR", y=10.0),
        *_line(document.document_id, 1, 2, "CPF/CNPJ: 36.426.996/0001-49 Inscricao Municipal: 192.699 Estadual:", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_municipal_registration") == ["192.699"]


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


def test_output_normalizer_scopes_labels_before_tomador_marker_as_provider_leakage() -> None:
    document = Document(document_id="doc-provider-label-before-tomador")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            "Nome/Razao Social: BRB SERVICOS DE TECNOLOGIA LTDA CNPJ/CPF: 10.468.075/0001-55 DADOS DO TOMADOR",
            y=26.0,
        ),
        *_line(document.document_id, 1, 3, "CNPJ/CPF: 02.876.218/0006-44", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)
    provider_document = next(candidate for candidate in candidates if candidate.field_name == "provider_document")

    assert values["provider_name"] == "BRB SERVICOS DE TECNOLOGIA LTDA"
    assert values["provider_document"] == "10.468.075/0001-55"
    assert values["recipient_document"] == "02.876.218/0006-44"
    assert provider_document.metadata["section_name"] == "provider"
    assert provider_document.metadata["raw_section_name"] == "recipient"
    assert provider_document.metadata["section_override_applied"] is True


def test_output_normalizer_keeps_document_pattern_before_tomador_marker_with_provider_fragment() -> None:
    document = Document(document_id="doc-provider-document-before-tomador")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            (
                "CNPJ/CPF: Nome/Razao Nome Endereco: Bairro: Municipio: E-mail: Pais: Fantasia: "
                "VELHA BRASIL bloemerrodolfo BLUMENAU DR. Social: 10.468.075/0001-55 "
                "ARTUR BRB SERVICOS BRB BALSINI gmail.com SERVICOS DE TECNOLOGIA DADOS DO TOMADOR"
            ),
            y=26.0,
        ),
        *_line(document.document_id, 1, 3, "CNPJ/CPF: 02.876.218/0006-44", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["10.468.075/0001-55"]
    assert _values_for(candidates, "recipient_document") == ["02.876.218/0006-44"]


def test_output_normalizer_extracts_party_names_from_razao_social_label() -> None:
    document = Document(document_id="doc-razao-social")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Razao Social: ENGETERME TECNOLOGIA EM CLIMATIZACAO LTDA ME", y=26.0),
        *_line(document.document_id, 1, 3, "TOMADOR DO SERVICO", y=42.0),
        *_line(document.document_id, 1, 4, "Razao Social/Nome: HAGI PIZZAS E SUPERMERCADO LTDA", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["provider_name"] == "ENGETERME TECNOLOGIA EM CLIMATIZACAO LTDA ME"
    assert values["recipient_name"] == "HAGI PIZZAS E SUPERMERCADO LTDA"


def test_output_normalizer_removes_document_suffix_from_party_name() -> None:
    document = Document(document_id="doc-party-name-document-suffix")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Razao Social: DANIEL CESAR BALDO 05021232908", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_name") == ["DANIEL CESAR BALDO"]


def test_output_normalizer_infers_provider_block_before_recipient() -> None:
    document = Document(document_id="doc-implicit-provider")
    elements = [
        *_line(document.document_id, 1, 1, "NOTA FISCAL ELETRONICA DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Nome/Razao Social: EMPRESA PRESTADORA LTDA", y=26.0),
        *_line(document.document_id, 1, 3, "CNPJ/CPF: 11.111.111/0001-11", y=42.0),
        *_line(document.document_id, 1, 4, "Endereco: Rua Um 100", y=58.0),
        *_line(document.document_id, 1, 5, "TOMADOR DO SERVICO", y=74.0),
        *_line(document.document_id, 1, 6, "Nome/Razao Social: CLIENTE LTDA", y=90.0),
        *_line(document.document_id, 1, 7, "CNPJ/CPF: 22.222.222/0001-22", y=106.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["provider_name"] == "EMPRESA PRESTADORA LTDA"
    assert values["provider_document"] == "11.111.111/0001-11"
    assert values["provider_address"] == "Rua Um 100"
    assert values["recipient_name"] == "CLIENTE LTDA"
    assert values["recipient_document"] == "22.222.222/0001-22"


def test_output_normalizer_ignores_billing_identifiers_before_fiscal_header() -> None:
    document = Document(document_id="doc-billing-noise")
    elements = [
        *_line(document.document_id, 1, 1, "Local de Pagamento Vencimento Valor do Documento", y=10.0),
        *_line(document.document_id, 1, 2, "CNPJ/CPF: 99.999.999/0001-99 Nosso Numero 123456", y=26.0),
        *_line(document.document_id, 1, 3, "Autenticacao Mecanica", y=42.0),
        *_line(document.document_id, 1, 4, "NOTA FISCAL ELETRONICA DE SERVICOS", y=58.0),
        *_line(document.document_id, 1, 5, "Nome/Razao Social: PRESTADOR REAL LTDA", y=74.0),
        *_line(document.document_id, 1, 6, "CNPJ/CPF: 11.111.111/0001-11", y=90.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["11.111.111/0001-11"]


def test_output_normalizer_requires_document_label_for_document_pattern_fallback() -> None:
    document = Document(document_id="doc-unlabeled-document-like-number")
    elements = [
        *_line(document.document_id, 1, 1, "NOTA FISCAL ELETRONICA DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "75590.00323 82355.850098 75092.790338 8 92560000018678", y=26.0),
        *_line(document.document_id, 1, 3, "CNPJ/CPF: 11.111.111/0001-11", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["11.111.111/0001-11"]


def test_output_normalizer_accepts_formatted_document_inside_party_block_when_label_is_lost() -> None:
    document = Document(document_id="doc-lost-document-label")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "e 06.040.270/0001-02 Inscricao Municipal: 282575", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["06.040.270/0001-02"]


def test_output_normalizer_normalizes_ocr_document_separator_and_avoids_unlabeled_party_leakage() -> None:
    document = Document(document_id="doc-ocr-document")
    elements = [
        *_line(document.document_id, 1, 1, "PRESTADOR DE SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "CNPJ/CPF 00,536.772/0021-96 Inscricao Municipal 5471747", y=26.0),
        *_line(document.document_id, 1, 3, "HAGI PIZZAS E SUPERMERCADO LTDA 02.876.218/0006-44", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "provider_document") == ["00.536.772/0021-96"]


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


def test_output_normalizer_extracts_service_fields_from_following_lines() -> None:
    document = Document(document_id="doc-service-following-lines")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DOS SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "SERVICO DE MANUTENCAO EM EQUIPAMENTOS DE INFORMATICA", y=26.0),
        *_line(document.document_id, 1, 3, "Codigo do Servico:", y=42.0),
        *_line(document.document_id, 1, 4, "10.02", y=58.0),
        *_line(document.document_id, 1, 5, "Natureza de Operacao:", y=74.0),
        *_line(document.document_id, 1, 6, "501 - ISS devido para Itajai Simples Nacional", y=90.0),
        *_line(document.document_id, 1, 7, "Local da prestacao do servico", y=106.0),
        *_line(document.document_id, 1, 8, "ITAJAI - SC", y=122.0),
        *_line(document.document_id, 1, 9, "VALORES", y=138.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["service_description"] == "SERVICO DE MANUTENCAO EM EQUIPAMENTOS DE INFORMATICA"
    assert values["service_code"] == "10.02"
    assert values["operation_nature"] == "501 - ISS devido para Itajai Simples Nacional"
    assert values["service_city"] == "ITAJAI/SC"


def test_output_normalizer_removes_inline_description_prefix() -> None:
    document = Document(document_id="doc-description-prefix")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DOS SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Descricao: LICENCIAMENTO DE USO DE SOFTWARE", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_description") == ["LICENCIAMENTO DE USO DE SOFTWARE"]


def test_output_normalizer_extracts_lc116_numeric_service_code_with_context() -> None:
    document = Document(document_id="doc-service-code-numeric")
    elements = [
        *_line(document.document_id, 1, 1, "DESCRICAO DOS SUBITENS DA LISTA DE SERVICO", y=10.0),
        *_line(document.document_id, 1, 2, "2601 Servicos de coleta remessa ou entrega de documentos", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_code") == ["2601"]


def test_output_normalizer_does_not_extract_unlabeled_year_as_service_code() -> None:
    document = Document(document_id="doc-service-code-year")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DOS SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Contrato mensal referente a agosto de 2021", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_code") == []


def test_output_normalizer_does_not_extract_law_number_or_money_prefix_as_service_code() -> None:
    document = Document(document_id="doc-service-code-false-positive")
    elements = [
        *_line(document.document_id, 1, 1, "DESCRICAO DOS SUBITENS DA LISTA DE SERVICO EM ACORDO COM A LEI COMPLEMENTAR 116/03", y=10.0),
        *_line(document.document_id, 1, 2, "VALORES", y=26.0),
        *_line(document.document_id, 1, 3, "Codigo do Servico", y=42.0),
        *_line(document.document_id, 1, 4, "R$ 479,90", y=58.0),
        *_line(document.document_id, 1, 5, "542 Documento interno sem contexto de servico", y=74.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_code") == []


def test_output_normalizer_extracts_service_city_from_collection_location_alias() -> None:
    document = Document(document_id="doc-service-city-collection")
    elements = [
        *_line(document.document_id, 1, 1, "ENQUADRAMENTO DO SERVICO", y=10.0),
        *_line(document.document_id, 1, 2, "Mes de Competencia: 08/2021 Local do Recolhimento: BLUMENAU/SC Data Geracao: 03/08/2021", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_city") == ["BLUMENAU/SC"]


def test_output_normalizer_normalizes_ocr_separator_in_service_city() -> None:
    document = Document(document_id="doc-service-city-separator")
    elements = [
        *_line(document.document_id, 1, 1, "Local do Recolhimento: ITAJAI! SC/BRASIL", y=10.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_city") == ["ITAJAI/SC/BRASIL"]


def test_output_normalizer_extracts_coded_service_city_and_rejects_legend_noise() -> None:
    document = Document(document_id="doc-service-city-coded")
    elements = [
        *_line(document.document_id, 1, 1, "DESCRICAO DOS SERVICOS PRESTADOS", y=10.0),
        *_line(document.document_id, 1, 2, "Servico Local Prestacao i Aliquota Situacao Trib. Valor Servico", y=26.0),
        *_line(document.document_id, 1, 3, "Descricao do Servico", y=42.0),
        *_line(document.document_id, 1, 4, "8039 Balneario Camboriu", y=58.0),
        *_line(document.document_id, 1, 5, "Outras Informacoes", y=74.0),
        *_line(document.document_id, 1, 6, "TI- Tributada Integralmente", y=90.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_city") == ["Balneario Camboriu"]


def test_output_normalizer_rejects_service_description_header_without_content() -> None:
    document = Document(document_id="doc-service-header")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DO SERVICO", y=10.0),
        *_line(document.document_id, 1, 2, "Valor Total dos Servicos 230,00", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_description") == []


def test_output_normalizer_rejects_value_table_as_service_description() -> None:
    document = Document(document_id="doc-description-value-table")
    elements = [
        *_line(document.document_id, 1, 1, "DISCRIMINACAO DOS SERVICOS E INFORMACOES RELEVANTES VALOR TOTAL DA NOTA R$ 221,43", y=10.0),
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


def test_output_normalizer_extracts_description_after_full_description_header_alias() -> None:
    document = Document(document_id="doc-description-header")
    elements = [
        *_line(document.document_id, 1, 1, "DESCRICAO DOS SERVICOS", y=10.0),
        *_line(document.document_id, 1, 2, "Manutencao preventiva mensal de equipamentos", y=26.0),
        *_line(document.document_id, 1, 3, "VALORES", y=42.0),
        *_line(document.document_id, 1, 4, "Valor Total dos Servicos R$ 500,00", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "service_description") == [
        "Manutencao preventiva mensal de equipamentos"
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


def test_output_normalizer_extracts_gross_amount_from_common_total_services_aliases() -> None:
    document = Document(document_id="doc-gross-alias")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Valor Total dos Servicos R$ 230,00", y=26.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)

    assert _values_for(candidates, "gross_amount") == ["R$ 230,00"]


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


def test_output_normalizer_uses_rate_column_as_financial_table_anchor() -> None:
    document = Document(document_id="doc-financial-rate-anchor")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            "Valor Serviço | | Base de Cálculo | Alíg. (4%): | Valor ISS | Ref.:",
            y=26.0,
        ),
        *_line(document.document_id, 1, 3, "7.249,46 | | | 7.249,46 | | 2,00 | 144,99 | 08/2021", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["gross_amount"] == "7.249,46"
    assert values["taxable_amount"] == "7.249,46"
    assert values["iss_rate"] == "2,00"
    assert values["iss_amount"] == "144,99"


def test_output_normalizer_maps_financial_table_with_split_iss_value() -> None:
    document = Document(document_id="doc-split-iss")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Valor Retencoes Base Calculo ISS Aliquota ISS Valor do ISS", y=26.0),
        *_line(document.document_id, 1, 3, "0,00 120,00 5,00% 0", y=42.0),
        *_line(document.document_id, 1, 4, ",00", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["taxable_amount"] == "120,00"
    assert values["iss_rate"] == "5,00%"
    assert values["iss_amount"] == "0,00"


def test_output_normalizer_maps_net_amount_after_total_tax_placeholder() -> None:
    document = Document(document_id="doc-net-after-tax-placeholder")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "PIS Outras Retencoes Total Trib. Federais Valor Liquido", y=26.0),
        *_line(document.document_id, 1, 3, "0,00 0,00 0,00 12.044,00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["pis_withheld_amount"] == "0,00"
    assert values["other_retentions_amount"] == "0,00"
    assert values["net_amount"] == "12.044,00"


def test_output_normalizer_maps_unmarked_rate_when_table_shape_is_clear() -> None:
    document = Document(document_id="doc-unmarked-rate")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Descricao Servico Prestado Valor Base Calculo Aliquota ISSQN", y=26.0),
        *_line(document.document_id, 1, 3, "SUPERVISAO DE LOJAS 6.000,00 6.000,00 3,00 180,00", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["gross_amount"] == "6.000,00"
    assert values["taxable_amount"] == "6.000,00"
    assert values["iss_rate"] == "3,00"
    assert values["iss_amount"] == "180,00"


def test_output_normalizer_maps_summary_table_with_discount_placeholder() -> None:
    document = Document(document_id="doc-summary-discount")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Valor Total Desconto Dedução Base de Cálculo ISSQN", y=26.0),
        *_line(document.document_id, 1, 3, "12.044,00 0,00 0,00 12.044,00 602,20", y=42.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["gross_amount"] == "12.044,00"
    assert values["deductions_amount"] == "0,00"
    assert values["taxable_amount"] == "12.044,00"
    assert values["iss_amount"] == "602,20"


def test_output_normalizer_maps_single_base_value_without_inventing_rate_or_iss() -> None:
    document = Document(document_id="doc-single-base-value")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Base de Cálculo do ISSQN ALÍQUOTA DO ISSQN", y=26.0),
        *_line(document.document_id, 1, 3, "Deduções do ISSQN ISSQN DEVIDO", y=42.0),
        *_line(document.document_id, 1, 4, "R$ 61,29", y=58.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["taxable_amount"] == "R$ 61,29"
    assert "iss_rate" not in values
    assert "iss_amount" not in values


def test_output_normalizer_ignores_informational_retention_law_lines() -> None:
    document = Document(document_id="doc-financial-noise")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            "E DISP.RET.PIS/COFINS/CSSL/ PGTO VLR IGUAL OU MENOR R$215,05 CONF-.LEI 13.137/15.",
            y=26.0,
        ),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert "pis_withheld_amount" not in values
    assert "cofins_withheld_amount" not in values
    assert "csll_withheld_amount" not in values


def test_output_normalizer_prefers_structural_financial_candidates() -> None:
    document = Document(document_id="doc-financial-ranking")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(document.document_id, 1, 2, "Valor ISS", y=26.0),
        *_line(document.document_id, 1, 3, "24,09", y=42.0),
        *_line(
            document.document_id,
            1,
            4,
            "Serviço Local Prestação i Alíquota Situação Trib. Valor Serviço Desc. Incondic. Valor Dedução Valor ISS",
            y=58.0,
        ),
        *_line(document.document_id, 1, 5, "2601 8039 5% TI 12.044,00 0,00 0,00 602,20", y=74.0),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert _values_for(candidates, "iss_amount") == ["602,20"]
    assert values["gross_amount"] == "12.044,00"
    assert values["iss_rate"] == "5%"
    assert values["iss_amount"] == "602,20"


def test_output_normalizer_corrects_merged_net_amount_credit_line() -> None:
    document = Document(document_id="doc-merged-net-credit")
    elements = [
        *_line(document.document_id, 1, 1, "VALORES", y=10.0),
        *_line(
            document.document_id,
            1,
            2,
            (
                "Valor 12.044,00 Base 0,00 PIS Total de Cálculo para o Crédito Outras "
                "Desconto 0,00 0,00 Retenções Total Alíquota Dedução Trib. 0,00 0,00 "
                "Utilizada Federais Base 12.044,00 de Cálculo Valor 12.044,00 "
                "Valor Líquido do Crédito 602,20 ISSQN"
            ),
            y=26.0,
        ),
    ]

    candidates = ConfigDrivenOutputNormalizer().normalize(document, elements)
    values = _candidate_values(candidates)

    assert values["net_amount"] == "12.044,00"


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
    assert provider_document.metadata["section_confidence"] > 0
    assert provider_document.metadata["section_reasons"]
    assert provider_document.metadata["label_text"] in {"cnpj cpf", "document pattern"}
