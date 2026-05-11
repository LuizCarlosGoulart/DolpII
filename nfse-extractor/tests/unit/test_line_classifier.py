from src.normalization.line_classifier import IGNORED_SECTION, classify_line_sections


def _sections(texts: list[str]) -> list[str]:
    return [classification.section for classification in classify_line_sections(texts)]


def test_line_classifier_marks_explicit_nfse_sections() -> None:
    sections = _sections(
        [
            "PREFEITURA MUNICIPAL Nota Fiscal Eletronica",
            "PRESTADOR DE SERVICOS",
            "CNPJ/CPF: 11.111.111/0001-11",
            "TOMADOR DO SERVICO",
            "CNPJ/CPF: 22.222.222/0001-22",
            "DISCRIMINACAO DOS SERVICOS",
            "Manutencao preventiva mensal",
            "VALORES Base de Calculo Valor ISS",
        ]
    )

    assert sections == [
        "header",
        "provider",
        "provider",
        "recipient",
        "recipient",
        "service",
        "service",
        "values",
    ]


def test_line_classifier_infers_provider_before_recipient_without_header() -> None:
    classifications = classify_line_sections(
        [
            "NOTA FISCAL ELETRONICA DE SERVICOS",
            "Nome/Razao Social: EMPRESA PRESTADORA LTDA",
            "CNPJ/CPF: 11.111.111/0001-11",
            "Endereco: Rua Um 100",
            "TOMADOR DO SERVICO",
            "Nome/Razao Social: CLIENTE LTDA",
        ]
    )

    assert [classification.section for classification in classifications] == [
        "header",
        "provider",
        "provider",
        "provider",
        "recipient",
        "recipient",
    ]
    assert "implicit_provider_before_recipient" in classifications[1].reasons


def test_line_classifier_ignores_billing_lines_until_fiscal_header() -> None:
    classifications = classify_line_sections(
        [
            "Local de Pagamento Vencimento Valor do Documento",
            "CNPJ/CPF 99.999.999/0001-99 Nosso Numero 123456",
            "Autenticacao Mecanica",
            "NOTA FISCAL ELETRONICA DE SERVICOS",
            "Nome/Razao Social: PRESTADOR REAL LTDA",
        ]
    )

    assert [classification.section for classification in classifications[:3]] == [
        IGNORED_SECTION,
        IGNORED_SECTION,
        IGNORED_SECTION,
    ]
    assert [classification.section for classification in classifications[3:]] == ["header", "provider"]
