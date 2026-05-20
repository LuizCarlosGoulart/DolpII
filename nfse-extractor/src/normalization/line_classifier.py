"""Lightweight OCR line section classification for NFS-e layouts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


PRODUCTIVE_SECTIONS = {"header", "provider", "recipient", "service", "values"}
IGNORED_SECTION = "ignored_billing"
_SECTION_SCORE_ORDER = (IGNORED_SECTION, "provider", "recipient", "service", "values", "header")


@dataclass(frozen=True)
class LineClassification:
    """Section assigned to one OCR line with traceable scoring reasons."""

    section: str
    confidence: float
    reasons: tuple[str, ...]


def classify_line_sections(texts: list[str]) -> list[LineClassification]:
    """Classify OCR lines using transparent, layout-agnostic fiscal cues."""
    classifications: list[LineClassification] = []
    current_section = "header"
    saw_recipient = False
    saw_service = False

    for text in texts:
        scored = _score_line(text)
        explicit_section = scored.section
        explicit_score = scored.score
        reasons = list(scored.reasons)
        normalized = _normalize(text)

        if explicit_section == IGNORED_SECTION and explicit_score >= 3:
            section = IGNORED_SECTION
            confidence = 0.95
        elif explicit_section in {"provider", "recipient", "service", "values"} and explicit_score >= 3:
            section = explicit_section
            confidence = 0.92
        elif current_section == IGNORED_SECTION and explicit_section not in {"header", "provider", "recipient", "service"}:
            section = IGNORED_SECTION
            confidence = 0.75
            reasons.append("billing_continuation")
        elif _looks_like_party_detail(normalized) and current_section in {"provider", "recipient"}:
            section = current_section
            confidence = 0.78
            reasons.append(f"{current_section}_detail_continuation")
        elif (
            _looks_like_party_detail(normalized)
            and not saw_recipient
            and not saw_service
            and current_section == "header"
            and not _looks_like_municipal_header(normalized)
        ):
            section = "provider"
            confidence = 0.72
            reasons.append("implicit_provider_before_recipient")
        elif _looks_like_service_detail(normalized) and current_section == "service":
            section = "service"
            confidence = 0.72
            reasons.append("service_detail_continuation")
        elif _looks_like_value_detail(normalized) and current_section in {"service", "values"}:
            section = "values"
            confidence = 0.72
            reasons.append("value_detail_continuation")
        elif explicit_section == "header" and explicit_score >= 2:
            section = "header"
            confidence = 0.85
        elif explicit_score > 0 and explicit_section in PRODUCTIVE_SECTIONS:
            section = explicit_section
            confidence = 0.62
        else:
            section = current_section if current_section != IGNORED_SECTION else "header"
            confidence = 0.55
            reasons.append("section_continuation")

        if section in PRODUCTIVE_SECTIONS:
            current_section = section
        elif section == IGNORED_SECTION:
            current_section = IGNORED_SECTION

        if section == "recipient":
            saw_recipient = True
        if section == "service":
            saw_service = True

        classifications.append(
            LineClassification(
                section=section,
                confidence=confidence,
                reasons=tuple(dict.fromkeys(reasons or ["section_continuation"])),
            )
        )

    return classifications


@dataclass(frozen=True)
class _ScoredLine:
    section: str
    score: int
    reasons: tuple[str, ...]


def _score_line(text: str) -> _ScoredLine:
    normalized = _normalize(text)
    scores: dict[str, int] = {section: 0 for section in _SECTION_SCORE_ORDER}
    reasons: dict[str, list[str]] = {section: [] for section in scores}

    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        IGNORED_SECTION,
        {
            "autenticacao mecanica": 5,
            "beneficiario": 3,
            "boleto": 5,
            "carteira": 3,
            "cedente": 3,
            "codigo beneficiario": 4,
            "especie moeda": 4,
            "linha digitavel": 5,
            "local de pagamento": 5,
            "nosso numero": 5,
            "pagador": 3,
            "recibo do sacado": 5,
            "sacado": 3,
            "valor cobrado": 4,
            "valor do documento": 5,
            "vencimento": 3,
        },
    )
    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        "header",
        {
            "data emissao": 2,
            "emissao": 2,
            "municipio de": 2,
            "nota fiscal": 4,
            "nfs e": 4,
            "nfse": 4,
            "nota no": 2,
            "numero da nota": 4,
            "prefeitura": 3,
            "secretaria": 2,
        },
    )
    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        "provider",
        {
            "dados do prestador": 7,
            "prestador": 6,
            "prestador de servicos": 8,
        },
    )
    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        "recipient",
        {
            "dados do tomador": 7,
            "destinatario": 4,
            "tomador": 6,
            "tomador de servicos": 8,
            "tomador do servico": 8,
        },
    )
    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        "service",
        {
            "atividade": 3,
            "codigo do servico": 5,
            "descricao do servico": 6,
            "descricao dos servicos": 6,
            "discriminacao do servico": 7,
            "discriminacao dos servicos": 7,
            "local da prestacao": 4,
            "local do recolhimento": 4,
            "municipio de incidencia": 4,
            "natureza da operacao": 5,
            "recolhimento": 2,
            "servicos prestados": 5,
            "subitem": 3,
            "subitens": 3,
            "tributacao": 4,
        },
    )
    _add_phrase_scores(
        normalized,
        scores,
        reasons,
        "values",
        {
            "aliquota": 3,
            "base de calculo": 5,
            "deducoes": 3,
            "desconto": 3,
            "impostos adicionais": 3,
            "iss retido": 4,
            "outras retencoes": 4,
            "retencoes": 3,
            "valor bruto": 5,
            "valor iss": 4,
            "valor liquido": 5,
            "valor servico": 4,
            "valor total": 4,
        },
    )

    # Densely numeric billing lines should not leak fiscal identifiers upward.
    if _digit_density(normalized) > 0.55 and any(term in normalized for term in ("vencimento", "documento", "pagamento")):
        scores[IGNORED_SECTION] += 3
        reasons[IGNORED_SECTION].append("numeric_billing_density")

    section = max(_SECTION_SCORE_ORDER, key=lambda item: scores[item])
    score = scores[section]
    if score == 0:
        section = "header"
    return _ScoredLine(section=section, score=score, reasons=tuple(reasons[section]))


def _add_phrase_scores(
    normalized: str,
    scores: dict[str, int],
    reasons: dict[str, list[str]],
    section: str,
    phrase_scores: dict[str, int],
) -> None:
    for phrase, score in phrase_scores.items():
        if _contains_phrase(normalized, phrase):
            scores[section] += score
            reasons[section].append(phrase)


def _looks_like_party_detail(normalized: str) -> bool:
    return _contains_any_phrase(
        normalized,
        (
            "cnpj",
            "cpf",
            "email",
            "endereco",
            "e mail",
            "insc municipal",
            "inscricao municipal",
            "municipio",
            "nome fantasia",
            "nome razao social",
            "razao social",
            "telefone",
            "uf",
        ),
    )


def _looks_like_service_detail(normalized: str) -> bool:
    return _contains_any_phrase(
        normalized,
        (
            "codigo do servico",
            "item da lista",
            "local da prestacao",
            "local do recolhimento",
            "municipio de incidencia",
            "natureza",
            "servico",
            "servicos",
        ),
    )


def _looks_like_value_detail(normalized: str) -> bool:
    return _contains_any_phrase(
        normalized,
        (
            "aliquota",
            "base de calculo",
            "cofins",
            "deducoes",
            "desconto",
            "inss",
            "iss",
            "pis",
            "retencoes",
            "valor",
        ),
    )


def _looks_like_municipal_header(normalized: str) -> bool:
    return "prefeitura" in normalized or "secretaria" in normalized or normalized.startswith("municipio de ")


def _digit_density(normalized: str) -> float:
    compact = normalized.replace(" ", "")
    if not compact:
        return 0.0
    return sum(character.isdigit() for character in compact) / len(compact)


def _contains_phrase(normalized: str, phrase: str) -> bool:
    pattern = r"(?:^|\s)" + re.escape(phrase) + r"(?:\s|$)"
    return re.search(pattern, normalized) is not None


def _contains_any_phrase(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(normalized, phrase) for phrase in phrases)


def _normalize(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().replace("@", " @ ")
    text = re.sub(r"[^a-z0-9@]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
