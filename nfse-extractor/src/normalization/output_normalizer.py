"""Config-driven conversion from raw OCR elements into field candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Any

import yaml

from src.core import Document, ExtractedElement, FieldCandidate, OutputNormalizer, load_field_dictionary
from src.normalization.line_classifier import IGNORED_SECTION, classify_line_sections


_LABEL_STOP_TOKENS = {
    "aliquota",
    "bairro",
    "base",
    "cep",
    "cnpj",
    "cofins",
    "compl",
    "cpf",
    "data",
    "deducoes",
    "desconto",
    "email",
    "emissao",
    "endereco",
    "fone",
    "inss",
    "insc",
    "inscricao",
    "irrf",
    "iss",
    "municipal",
    "municipio",
    "natureza",
    "nome",
    "numero",
    "pais",
    "pis",
    "prrf",
    "serie",
    "social",
    "telefone",
    "tomador",
    "uf",
    "valor",
}

_SECTION_SCOPED_LABELS = {
    ("cnpj", "cpf"): {"provider": "provider_document", "recipient": "recipient_document"},
    ("cpf", "cnpj"): {"provider": "provider_document", "recipient": "recipient_document"},
    ("nome", "razao", "social"): {"provider": "provider_name", "recipient": "recipient_name"},
    ("insc", "municipal"): {
        "provider": "provider_municipal_registration",
        "recipient": "recipient_municipal_registration",
    },
    ("inscricao", "municipal"): {
        "provider": "provider_municipal_registration",
        "recipient": "recipient_municipal_registration",
    },
    ("endereco",): {"provider": "provider_address", "recipient": "recipient_address"},
    ("email",): {"provider": "provider_email", "recipient": "recipient_email"},
    ("e", "mail"): {"provider": "provider_email", "recipient": "recipient_email"},
    ("telefone",): {"provider": "provider_phone", "recipient": "recipient_phone"},
    ("fone",): {"provider": "provider_phone", "recipient": "recipient_phone"},
    ("uf",): {"provider": "provider_uf", "recipient": "recipient_uf"},
}

_PATTERN_FIELD_HINTS = {
    "document_id": re.compile(r"\b(?:\d{2}[\.,]\d{3}\.\d{3}/\d{4}-\d{2}|\d{14}|\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b"),
    "email": re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b"),
    "date": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "month_year": re.compile(r"\b\d{2}/\d{4}\b"),
    "service_code": re.compile(r"\b(?:\d{2}\.\d{2}\.\d{2}|\d{1,2}\.\d{2}|\d{3,4})\b"),
    "money": re.compile(r"(?:R\$\s*)?\b\d{1,3}(?:\.\d{3})*,\d{2}\b|\b\d+(?:\.\d{2})\b"),
    "percentage": re.compile(r"\b(?:\d{1,2},\d{1,4}%?|\d{1,2}%)"),
    "verification_code": re.compile(r"\b(?:[A-Z0-9]{4,12}-[A-Z0-9]{2,12}(?:-[A-Z0-9]{2,12}){0,4}|[A-Z0-9]{5,12})\b"),
}

_DOCUMENT_FIELDS = {"provider_document", "recipient_document"}
_EMAIL_FIELDS = {"provider_email", "recipient_email"}
_MUNICIPAL_REGISTRATION_FIELDS = {"provider_municipal_registration", "recipient_municipal_registration"}
_MONEY_FIELDS = {
    "gross_amount",
    "taxable_amount",
    "iss_amount",
    "iss_withheld_amount",
    "unconditional_discount",
    "conditional_discount",
    "pis_withheld_amount",
    "cofins_withheld_amount",
    "inss_withheld_amount",
    "ir_withheld_amount",
    "csll_withheld_amount",
    "social_contributions_withheld_amount",
    "other_retentions_amount",
    "deductions_amount",
    "net_amount",
}
_PHONE_FIELDS = {"provider_phone", "recipient_phone"}
_UF_FIELDS = {"provider_uf", "recipient_uf"}
_SECTION_ONLY_VALUES = {
    "de servicos",
    "dos servicos",
    "prestador de servicos",
    "tomador de servicos",
    "discriminacao do servico",
    "discriminacao dos servicos",
    "do servico",
    "do servicos",
}

_SERVICE_DESCRIPTION_STOP_TOKENS = {
    "aliquota",
    "base",
    "codigo",
    "cofins",
    "deducoes",
    "desconto",
    "inss",
    "iss",
    "item",
    "local",
    "municipio",
    "natureza",
    "pis",
    "retencoes",
    "valor",
    "valores",
}

_NONZERO_TABLE_VALUE_REQUIRES_EXPLICIT_LABEL = {
    "deductions_amount",
}

_SERVICE_NEARBY_FIELDS = {"operation_nature", "service_city", "service_code"}
_SERVICE_TEXT_FIELDS = {"operation_nature", "service_city", "service_description"}

_VERIFICATION_CODE_STOP_VALUES = {
    "ASSINATURA",
    "AUTENTICACAO",
    "AUTENTICIDADE",
    "CERTIFICACAO",
    "CODIGO",
    "CONTROLE",
    "DIGITAL",
    "DOCUMENTO",
    "ELETRONICA",
    "ELETRONICO",
    "ELETRONICANFSE",
    "FISCAL",
    "MODELO",
    "NFSE",
    "NFS",
    "NOTA",
    "PRESTADOR",
    "RPS",
    "SERVICO",
    "SERVICOS",
    "VALIDACAO",
    "VERIFICACAO",
}


@dataclass(frozen=True)
class _Line:
    page_number: int
    block_num: int | None
    line_num: int | None
    text: str
    elements: list[ExtractedElement]
    bounding_box: tuple[float, float, float, float] | None
    section: str
    section_confidence: float
    section_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _LabelMatch:
    field_name: str
    label_text: str
    end_element_index: int
    token_count: int


class ConfigDrivenOutputNormalizer(OutputNormalizer):
    """Create field candidates from OCR elements using labels, patterns, and layout context."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        normalizer_name: str = "config-driven-output-normalizer",
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
        self.normalizer_name = normalizer_name
        self.field_dictionary = load_field_dictionary(self.config_dir / "field_dictionary.yaml")
        self.field_map = self.field_dictionary.by_internal_name()
        self.aliases = self._load_yaml("field_aliases.yaml").get("aliases", {})
        self._label_aliases = self._build_label_aliases()

    def normalize(
        self,
        document: Document,
        elements: list[ExtractedElement],
    ) -> list[FieldCandidate]:
        lines = self._build_lines(elements)
        candidates: list[FieldCandidate] = []

        for line_index, line in enumerate(lines):
            if line.section == IGNORED_SECTION:
                continue
            candidates.extend(self._candidates_from_labels(document, line, line_index, lines))
            candidates.extend(self._candidates_from_patterns(document, line, line_index))

        return self._deduplicate_candidates(candidates)

    def _build_label_aliases(self) -> list[tuple[tuple[str, ...], str, str]]:
        label_aliases: list[tuple[tuple[str, ...], str, str]] = []
        for field in self.field_dictionary.fields:
            aliases = {field.internal_name.replace("_", " "), *field.aliases, *self.aliases.get(field.internal_name, [])}
            for alias in aliases:
                tokens = tuple(_tokens(alias))
                if field.internal_name == "nfse_number" and tokens in {("nfs", "e"), ("nfse",)}:
                    continue
                if field.internal_name == "provider_name" and tokens == ("prestador",):
                    continue
                if field.internal_name == "recipient_name" and tokens == ("tomador",):
                    continue
                if tokens == ("iss",) and field.internal_name == "iss_amount":
                    continue
                if tokens:
                    label_aliases.append((tokens, field.internal_name, alias))
        label_aliases.sort(key=lambda item: len(item[0]), reverse=True)
        return label_aliases

    def _build_lines(self, elements: list[ExtractedElement]) -> list[_Line]:
        groups: dict[tuple[Any, ...], list[ExtractedElement]] = {}
        for element in elements:
            if not element.text.strip():
                continue
            key = self._line_key(element)
            groups.setdefault(key, []).append(element)

        raw_lines: list[tuple[int, float, float, list[ExtractedElement]]] = []
        for key, line_elements in groups.items():
            ordered_elements = sorted(line_elements, key=lambda item: _bbox_x(item.bounding_box))
            page_number = int(key[0] or 1)
            top = min(_bbox_y(element.bounding_box) for element in ordered_elements)
            left = min(_bbox_x(element.bounding_box) for element in ordered_elements)
            raw_lines.append((page_number, top, left, ordered_elements))

        raw_lines.sort(key=lambda item: (item[0], item[1], item[2]))

        line_texts = [" ".join(element.text for element in item[3]).strip() for item in raw_lines]
        classifications = classify_line_sections(line_texts)

        lines: list[_Line] = []
        for (page_number, _top, _left, line_elements), classification in zip(raw_lines, classifications):
            line_text = " ".join(element.text for element in line_elements).strip()
            first = line_elements[0]
            lines.append(
                _Line(
                    page_number=page_number,
                    block_num=_optional_int(first.metadata.get("block_num")),
                    line_num=_optional_int(first.metadata.get("line_num")),
                    text=line_text,
                    elements=line_elements,
                    bounding_box=_merge_bounding_boxes([element.bounding_box for element in line_elements]),
                    section=classification.section,
                    section_confidence=classification.confidence,
                    section_reasons=classification.reasons,
                )
            )
        return lines

    @staticmethod
    def _line_key(element: ExtractedElement) -> tuple[Any, ...]:
        page_number = element.page_number or 1
        block_num = element.metadata.get("block_num")
        line_num = element.metadata.get("line_num")
        if block_num is not None and line_num is not None:
            return (page_number, block_num, line_num)
        return (page_number, round(_bbox_y(element.bounding_box) / 12))

    def _candidates_from_labels(
        self,
        document: Document,
        line: _Line,
        line_index: int,
        lines: list[_Line],
    ) -> list[FieldCandidate]:
        matches = self._find_label_matches(line)
        candidates: list[FieldCandidate] = []
        for match_index, match in enumerate(matches):
            next_match = matches[match_index + 1] if match_index + 1 < len(matches) else None
            value_elements = self._value_elements_after_label(line, match, next_match)
            value_source = "same_line"
            raw_value = self._clean_value(" ".join(element.text for element in value_elements))
            value = self._extract_typed_value(match.field_name, raw_value)

            if value is None and match.field_name in _EMAIL_FIELDS:
                value = self._extract_ocr_corrected_email(raw_value)
                if value is not None:
                    value_source = "ocr_corrected_email"

            if value is None:
                value = self._extract_inline_value(match.field_name, line.text)

            if value is None and match.field_name == "verification_code":
                nearby_code = self._extract_verification_code_from_nearby(line_index, lines)
                if nearby_code is not None:
                    value, value_elements = nearby_code
                    value_source = "nearby_lines"

            if value is None and line_index + 1 < len(lines):
                table_value = self._extract_table_value(match, matches, lines[line_index + 1])
                if table_value is not None:
                    value = table_value
                    value_elements = lines[line_index + 1].elements
                    value_source = "next_line_table"

            if value is None and self._can_scan_nearby(match.field_name):
                for nearby_line in lines[line_index + 1 : line_index + 5]:
                    if not self._can_use_nearby_line(match.field_name, nearby_line):
                        continue
                    value = self._extract_typed_value(match.field_name, nearby_line.text)
                    if value is not None:
                        value_elements = nearby_line.elements
                        value_source = "nearby_line"
                        break

            if value is None and not self._requires_typed_value(match.field_name):
                value = raw_value

            if not value and match.field_name == "service_description":
                service_description = self._extract_service_description_after_header(line, line_index, lines)
                if service_description is not None:
                    value, value_elements = service_description
                    value_source = "following_service_lines"

            if not value:
                continue
            if not self._is_acceptable_value(match.field_name, value):
                if match.field_name != "service_description":
                    continue
                service_description = self._extract_service_description_after_header(line, line_index, lines)
                if service_description is None:
                    continue
                value, value_elements = service_description
                value_source = "following_service_lines"

            candidates.append(
                self._build_candidate(
                    document=document,
                    field_name=match.field_name,
                    value=value,
                    source_elements=value_elements,
                    label_text=match.label_text,
                    line=line,
                    line_index=line_index,
                    value_source=value_source,
                    confidence_boost=0.08,
                )
            )
        return candidates

    def _find_label_matches(self, line: _Line) -> list[_LabelMatch]:
        flattened = self._flatten_line_tokens(line)
        token_values = [token for token, _element_index in flattened]
        matches: list[_LabelMatch] = []

        for label_tokens, field_name, label_text in self._label_aliases:
            position = _find_subsequence(token_values, label_tokens)
            if position is None:
                continue
            end_token_index = position + len(label_tokens) - 1
            end_element_index = flattened[end_token_index][1]
            matches.append(
                _LabelMatch(
                    field_name=self._scope_field(field_name, label_tokens, line.section),
                    label_text=label_text,
                    end_element_index=end_element_index,
                    token_count=len(label_tokens),
                )
            )

        for label_tokens, scoped_fields in _SECTION_SCOPED_LABELS.items():
            position = _find_subsequence(token_values, label_tokens)
            if position is None:
                continue
            scoped_field = scoped_fields.get(line.section)
            if scoped_field is None:
                continue
            end_token_index = position + len(label_tokens) - 1
            matches.append(
                _LabelMatch(
                    field_name=scoped_field,
                    label_text=" ".join(label_tokens),
                    end_element_index=flattened[end_token_index][1],
                    token_count=len(label_tokens),
                )
            )

        deduped: dict[str, _LabelMatch] = {}
        for match in matches:
            current = deduped.get(match.field_name)
            if current is None:
                deduped[match.field_name] = match
            elif match.field_name in _SERVICE_TEXT_FIELDS and (
                match.token_count > current.token_count
                or (match.token_count == current.token_count and match.end_element_index < current.end_element_index)
            ):
                deduped[match.field_name] = match
            elif match.field_name not in _SERVICE_TEXT_FIELDS and match.end_element_index > current.end_element_index:
                deduped[match.field_name] = match
        return sorted(deduped.values(), key=lambda item: item.end_element_index)

    @staticmethod
    def _flatten_line_tokens(line: _Line) -> list[tuple[str, int]]:
        flattened: list[tuple[str, int]] = []
        for element_index, element in enumerate(line.elements):
            for token in _tokens(element.text):
                flattened.append((token, element_index))
        return flattened

    @staticmethod
    def _scope_field(field_name: str, label_tokens: tuple[str, ...], section: str) -> str:
        scoped_fields = _SECTION_SCOPED_LABELS.get(label_tokens)
        if scoped_fields and section in scoped_fields:
            return scoped_fields[section]
        return field_name

    @staticmethod
    def _value_elements_after_label(
        line: _Line,
        match: _LabelMatch,
        next_match: _LabelMatch | None,
    ) -> list[ExtractedElement]:
        start_index = match.end_element_index + 1
        end_index = (
            next_match.end_element_index
            if next_match and next_match.end_element_index > match.end_element_index
            else len(line.elements)
        )
        value_elements: list[ExtractedElement] = []
        ignore_stop_tokens = match.field_name in {"operation_nature", "service_description"}
        for element in line.elements[start_index:end_index]:
            tokens = _tokens(element.text)
            if not ignore_stop_tokens and value_elements and tokens and tokens[0] in _LABEL_STOP_TOKENS:
                break
            if not ignore_stop_tokens and not value_elements and tokens and tokens[0] in _LABEL_STOP_TOKENS:
                break
            if not tokens or tokens[0] in {":", "-"}:
                continue
            value_elements.append(element)
        return value_elements

    def _candidates_from_patterns(
        self,
        document: Document,
        line: _Line,
        line_index: int,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        normalized_line = _normalize(line.text)

        for match in _PATTERN_FIELD_HINTS["document_id"].finditer(line.text):
            if not _has_document_context(normalized_line, line.section, match.group(0)):
                continue
            if self._looks_like_phone_not_document(line.text, match):
                continue
            value = _normalize_document_id(match.group(0))
            field_name = "recipient_document" if line.section == "recipient" else "provider_document"
            candidates.append(
                self._build_candidate(
                    document=document,
                    field_name=field_name,
                    value=value,
                    source_elements=line.elements,
                    label_text="document pattern",
                    line=line,
                    line_index=line_index,
                    value_source="regex",
                    confidence_boost=0.03,
                )
            )

        if "emissao" in normalized_line:
            candidates.extend(self._regex_candidates(document, line, line_index, "issue_date", "date"))

        if line.section in {"provider", "recipient"}:
            field_name = "recipient_email" if line.section == "recipient" else "provider_email"
            candidates.extend(self._regex_candidates(document, line, line_index, field_name, "email"))

        if line.section == "service" and _has_service_code_context(normalized_line):
            candidates.extend(self._regex_candidates(document, line, line_index, "service_code", "service_code"))

        return candidates

    def _extract_typed_value(self, field_name: str, value: str) -> str | None:
        value = self._clean_value(value)
        if not value:
            return None

        if field_name in _DOCUMENT_FIELDS:
            return _extract_document_id(value)
        if field_name in _MUNICIPAL_REGISTRATION_FIELDS:
            return _extract_municipal_registration(value)
        if field_name in _EMAIL_FIELDS:
            match = _PATTERN_FIELD_HINTS["email"].search(value)
            return match.group(0) if match else None
        if field_name in _UF_FIELDS:
            match = re.search(r"\b[A-Z]{2}\b", value.upper())
            return match.group(0) if match else None
        if field_name in _PHONE_FIELDS:
            phones = re.findall(r"\b0?\d{7,11}\b", value.replace("/", " "))
            return " ".join(phones) if phones else None
        if field_name == "issue_date":
            match = _PATTERN_FIELD_HINTS["date"].search(value)
            return match.group(0) if match else None
        if field_name == "nfse_series":
            return _extract_nfse_series(value)
        if field_name == "competence_date":
            match = _PATTERN_FIELD_HINTS["month_year"].search(value)
            if match:
                return match.group(0)
            match = _PATTERN_FIELD_HINTS["date"].search(value)
            return match.group(0) if match else None
        if field_name == "service_code":
            return _extract_service_code(value)
        if field_name in _SERVICE_TEXT_FIELDS:
            return _clean_service_text_value(field_name, value)
        if field_name == "iss_rate":
            match = _PATTERN_FIELD_HINTS["percentage"].search(value)
            return match.group(0) if match else None
        if field_name in _MONEY_FIELDS:
            match = next(iter(_money_matches(value)), None)
            return match.group(0).strip() if match else None
        if field_name == "verification_code":
            return _extract_verification_code(value)
        if field_name == "nfse_number":
            if "/" in value or _PATTERN_FIELD_HINTS["date"].search(value):
                return None
            number_matches = re.findall(r"\b\d[\d.]{0,11}\b", value)
            if not number_matches:
                return None
            plain_numbers = [number for number in number_matches if "." not in number and number.isdigit()]
            return (plain_numbers or number_matches)[-1]

        return value

    def _extract_inline_value(self, field_name: str, line_text: str) -> str | None:
        if field_name not in _UF_FIELDS:
            return None
        match = re.search(r"\bUF\s*:?\s*([A-Z]{2})\b", line_text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_table_value(
        self,
        match: _LabelMatch,
        matches: list[_LabelMatch],
        next_line: _Line,
    ) -> str | None:
        if match.field_name not in _MONEY_FIELDS and match.field_name != "iss_rate":
            return None

        value_matches = _money_matches(next_line.text)
        table_matches = [
            item
            for item in matches
            if item.field_name in _MONEY_FIELDS
        ]
        if match.field_name == "iss_rate":
            value_matches = list(_PATTERN_FIELD_HINTS["percentage"].finditer(next_line.text))
            table_matches = [
                item
                for item in matches
                if item.field_name == "iss_rate"
            ]
        if not value_matches:
            return None

        table_matches.sort(key=lambda item: item.end_element_index)
        if len(value_matches) < len(table_matches):
            return None
        try:
            value_index = table_matches.index(match)
        except ValueError:
            return None
        if value_index >= len(value_matches):
            return None
        value_match = value_matches[value_index]
        value = value_match.group(0).strip()
        if match.field_name == "iss_rate" and "%" not in value:
            return None
        if (
            match.field_name in _NONZERO_TABLE_VALUE_REQUIRES_EXPLICIT_LABEL
            and not _is_zero_money(value)
            and _normalize(match.label_text) == "deducoes"
        ):
            return None
        return value

    @staticmethod
    def _extract_ocr_corrected_email(value: str) -> str | None:
        value = re.sub(r"\s+", " ", value.strip())
        if not value:
            return None

        patterns = (
            r"\b([A-Za-z0-9._%+-]{2,}?)\s+[OQ0]([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b",
            r"\b([A-Za-z0-9._%+-]{2,}?)[OQ0][)\]]?([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, value)
            if not match:
                continue
            local_part, domain = match.groups()
            if domain.lower().startswith("www."):
                continue
            return f"{local_part}@{domain}".lower()
        return None

    def _extract_service_description_after_header(
        self,
        line: _Line,
        line_index: int,
        lines: list[_Line],
    ) -> tuple[str, list[ExtractedElement]] | None:
        normalized_header = _normalize(line.text)
        if not any(keyword in normalized_header for keyword in ("descricao", "discriminacao", "servicos prestados")):
            return None

        description_lines: list[_Line] = []
        for nearby_line in lines[line_index + 1 : line_index + 7]:
            normalized = _normalize(nearby_line.text)
            if nearby_line.section != "service":
                break
            if normalized in _SECTION_ONLY_VALUES:
                continue
            tokens = _tokens(nearby_line.text)
            if not tokens or tokens[0] in _SERVICE_DESCRIPTION_STOP_TOKENS:
                continue
            if len(tokens) < 3:
                continue
            if _PATTERN_FIELD_HINTS["money"].search(nearby_line.text):
                break
            description_lines.append(nearby_line)
            if len(description_lines) >= 3:
                break

        if not description_lines:
            return None

        value = _clean_service_text_value("service_description", self._clean_value(" ".join(item.text for item in description_lines)))
        if value is None or not self._is_acceptable_value("service_description", value):
            return None
        elements = [element for item in description_lines for element in item.elements]
        return value, elements

    @staticmethod
    def _extract_verification_code_from_nearby(
        line_index: int,
        lines: list[_Line],
    ) -> tuple[str, list[ExtractedElement]] | None:
        nearby_lines = lines[line_index + 1 : line_index + 4]
        for window_size in range(min(3, len(nearby_lines)), 0, -1):
            selected = nearby_lines[:window_size]
            text = " ".join(item.text for item in selected)
            code = _extract_verification_code(text)
            if code is None:
                continue
            elements = [element for item in selected for element in item.elements]
            return code, elements
        return None

    @staticmethod
    def _can_scan_nearby(field_name: str) -> bool:
        return field_name in {"issue_date", "verification_code", "nfse_number"} | _SERVICE_NEARBY_FIELDS

    @staticmethod
    def _can_use_nearby_line(field_name: str, line: _Line) -> bool:
        if line.section == IGNORED_SECTION:
            return False
        if field_name in _SERVICE_NEARBY_FIELDS:
            normalized = _normalize(line.text)
            if line.section not in {"service", "values"}:
                return False
            if normalized in _SECTION_ONLY_VALUES:
                return False
            if field_name == "service_code":
                return _looks_like_nearby_service_code_line(line.text)
            return _clean_service_text_value(field_name, line.text) is not None
        if field_name == "verification_code":
            return _extract_verification_code(line.text) is not None
        return True

    @staticmethod
    def _requires_typed_value(field_name: str) -> bool:
        return (
            field_name in _DOCUMENT_FIELDS
            or field_name in _EMAIL_FIELDS
            or field_name in _MUNICIPAL_REGISTRATION_FIELDS
            or field_name in _UF_FIELDS
            or field_name in _PHONE_FIELDS
            or field_name in _MONEY_FIELDS
            or field_name in _SERVICE_TEXT_FIELDS
            or field_name in {"competence_date", "issue_date", "iss_rate", "nfse_number", "nfse_series", "service_code", "verification_code"}
        )

    @staticmethod
    def _is_acceptable_value(field_name: str, value: str) -> bool:
        normalized = _normalize(value)
        if normalized in _SECTION_ONLY_VALUES:
            return False
        if field_name in {"provider_name", "recipient_name", "service_description"} and normalized in _SECTION_ONLY_VALUES:
            return False
        if field_name == "service_description":
            if _PATTERN_FIELD_HINTS["money"].search(value):
                return False
            if any(
                phrase in normalized
                for phrase in (
                    "aliquota",
                    "base de calculo",
                    "valor total",
                    "valor iss",
                    "valor liquido",
                    "vencimento",
                )
            ):
                return False
        return True

    @staticmethod
    def _looks_like_phone_not_document(line_text: str, match: re.Match[str]) -> bool:
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if len(digits) != 11 or any(separator in value for separator in (".", "-", "/")):
            return False
        normalized_prefix = _normalize(line_text[: match.start()])
        return "telefone" in normalized_prefix or "fone" in normalized_prefix or "celular" in normalized_prefix

    def _regex_candidates(
        self,
        document: Document,
        line: _Line,
        line_index: int,
        field_name: str,
        pattern_name: str,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        if field_name == "service_code":
            value = _extract_service_code(line.text)
            if value is None:
                return []
            return [
                self._build_candidate(
                    document=document,
                    field_name=field_name,
                    value=value,
                    source_elements=line.elements,
                    label_text=f"{pattern_name} pattern",
                    line=line,
                    line_index=line_index,
                    value_source="regex",
                    confidence_boost=0.03,
                )
            ]
        for match in _PATTERN_FIELD_HINTS[pattern_name].finditer(line.text):
            candidates.append(
                self._build_candidate(
                    document=document,
                    field_name=field_name,
                    value=match.group(0),
                    source_elements=line.elements,
                    label_text=f"{pattern_name} pattern",
                    line=line,
                    line_index=line_index,
                    value_source="regex",
                    confidence_boost=0.03,
                )
            )
        return candidates

    def _build_candidate(
        self,
        *,
        document: Document,
        field_name: str,
        value: str,
        source_elements: list[ExtractedElement],
        label_text: str,
        line: _Line,
        line_index: int,
        value_source: str,
        confidence_boost: float,
    ) -> FieldCandidate:
        confidence_values = [element.confidence for element in source_elements if element.confidence is not None]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
        confidence = max(0.0, min(confidence + confidence_boost, 1.0))
        if value_source == "ocr_corrected_email":
            confidence = min(confidence * 0.75, 0.7)
        value_hash = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        candidate_id = f"{document.document_id}:{self.normalizer_name}:{field_name}:{line_index}:{len(source_elements)}:{value_hash}"

        return FieldCandidate(
            candidate_id=candidate_id,
            field_name=field_name,
            value=value,
            source_element_ids=[element.element_id for element in source_elements],
            source_name=self.normalizer_name,
            confidence=confidence,
            metadata={
                "label_text": label_text,
                "context_text": line.text,
                "section_name": line.section,
                "section_confidence": line.section_confidence,
                "section_reasons": line.section_reasons,
                "page_number": line.page_number,
                "line_text": line.text,
                "line_index": line_index,
                "value_source": value_source,
                "same_block_as_label": value_source == "same_line",
                "block_num": line.block_num,
                "label_block_num": line.block_num,
                "bounding_box": line.bounding_box,
                "ocr_correction_applied": value_source == "ocr_corrected_email",
            },
        )

    @staticmethod
    def _clean_value(value: str) -> str:
        value = value.strip(" :-\t\r\n")
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _deduplicate_candidates(candidates: list[FieldCandidate]) -> list[FieldCandidate]:
        deduped: dict[tuple[str, str, str], FieldCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.field_name,
                _normalize(candidate.value),
                str(candidate.metadata.get("section_name", "")),
            )
            current = deduped.get(key)
            if current is None or (candidate.confidence or 0.0) > (current.confidence or 0.0):
                deduped[key] = candidate

        singleton_fields = {"issue_date", "nfse_number", "nfse_series", "verification_code"}
        best_singletons: dict[str, FieldCandidate] = {}
        results: list[FieldCandidate] = []
        for candidate in deduped.values():
            if candidate.field_name not in singleton_fields:
                results.append(candidate)
                continue
            current = best_singletons.get(candidate.field_name)
            if current is None or _candidate_rank(candidate) > _candidate_rank(current):
                best_singletons[candidate.field_name] = candidate
        results.extend(best_singletons.values())
        return sorted(results, key=lambda item: (item.field_name, str(item.metadata.get("line_index", ""))))

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        with (self.config_dir / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}


def _candidate_rank(candidate: FieldCandidate) -> tuple[float, float, float]:
    confidence = candidate.confidence or 0.0
    field_name = candidate.field_name
    value = str(candidate.value)
    normalized_value = _normalize(value)
    context = _normalize(candidate.metadata.get("line_text", ""))
    label = _normalize(candidate.metadata.get("label_text", ""))
    score = confidence

    if field_name == "issue_date":
        if "emissao" in label or "emissao" in context or "data hora" in context:
            score += 0.45
        if any(term in context for term in ("autorizacao", "fato gerador", "rps", "vencimento")):
            score -= 0.65
    elif field_name == "nfse_number":
        if any(term in context for term in ("numero da nota", "numero da nfs", "nota no")):
            score += 0.45
        if any(term in context for term in ("rps", "recibo provisorio")):
            score -= 0.55
        score -= min(len(re.sub(r"\D", "", value)), 12) * 0.005
    elif field_name == "nfse_series":
        score += max(0.0, 0.35 - len(value) * 0.02)
        if label == "serie" or "serie" in context:
            score += 0.25
        if any(term in context for term in ("codigo de verificacao", "rps", "emitid")):
            score -= 0.15
    elif field_name == "verification_code":
        if any(term in context for term in ("codigo de verificacao", "codigo verificador", "certificacao")):
            score += 0.5
        if "-" in value:
            score += 0.2
        if normalized_value.upper() in _VERIFICATION_CODE_STOP_VALUES:
            score -= 1.0

    line_index = float(candidate.metadata.get("line_index", 0) or 0)
    return (score, confidence, -line_index)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(value))


def _normalize(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().replace("@", " @ ")
    text = re.sub(r"[^a-z0-9@]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_subsequence(values: list[str], expected: tuple[str, ...]) -> int | None:
    if not expected or len(expected) > len(values):
        return None
    for index in range(0, len(values) - len(expected) + 1):
        if tuple(values[index : index + len(expected)]) == expected:
            return index
    return None


def _money_matches(value: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in _PATTERN_FIELD_HINTS["money"].finditer(value):
        if value[match.end() : match.end() + 1] == "%":
            continue
        matches.append(match)
    return matches


def _is_zero_money(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return bool(digits) and set(digits) == {"0"}


def _extract_document_id(value: str) -> str | None:
    match = _PATTERN_FIELD_HINTS["document_id"].search(value)
    return _normalize_document_id(match.group(0)) if match else None


def _normalize_document_id(value: str) -> str:
    value = value.replace(",", ".")
    digits = re.sub(r"\D", "", value)
    if len(digits) == 14:
        return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
    if len(digits) == 11:
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
    return value


def _extract_municipal_registration(value: str) -> str | None:
    value = re.split(
        r"\b(?:estadual|insc\.?\s*estadual|inscricao\s+estadual)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    match = re.search(r"\b(?:isento|[A-Z0-9][A-Z0-9\.\-\/]{1,20})\b", value.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).strip(" .:-").upper() if match.group(0).lower() == "isento" else match.group(0).strip(" .:-")


def _extract_nfse_series(value: str) -> str | None:
    value = re.split(
        r"\b(?:codigo|c[oó]digo|data|emissao|emiss[aã]o|emitid[ao]|nota|rps)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" :-,.\t\r\n()")
    if not value:
        return None
    candidates = re.findall(r"\b[A-Z]{1,4}\d{0,4}\b|\b\d{1,4}\b", value.upper())
    if not candidates:
        return None
    return candidates[-1].strip(" :-,.()")


def _extract_verification_code(value: str) -> str | None:
    if "http" in value.lower() or "/" in value:
        return None
    compact_value = re.sub(r"-\s+", "-", value.upper())
    compact_value = re.sub(r"\s+-", "-", compact_value)
    for match in _PATTERN_FIELD_HINTS["verification_code"].finditer(compact_value):
        code = match.group(0).strip("-")
        if _is_valid_verification_code(code):
            return code
    return None


def _is_valid_verification_code(code: str) -> bool:
    normalized_code = _normalize(code).replace(" ", "").upper()
    if not normalized_code or normalized_code in _VERIFICATION_CODE_STOP_VALUES:
        return False
    segments = [_normalize(segment).replace(" ", "").upper() for segment in code.split("-")]
    if any(segment in _VERIFICATION_CODE_STOP_VALUES for segment in segments):
        return False
    if normalized_code.isdigit():
        return False
    if len(normalized_code) < 5:
        return False
    if "-" in code:
        return True
    return any(character.isdigit() for character in normalized_code) and any(character.isalpha() for character in normalized_code)


def _extract_service_code(value: str) -> str | None:
    for pattern in (r"\b\d{2}\.\d{2}\.\d{2}\b", r"\b\d{1,2}\.\d{2}\b", r"\b\d{4}\b", r"\b\d{3}\b"):
        match = re.search(pattern, value)
        if not match:
            continue
        if value[match.end() : match.end() + 1] in {",", "/", "."}:
            continue
        code = match.group(0)
        if code == "116" and "lei complementar" in _normalize(value):
            continue
        if code.isdigit() and 1900 <= int(code) <= 2099:
            continue
        return code
    return None


def _looks_like_nearby_service_code_line(value: str) -> bool:
    code = _extract_service_code(value)
    if code is None:
        return False
    normalized = _normalize(value)
    if _has_service_code_context(normalized):
        return True
    stripped_value = value.lstrip(" (")
    if not stripped_value.startswith(code):
        return False
    if "." in code:
        return True
    if _PATTERN_FIELD_HINTS["money"].search(value):
        return False
    return len(code) == 4 and len(_tokens(value)) >= 4


def _clean_service_text_value(field_name: str, value: str) -> str | None:
    value = re.sub(r"\s+", " ", value.strip(" :-\t\r\n"))
    if not value:
        return None

    if field_name == "service_description":
        value = re.sub(r"^(?:descri[cç][aã]o|discriminacao|discrimina[cç][aã]o)\s*:?\s*", "", value, flags=re.IGNORECASE)
    value = _truncate_service_text(field_name, value)
    if field_name == "service_city":
        value = re.sub(r"^\d{3,5}\s+", "", value).strip(" :-\t\r\n")
    normalized = _normalize(value)
    if not normalized or normalized in _SECTION_ONLY_VALUES:
        return None
    if _looks_like_service_text_noise(field_name, normalized, value):
        return None
    if field_name == "service_city":
        value = re.sub(r"\s*!\s*", "/", value)
        value = value.replace(" - ", "/")
    return value


def _truncate_service_text(field_name: str, value: str) -> str:
    if field_name == "operation_nature":
        stop_pattern = r"\b(?:codigo|c[óo]digo|local\s+d[aeo]|valor|base\s+de\s+c[aá]lculo)\b"
    elif field_name == "service_city":
        stop_pattern = r"\b(?:data|codigo|c[óo]digo|natureza|valor|base\s+de\s+c[aá]lculo|iss)\b"
    else:
        stop_pattern = (
            r"\b(?:atividade|base\s+de\s+c[aá]lculo|c[óo]digo\s+do\s+servi[çc]o|"
            r"informacoes\s+adicionais|informa[çc][õo]es\s+adicionais|local\s+d[aeo]|"
            r"natureza|outras\s+informacoes|outras\s+informa[çc][õo]es|valor|vencimento)\b"
        )
    parts = re.split(stop_pattern, value, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip(" :-\t\r\n")


def _looks_like_service_text_noise(field_name: str, normalized: str, value: str) -> bool:
    tokens = _tokens(value)
    if field_name == "service_description":
        if len(tokens) < 3:
            return True
        return any(
            phrase in normalized
            for phrase in (
                "aliquota",
                "base de calculo",
                "informacoes relevantes",
                "servico local prestacao",
                "valor iss",
                "valor liquido",
                "valor total",
            )
        )
    if field_name == "service_city":
        if normalized in {"da prestacao do servico", "de prestacao", "do servico"}:
            return True
        if "descricao do servico" in normalized or "descricao dos servicos" in normalized:
            return True
        if "outras informacoes" in normalized:
            return True
        if "servico local prestacao" in normalized:
            return True
        if "aliquota" in normalized or "situacao trib" in normalized or "tributad" in normalized:
            return True
        if "local de prestacao" in normalized or "prestacao do servico" in normalized:
            return True
        if "servicos prestados" in normalized or "ervicos prestados" in normalized:
            return True
        if len("".join(tokens)) < 3:
            return True
        return _PATTERN_FIELD_HINTS["money"].search(value) is not None
    if field_name == "operation_nature":
        if normalized in {"da operacao", "de operacao"}:
            return True
        if len("".join(tokens)) < 3:
            return True
        return _PATTERN_FIELD_HINTS["money"].search(value) is not None
    return False


def _has_service_code_context(normalized_line: str) -> bool:
    return any(
        phrase in normalized_line
        for phrase in (
            "atividade",
            "cnae",
            "cod servico",
            "codigo atividade",
            "codigo do servico",
            "codigo servico",
            "item da lista",
            "lista de servico",
            "subitem",
            "subitens",
        )
    )


def _has_document_context(normalized_line: str, section: str, value: str) -> bool:
    if "cnpj" in normalized_line or "cpf" in normalized_line:
        return True
    if section not in {"provider", "recipient"} or value.isdigit():
        return False
    return any(term in normalized_line for term in ("insc", "inscricao", "municipal", "razao", "social"))


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _bbox_x(bounding_box: tuple[float, float, float, float] | None) -> float:
    return bounding_box[0] if bounding_box is not None else 0.0


def _bbox_y(bounding_box: tuple[float, float, float, float] | None) -> float:
    return bounding_box[1] if bounding_box is not None else 0.0


def _merge_bounding_boxes(
    boxes: list[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    left = min(box[0] for box in present)
    top = min(box[1] for box in present)
    right = max(box[0] + box[2] for box in present)
    bottom = max(box[1] + box[3] for box in present)
    return (left, top, right - left, bottom - top)
