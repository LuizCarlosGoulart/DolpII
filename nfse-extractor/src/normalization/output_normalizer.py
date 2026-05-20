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
    ("razao", "social"): {"provider": "provider_name", "recipient": "recipient_name"},
    ("razao", "social", "nome"): {"provider": "provider_name", "recipient": "recipient_name"},
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
_PARTY_NAME_FIELDS = {"provider_name", "recipient_name"}
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
    "other_retentions_amount",
    "deductions_amount",
    "net_amount",
}
_FINANCIAL_SINGLETON_FIELDS = _MONEY_FIELDS | {"iss_rate"}
_PHONE_FIELDS = {"provider_phone", "recipient_phone"}
_UF_FIELDS = {"provider_uf", "recipient_uf"}
_REVIEW_LOW_CONFIDENCE_THRESHOLDS = {
    "nfse_number": 0.85,
    "nfse_series": 0.80,
    "issue_date": 0.85,
    "verification_code": 0.85,
    "provider_document": 0.88,
    "recipient_document": 0.88,
    "provider_email": 0.75,
    "recipient_email": 0.75,
    "provider_name": 0.75,
    "recipient_name": 0.75,
    "provider_address": 0.75,
    "recipient_address": 0.75,
    "provider_phone": 0.75,
    "recipient_phone": 0.75,
    "provider_uf": 0.75,
    "recipient_uf": 0.75,
    "service_code": 0.85,
    "service_description": 0.75,
}
_FINANCIAL_REVIEW_THRESHOLD = 0.85
_DEFAULT_REVIEW_THRESHOLD = 0.70
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

_SERVICE_NEARBY_FIELDS = {"operation_nature", "service_code"}
_SERVICE_TEXT_FIELDS = {"operation_nature", "service_city", "service_description"}
# Fields that should only be extracted from lines the section classifier places
# in the "service" or "values" section.  Matching these labels on header,
# provider or recipient lines produces systematic false positives because the
# same vocabulary appears in addresses, document headers, and party blocks.
_SERVICE_SECTION_ONLY_LABELS = {"operation_nature", "service_city", "service_code"}

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


@dataclass(frozen=True)
class _FinancialColumn:
    field_name: str | None
    label_text: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _FinancialValue:
    value: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _FinancialMapping:
    value: str
    column: _FinancialColumn
    value_lines: tuple[_Line, ...]


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

        candidates.extend(self._candidates_from_financial_tables(document, lines))
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
            # Service-scoped fields must come from a line already classified in
            # the service or values section.  When these labels appear on
            # header/provider/recipient lines they almost always produce false
            # positives (city names from addresses, nature text from document
            # titles, etc.).  This guard is layout-agnostic: it relies only on
            # the section classifier's output, not on document-specific patterns.
            if match.field_name in _SERVICE_SECTION_ONLY_LABELS and line.section not in {"service", "values"}:
                continue
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
                table_value = self._extract_table_value(match, line, line_index, lines)
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

            if value and match.field_name == "net_amount":
                value = _correct_merged_net_amount(line.text, value)

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
                    section_override=_party_section_from_field(match.field_name),
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
            label_section = _party_section_for_label_position(line.section, token_values, position, label_tokens)
            end_token_index = position + len(label_tokens) - 1
            end_element_index = flattened[end_token_index][1]
            matches.append(
                _LabelMatch(
                    field_name=self._scope_field(field_name, label_tokens, label_section),
                    label_text=label_text,
                    end_element_index=end_element_index,
                    token_count=len(label_tokens),
                )
            )

        for label_tokens, scoped_fields in _SECTION_SCOPED_LABELS.items():
            position = _find_subsequence(token_values, label_tokens)
            if position is None:
                continue
            label_section = _party_section_for_label_position(line.section, token_values, position, label_tokens)
            scoped_field = scoped_fields.get(label_section)
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
            document_section = _party_section_for_document_match(line.text, line.section, match)
            field_name = "recipient_document" if document_section == "recipient" else "provider_document"
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
                    section_override=_party_section_from_field(field_name),
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
        if field_name in _PARTY_NAME_FIELDS:
            return _clean_party_name_value(value)
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
        """Extract a field value embedded in the same text fragment as its label.

        This is the fallback used when ``_value_elements_after_label`` finds no
        elements after the label — which happens when a VLM (e.g. Dolphin) returns
        whole label:value pairs as a single OCR element.  The method is
        intentionally conservative: it only tries extraction for field types whose
        typed extractor is robust enough to work correctly on mixed label+value text.
        """
        # ── UF: exact contextual match ────────────────────────────────────────
        if field_name in _UF_FIELDS:
            match = re.search(r"\bUF\s*:?\s*([A-Z]{2})\b", line_text, flags=re.IGNORECASE)
            return match.group(1).upper() if match else None

        # ── Party names: take everything after the first ":" separator ────────
        # Covers "Nome/Razão Social: NCR BRASIL LTDA" and the Dolphin OCR variant
        # "Name/Razão Social: NCR BRASIL LTDA" (English "Name" misread for "Nome").
        # Guard: only trigger when the text *before* the colon looks like a name
        # label keyword — this prevents "Inscrição Municipal: 474043" from being
        # mis-classified as a party name.
        if field_name in _PARTY_NAME_FIELDS:
            colon_pos = line_text.find(":")
            if colon_pos == -1:
                return None
            prefix_norm = _normalize(line_text[:colon_pos])
            name_label_keywords = ("nome", "razao social", "razao", "social", "denominacao", "name")
            if not any(kw in prefix_norm for kw in name_label_keywords):
                return None
            suffix = line_text[colon_pos + 1:].strip()
            return _clean_party_name_value(suffix)

        # ── Identifier / header fields: typed extraction on the full line ─────
        # These extractors use precise regex patterns or strict validators that
        # correctly isolate the value even in mixed label+value text, so calling
        # them on the complete line text is safe.
        if field_name in {"nfse_number", "nfse_series"}:
            return self._extract_typed_value(field_name, line_text)

        return None

    def _extract_table_value(
        self,
        match: _LabelMatch,
        line: _Line,
        line_index: int,
        lines: list[_Line],
    ) -> str | None:
        if match.field_name not in _MONEY_FIELDS and match.field_name != "iss_rate":
            return None

        table_mapping = _map_financial_table(line, line_index, lines)
        if table_mapping:
            mapped = table_mapping.get(match.field_name)
            return mapped.value if mapped is not None else None

        if line_index + 1 >= len(lines):
            return None
        next_line = lines[line_index + 1]
        matches = self._find_label_matches(line)
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

    def _candidates_from_financial_tables(
        self,
        document: Document,
        lines: list[_Line],
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for line_index, line in enumerate(lines):
            mapping = _map_financial_table(line, line_index, lines)
            for field_name, mapped in mapping.items():
                if field_name not in _MONEY_FIELDS and field_name != "iss_rate":
                    continue
                value_elements = [element for value_line in mapped.value_lines for element in value_line.elements]
                if not value_elements:
                    continue
                candidates.append(
                    self._build_candidate(
                        document=document,
                        field_name=field_name,
                        value=mapped.value,
                        source_elements=value_elements,
                        label_text=mapped.column.label_text,
                        line=line,
                        line_index=line_index,
                        value_source="financial_table",
                        confidence_boost=0.10,
                    )
                )
        return candidates

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
            or field_name in _PARTY_NAME_FIELDS
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
                    section_override=_party_section_from_field(field_name),
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
                    section_override=_party_section_from_field(field_name),
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
        section_override: str | None = None,
    ) -> FieldCandidate:
        confidence_values = [element.confidence for element in source_elements if element.confidence is not None]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
        confidence = max(0.0, min(confidence + confidence_boost, 1.0))
        if value_source == "ocr_corrected_email":
            confidence = min(confidence * 0.75, 0.7)
        value_hash = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
        candidate_id = f"{document.document_id}:{self.normalizer_name}:{field_name}:{line_index}:{len(source_elements)}:{value_hash}"
        effective_section = section_override or line.section

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
                "section_name": effective_section,
                "raw_section_name": line.section,
                "section_override_applied": effective_section != line.section,
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
        field_candidate_counts = _field_candidate_counts(candidates)
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

        singleton_fields = {"issue_date", "nfse_number", "nfse_series", "verification_code"} | _FINANCIAL_SINGLETON_FIELDS
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
        results = [
            _with_review_metadata(candidate, field_candidate_counts.get(candidate.field_name, 1))
            for candidate in results
        ]
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
        if any(term in context for term in ("numero da nota", "numero da nfs", "numero da nf", "nota no")):
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
    elif field_name in _FINANCIAL_SINGLETON_FIELDS:
        value_source = str(candidate.metadata.get("value_source", ""))
        if value_source == "financial_table":
            score += 0.35
        elif value_source == "next_line_table":
            score += 0.15
        if any(term in context for term in ("credito", "trib federais", "total trib")):
            score -= 0.45
        if field_name == "gross_amount" and any(term in label for term in ("valor bruto", "valor total", "valor servico")):
            score += 0.25
        elif field_name == "taxable_amount" and "base" in label:
            score += 0.25
        elif field_name == "iss_rate":
            if "aliquota" in label:
                score += 0.25
            if _is_zero_money(value):
                score -= 0.85
            if value.endswith("%"):
                score += 0.10
        elif field_name == "iss_amount" and ("valor iss" in label or "issqn devido" in label):
            score += 0.25
        elif field_name == "net_amount":
            if "valor liquido" in label:
                score += 0.25
            if "credito" in context:
                score -= 0.55
        elif field_name in {
            "deductions_amount",
            "unconditional_discount",
            "conditional_discount",
            "pis_withheld_amount",
            "cofins_withheld_amount",
            "inss_withheld_amount",
            "ir_withheld_amount",
            "csll_withheld_amount",
            "other_retentions_amount",
        } and _is_zero_money(value):
            score += 0.08

    line_index = float(candidate.metadata.get("line_index", 0) or 0)
    return (score, confidence, -line_index)


def _field_candidate_counts(candidates: list[FieldCandidate]) -> dict[str, int]:
    values_by_field: dict[str, set[str]] = {}
    for candidate in candidates:
        values_by_field.setdefault(candidate.field_name, set()).add(_normalize(candidate.value))
    return {field_name: len(values) for field_name, values in values_by_field.items()}


def _with_review_metadata(candidate: FieldCandidate, field_candidate_count: int) -> FieldCandidate:
    reasons = _candidate_review_reasons(candidate, field_candidate_count)
    metadata = dict(candidate.metadata)
    metadata["review_status"] = "needs_review" if reasons else "accepted"
    metadata["review_reasons"] = reasons
    metadata["manual_review_required"] = bool(reasons)
    metadata["review_confidence_threshold"] = _review_confidence_threshold(candidate.field_name)
    if field_candidate_count > 1:
        metadata["candidate_conflict_count"] = field_candidate_count
    else:
        metadata.pop("candidate_conflict_count", None)
    return candidate.model_copy(update={"metadata": metadata})


def _candidate_review_reasons(candidate: FieldCandidate, field_candidate_count: int) -> list[str]:
    field_name = candidate.field_name
    value = str(candidate.value)
    metadata = candidate.metadata
    context = _normalize(metadata.get("line_text", ""))
    value_source = str(metadata.get("value_source", ""))
    reasons: list[str] = []

    confidence = candidate.confidence
    threshold = _review_confidence_threshold(field_name)
    if confidence is not None and confidence < threshold:
        reasons.append("low_confidence")

    section_confidence = metadata.get("section_confidence")
    if isinstance(section_confidence, (float, int)) and section_confidence < 0.65:
        reasons.append("low_section_confidence")

    if metadata.get("section_override_applied") is True:
        reasons.append("section_inferred_from_field")

    if field_candidate_count > 1:
        reasons.append("multiple_candidates")

    if value_source in {"nearby_line", "nearby_lines"}:
        reasons.append("nearby_value")
    elif value_source == "next_line_table":
        reasons.append("table_value")
    elif value_source == "ocr_corrected_email":
        reasons.append("email_ocr_corrected")

    if field_name in _EMAIL_FIELDS and metadata.get("ocr_correction_applied") is True:
        reasons.append("email_ocr_separator_uncertain")

    if field_name == "service_code" and re.fullmatch(r"\d{3,4}", value):
        reasons.append("service_code_format_uncertain")

    if field_name == "iss_rate" and not value.endswith("%"):
        reasons.append("rate_without_percent")

    if field_name in _FINANCIAL_SINGLETON_FIELDS and any(
        term in context for term in ("credito", "trib federais", "total trib", "linha digitavel")
    ):
        reasons.append("merged_financial_context")

    return list(dict.fromkeys(reasons))


def _review_confidence_threshold(field_name: str) -> float:
    if field_name in _FINANCIAL_SINGLETON_FIELDS:
        return _FINANCIAL_REVIEW_THRESHOLD
    return _REVIEW_LOW_CONFIDENCE_THRESHOLDS.get(field_name, _DEFAULT_REVIEW_THRESHOLD)


_FINANCIAL_LABEL_SPECS: tuple[tuple[str | None, str, str], ...] = (
    ("gross_amount", "valor total dos servicos", "money"),
    ("gross_amount", "valor total do servico", "money"),
    ("gross_amount", "valor total da nota", "money"),
    ("gross_amount", "valor bruto da nota", "money"),
    ("gross_amount", "valor total", "money"),
    ("gross_amount", "prestado valor", "money"),
    ("deductions_amount", "valor total das deducoes", "money"),
    ("unconditional_discount", "desconto incondicionado", "money"),
    ("unconditional_discount", "desconto incondicional", "money"),
    ("unconditional_discount", "desc incondicionado", "money"),
    ("unconditional_discount", "desc incondic", "money"),
    ("conditional_discount", "desconto condicionado", "money"),
    ("conditional_discount", "desconto condicional", "money"),
    ("conditional_discount", "desc condicionado", "money"),
    ("other_retentions_amount", "outras retencoes", "money"),
    (None, "valor retencoes", "money"),
    (None, "total trib federais", "money"),
    (None, "total trib", "money"),
    (None, "valor do credito", "money"),
    (None, "desconto", "money"),
    ("taxable_amount", "base de calculo do issqn", "money"),
    ("taxable_amount", "base de calculo issqn", "money"),
    ("taxable_amount", "base calculo iss", "money"),
    ("taxable_amount", "base de calculo", "money"),
    ("taxable_amount", "base calculo", "money"),
    ("iss_rate", "aliquota do issqn", "rate"),
    ("iss_rate", "aliquota issqn", "rate"),
    ("iss_rate", "aliquota iss", "rate"),
    ("iss_rate", "aliquota", "rate"),
    ("iss_rate", "alig", "rate"),
    ("iss_rate", "aliq", "rate"),
    ("iss_amount", "issqn devido", "money"),
    ("iss_amount", "valor do iss", "money"),
    ("iss_amount", "valor iss", "money"),
    ("net_amount", "valor liquido da nota", "money"),
    ("net_amount", "valor liquido", "money"),
    ("pis_withheld_amount", "pis pasep", "money"),
    ("pis_withheld_amount", "pis", "money"),
    ("cofins_withheld_amount", "cofins", "money"),
    ("inss_withheld_amount", "inss", "money"),
    ("ir_withheld_amount", "irrf", "money"),
    ("ir_withheld_amount", "ir", "money"),
    ("csll_withheld_amount", "csll", "money"),
    ("deductions_amount", "valor deducao", "money"),
    ("deductions_amount", "deducoes", "money"),
    ("deductions_amount", "deducao", "money"),
    ("gross_amount", "valor servico", "money"),
)

_FINANCIAL_LABEL_SPECS = tuple(sorted(_FINANCIAL_LABEL_SPECS, key=lambda item: len(item[1]), reverse=True))
_FINANCIAL_PLACEHOLDER_FIELDS = {None}
_FINANCIAL_VALUE_RE = re.compile(
    r"(?:R\$\s*-?\s*)?\d{1,3}(?:\.\d{3})*,\d{2,4}\s*%?"
    r"|(?:R\$\s*-?\s*)?\d+,\d{2,4}\s*%?"
    r"|\d{1,2}\s*%"
)


def _map_financial_table(
    line: _Line,
    line_index: int,
    lines: list[_Line],
) -> dict[str, _FinancialMapping]:
    if line.section != "values":
        return {}
    normalized_header = _normalize(line.text)
    if _looks_like_financial_noise_header(normalized_header):
        return {}
    columns = _financial_columns_from_line(line.text)
    if not _looks_like_financial_table(columns):
        return {}
    value_lines = _financial_value_lines_after_header(line_index, lines)
    if not value_lines:
        return {}
    values = _financial_values_from_text(" ".join(value_line.text for value_line in value_lines))
    if not values:
        return {}
    return _align_financial_columns(columns, values, value_lines)


def _financial_columns_from_line(text: str) -> list[_FinancialColumn]:
    normalized = _normalize(text)
    columns: list[_FinancialColumn] = []
    occupied: list[tuple[int, int]] = []
    for field_name, label_text, kind in _FINANCIAL_LABEL_SPECS:
        pattern = r"(?:^|\s)" + re.escape(label_text) + r"(?=\s|$)"
        for match in re.finditer(pattern, normalized):
            start = match.start()
            end = match.end()
            if any(not (end <= used_start or start >= used_end) for used_start, used_end in occupied):
                continue
            columns.append(_FinancialColumn(field_name, label_text, kind, start, end))
            occupied.append((start, end))
    columns.sort(key=lambda item: item.start)
    return columns


def _looks_like_financial_noise_header(normalized_header: str) -> bool:
    return any(
        phrase in normalized_header
        for phrase in (
            "disp ret",
            "igual ou menor",
            "lei ",
            "lei complementar",
            "pgto",
            "transparencia",
            "valor aproximado",
        )
    )


def _looks_like_financial_table(columns: list[_FinancialColumn]) -> bool:
    value_columns = [column for column in columns if column.field_name not in _FINANCIAL_PLACEHOLDER_FIELDS]
    if len(value_columns) >= 2:
        return True
    return any(column.field_name == "iss_rate" for column in columns) and len(columns) >= 2


def _financial_value_lines_after_header(line_index: int, lines: list[_Line]) -> tuple[_Line, ...]:
    selected: list[_Line] = []
    for nearby_line in lines[line_index + 1 : line_index + 4]:
        if nearby_line.section != "values":
            break
        normalized = _normalize(nearby_line.text)
        if normalized in _SECTION_ONLY_VALUES:
            continue
        if _financial_columns_from_line(nearby_line.text) and selected:
            break
        if _FINANCIAL_VALUE_RE.search(_merge_split_money_fragments(nearby_line.text)) or re.fullmatch(r"\s*,\s*\d{2}\s*", nearby_line.text):
            selected.append(nearby_line)
            if len(selected) >= 2 and not re.fullmatch(r"\s*,\s*\d{2}\s*", nearby_line.text):
                break
            continue
        if selected:
            break
    return tuple(selected)


def _financial_values_from_text(text: str) -> list[_FinancialValue]:
    text = _merge_split_money_fragments(text)
    values: list[_FinancialValue] = []
    for match in _FINANCIAL_VALUE_RE.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        value = re.sub(r"\s*%\s*$", "%", value)
        kind = "rate" if value.endswith("%") else "money"
        values.append(_FinancialValue(value, kind, match.start(), match.end()))
    return values


def _merge_split_money_fragments(text: str) -> str:
    text = re.sub(r"\b(\d+)\s+,\s*(\d{2,4})\b", r"\1,\2", text)
    text = re.sub(r"(R\$)\s+(-)\s+", r"\1 \2", text)
    return text


def _align_financial_columns(
    columns: list[_FinancialColumn],
    values: list[_FinancialValue],
    value_lines: tuple[_Line, ...],
) -> dict[str, _FinancialMapping]:
    mapping: dict[str, _FinancialMapping] = {}
    rate_column_index = next((index for index, column in enumerate(columns) if column.field_name == "iss_rate"), None)
    if rate_column_index is not None:
        rate_value_index = _find_rate_value_index(columns, values, rate_column_index)
        if rate_value_index is not None:
            _map_financial_side(columns[:rate_column_index], values[:rate_value_index], value_lines, mapping)
            rate_value = values[rate_value_index]
            if _is_acceptable_financial_value("iss_rate", rate_value.value, columns[rate_column_index]):
                mapping["iss_rate"] = _FinancialMapping(rate_value.value, columns[rate_column_index], value_lines)
            right_values = values[rate_value_index + 1 :]
            _map_financial_side(columns[rate_column_index + 1 :], right_values, value_lines, mapping)
            if "iss_amount" not in mapping:
                iss_value = next((value for value in right_values if value.kind == "money"), None)
                if iss_value is not None:
                    mapping["iss_amount"] = _FinancialMapping(
                        iss_value.value,
                        _FinancialColumn(
                            "iss_amount",
                            "valor iss",
                            "money",
                            columns[rate_column_index].end,
                            columns[rate_column_index].end,
                        ),
                        value_lines,
                    )
            return mapping

    _map_financial_side(columns, values, value_lines, mapping)
    if "iss_amount" not in mapping and any(column.field_name == "taxable_amount" for column in columns):
        consumable_columns = [column for column in columns if _financial_column_consumes_value(column)]
        money_values = [value for value in values if value.kind == "money"]
        if len(money_values) > len(consumable_columns):
            iss_value = money_values[-1]
            mapping["iss_amount"] = _FinancialMapping(
                iss_value.value,
                _FinancialColumn("iss_amount", "valor iss", "money", columns[-1].end, columns[-1].end),
                value_lines,
            )
    return mapping


def _find_rate_value_index(
    columns: list[_FinancialColumn],
    values: list[_FinancialValue],
    rate_column_index: int,
) -> int | None:
    for index, value in enumerate(values):
        if value.kind == "rate":
            return index
    consumable_before_rate = len([column for column in columns[:rate_column_index] if _financial_column_consumes_value(column)])
    if consumable_before_rate < len(values):
        candidate = values[consumable_before_rate]
        if _looks_like_unmarked_rate(candidate.value):
            return consumable_before_rate
    return None


def _map_financial_side(
    columns: list[_FinancialColumn],
    values: list[_FinancialValue],
    value_lines: tuple[_Line, ...],
    mapping: dict[str, _FinancialMapping],
) -> None:
    consumable_columns = [column for column in columns if _financial_column_consumes_value(column)]
    money_values = [value for value in values if value.kind == "money"]
    if not consumable_columns or not money_values:
        return

    if len(money_values) >= len(consumable_columns):
        pairs = zip(consumable_columns, money_values)
    else:
        pairs = _align_sparse_financial_side(consumable_columns, money_values)

    for column, value in pairs:
        if column.field_name is None:
            continue
        if column.field_name in mapping:
            continue
        if not _is_acceptable_financial_value(column.field_name, value.value, column):
            continue
        mapping[column.field_name] = _FinancialMapping(value.value, column, value_lines)


def _align_sparse_financial_side(
    columns: list[_FinancialColumn],
    values: list[_FinancialValue],
) -> list[tuple[_FinancialColumn, _FinancialValue]]:
    if not columns or not values:
        return []
    if len(values) == 1:
        taxable_column = next((column for column in columns if column.field_name == "taxable_amount"), None)
        if taxable_column is not None and len(columns) <= 2:
            return [(taxable_column, values[-1])]
        return []
    pairs: list[tuple[_FinancialColumn, _FinancialValue]] = [(columns[-1], values[-1])]
    first_column = columns[0]
    first_value = values[0]
    if first_column.field_name in {"deductions_amount", "other_retentions_amount", None} and _is_zero_money(first_value.value):
        pairs.insert(0, (first_column, first_value))
    return pairs


def _financial_column_consumes_value(column: _FinancialColumn) -> bool:
    return column.kind in {"money", "rate"}


def _looks_like_unmarked_rate(value: str) -> bool:
    if value.endswith("%"):
        return True
    if "R$" in value.upper():
        return False
    if _is_zero_money(value):
        return False
    number = _decimal_from_br_number(value)
    return number is not None and 0 < number <= 100


def _is_acceptable_financial_value(field_name: str, value: str, column: _FinancialColumn) -> bool:
    if field_name == "iss_rate":
        return value.endswith("%") or _looks_like_unmarked_rate(value)
    if field_name == "deductions_amount" and not _is_zero_money(value):
        return False
    if field_name in _MONEY_FIELDS:
        return not value.endswith("%")
    return True


def _correct_merged_net_amount(line_text: str, value: str) -> str:
    label_match = re.search(r"\bvalor\s+l[ií]quido\b", line_text, flags=re.IGNORECASE)
    if label_match is None:
        return value

    normalized_suffix = _normalize(line_text[label_match.end() :])
    if not normalized_suffix.startswith("do credito") and not _is_zero_money(value):
        return value

    prefix_values = [
        match.group(0).strip()
        for match in _money_matches(line_text[: label_match.start()])
        if not _is_zero_money(match.group(0))
    ]
    return prefix_values[-1] if prefix_values else value


def _decimal_from_br_number(value: str) -> float | None:
    normalized = re.sub(r"[^\d,.-]", "", value).replace(".", "").replace(",", ".")
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(value))


def _party_section_for_label_position(
    fallback_section: str,
    token_values: list[str],
    label_position: int,
    label_tokens: tuple[str, ...],
) -> str:
    if fallback_section not in {"provider", "recipient"}:
        return fallback_section

    provider_positions = [index for index, token in enumerate(token_values) if token == "prestador"]
    recipient_positions = [index for index, token in enumerate(token_values) if token == "tomador"]
    last_provider = max((index for index in provider_positions if index <= label_position), default=-1)
    last_recipient = max((index for index in recipient_positions if index <= label_position), default=-1)
    if last_provider > last_recipient:
        return "provider"
    if last_recipient > last_provider:
        return "recipient"

    first_recipient = min(recipient_positions, default=None)
    if (
        fallback_section == "recipient"
        and first_recipient is not None
        and label_position < first_recipient
        and label_tokens in _SECTION_SCOPED_LABELS
    ):
        return "provider"

    return fallback_section


def _party_section_from_field(field_name: str) -> str | None:
    if field_name.startswith("provider_"):
        return "provider"
    if field_name.startswith("recipient_"):
        return "recipient"
    return None


def _party_section_for_document_match(
    line_text: str,
    fallback_section: str,
    match: re.Match[str],
) -> str:
    if fallback_section not in {"provider", "recipient"}:
        return fallback_section

    prefix = _normalize(line_text[: match.start()])
    suffix = _normalize(line_text[match.end() :])
    if _contains_phrase(prefix, "tomador"):
        return "recipient"
    if _contains_phrase(prefix, "prestador"):
        return "provider"
    if (
        fallback_section == "recipient"
        and _contains_phrase(suffix, "tomador")
        and _looks_like_party_fragment_before_boundary(prefix)
    ):
        return "provider"
    return fallback_section


def _looks_like_party_fragment_before_boundary(normalized_prefix: str) -> bool:
    has_document_cue = "cnpj" in normalized_prefix or "cpf" in normalized_prefix
    has_party_detail = any(
        phrase in normalized_prefix
        for phrase in (
            "bairro",
            "e mail",
            "email",
            "endereco",
            "fantasia",
            "municipio",
            "nome razao",
            "pais",
            "razao social",
            "social",
            "telefone",
        )
    )
    return has_document_cue and has_party_detail


def _contains_phrase(normalized: str, phrase: str) -> bool:
    pattern = r"(?:^|\s)" + re.escape(phrase) + r"(?:\s|$)"
    return re.search(pattern, normalized) is not None


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


_NFSE_SERIES_STOP_TOKENS = {
    # Common Portuguese articles, prepositions, conjunctions and abbreviations
    # that the old regex was incorrectly accepting as series identifiers.
    "A", "AS", "DA", "DAS", "DE", "DO", "DOS", "E", "EM", "NA", "NAS",
    "NO", "NOS", "O", "OS", "OU", "SE", "UF", "UM", "UMA",
    # Month names (abbreviated)
    "JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT",
    "NOV", "DEZ",
    # Day-of-week abbreviations
    "SEG", "TER", "QUA", "QUI", "SEX", "SAB", "DOM",
    # Fiscal document abbreviations that should not be mistaken for series
    "NFE", "NFSE", "NFS", "RPS", "NF",
}


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
    # Filter out common Portuguese words that are not valid series identifiers.
    candidates = [c for c in candidates if c not in _NFSE_SERIES_STOP_TOKENS]
    if not candidates:
        return None
    return candidates[-1].strip(" :-,.()")


def _clean_party_name_value(value: str) -> str | None:
    value = re.split(
        r"\b(?:cnpj|cpf|endereco|e\s*mail|email|fone|insc\.?|inscricao|municipio|telefone|uf)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = _PATTERN_FIELD_HINTS["document_id"].sub("", value)
    value = re.sub(r"\b\d{11,14}\b$", "", value)
    value = re.sub(r"\s+", " ", value).strip(" :-,\t\r\n")
    if len(_tokens(value)) < 2:
        return None
    return value


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
    # Reject purely alphabetic codes (e.g. "QUARTA-FEIRA", "SEGUNDA") — valid
    # verification codes always contain at least one digit.
    if not any(character.isdigit() for character in normalized_code):
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
        stop_pattern = r"\b(?:codigo|c[óo]digo|data|emiss[aã]o|local\s+d[aeo]|valor|base\s+de\s+c[aá]lculo)\b"
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
