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

_SECTION_KEYWORDS = {
    "provider": ("prestador",),
    "recipient": ("tomador",),
    "service": ("servico", "servicos", "discriminacao"),
    "values": ("valor", "valores", "retencoes", "imposto", "iss", "base de calculo"),
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
    "document_id": re.compile(r"\b(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14}|\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b"),
    "email": re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b"),
    "date": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "service_code": re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b"),
    "money": re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b|\b\d+(?:\.\d{2})\b"),
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


@dataclass(frozen=True)
class _LabelMatch:
    field_name: str
    label_text: str
    end_element_index: int


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
            candidates.extend(self._candidates_from_labels(document, line, line_index, lines))
            candidates.extend(self._candidates_from_patterns(document, line, line_index))

        return self._deduplicate_candidates(candidates)

    def _build_label_aliases(self) -> list[tuple[tuple[str, ...], str, str]]:
        label_aliases: list[tuple[tuple[str, ...], str, str]] = []
        for field in self.field_dictionary.fields:
            aliases = {field.internal_name.replace("_", " "), *field.aliases, *self.aliases.get(field.internal_name, [])}
            for alias in aliases:
                tokens = tuple(_tokens(alias))
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

        lines: list[_Line] = []
        current_section = "header"
        for page_number, _top, _left, line_elements in raw_lines:
            line_text = " ".join(element.text for element in line_elements).strip()
            current_section = self._next_section(current_section, line_text)
            first = line_elements[0]
            lines.append(
                _Line(
                    page_number=page_number,
                    block_num=_optional_int(first.metadata.get("block_num")),
                    line_num=_optional_int(first.metadata.get("line_num")),
                    text=line_text,
                    elements=line_elements,
                    bounding_box=_merge_bounding_boxes([element.bounding_box for element in line_elements]),
                    section=current_section,
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

    def _next_section(self, current_section: str, text: str) -> str:
        normalized = _normalize(text)
        for section, keywords in _SECTION_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                return section
        return current_section

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
            if not value_elements and line_index + 1 < len(lines):
                value_elements = lines[line_index + 1].elements
                value_source = "next_line"

            value = self._clean_value(" ".join(element.text for element in value_elements))
            if not value:
                continue

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
                )
            )

        deduped: dict[str, _LabelMatch] = {}
        for match in matches:
            current = deduped.get(match.field_name)
            if current is None or match.end_element_index > current.end_element_index:
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
            field_name = "recipient_document" if line.section == "recipient" else "provider_document"
            candidates.append(
                self._build_candidate(
                    document=document,
                    field_name=field_name,
                    value=match.group(0),
                    source_elements=line.elements,
                    label_text="document pattern",
                    line=line,
                    line_index=line_index,
                    value_source="regex",
                    confidence_boost=0.03,
                )
            )

        if "emissao" in normalized_line or "data" in normalized_line:
            candidates.extend(self._regex_candidates(document, line, line_index, "issue_date", "date"))

        if line.section in {"provider", "recipient"}:
            field_name = "recipient_email" if line.section == "recipient" else "provider_email"
            candidates.extend(self._regex_candidates(document, line, line_index, field_name, "email"))

        if line.section == "service":
            candidates.extend(self._regex_candidates(document, line, line_index, "service_code", "service_code"))

        return candidates

    def _regex_candidates(
        self,
        document: Document,
        line: _Line,
        line_index: int,
        field_name: str,
        pattern_name: str,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
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
                "page_number": line.page_number,
                "line_text": line.text,
                "line_index": line_index,
                "value_source": value_source,
                "same_block_as_label": value_source == "same_line",
                "block_num": line.block_num,
                "label_block_num": line.block_num,
                "bounding_box": line.bounding_box,
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
        return sorted(deduped.values(), key=lambda item: (item.field_name, str(item.metadata.get("line_index", ""))))

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        with (self.config_dir / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}


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
