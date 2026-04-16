"""Validation layer for resolved canonical NFS-e fields."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

from src.core import Document, ResolvedField, ValidationIssue, Validator, load_field_dictionary


_UF_CODES = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}

_RETENTION_FIELDS = (
    "pis_withheld_amount",
    "cofins_withheld_amount",
    "inss_withheld_amount",
    "ir_withheld_amount",
    "csll_withheld_amount",
    "iss_withheld_amount",
)


class ConfigDrivenValidator(Validator):
    """Validate resolved fields using the canonical field dictionary and small built-in rules."""

    def __init__(self, *, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
        self.field_dictionary = load_field_dictionary(self.config_dir / "field_dictionary.yaml")
        self.field_map = self.field_dictionary.by_internal_name()

    def validate(
        self,
        document: Document,
        fields: list[ResolvedField],
    ) -> list[ValidationIssue]:
        field_index = {field.field_name: field for field in fields}
        issues: list[ValidationIssue] = []

        issues.extend(self._validate_required_fields(field_index))
        for field in fields:
            issues.extend(self._validate_single_field(document, field))
        issues.extend(self._validate_relationships(field_index))

        return issues

    def _validate_required_fields(self, field_index: dict[str, ResolvedField]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for field_name, definition in self.field_map.items():
            if not definition.required:
                continue
            resolved = field_index.get(field_name)
            if resolved is None:
                issues.append(
                    ValidationIssue(
                        code="required_field_missing",
                        message=f"Required field '{field_name}' is missing.",
                        severity="error",
                        field_name=field_name,
                    )
                )
        return issues

    def _validate_single_field(self, document: Document, field: ResolvedField) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        definition = self.field_map.get(field.field_name)

        if field.status == "conflict":
            issues.append(
                ValidationIssue(
                    code="field_conflict",
                    message=f"Field '{field.field_name}' has ambiguous candidate values.",
                    severity="error" if definition and definition.required else "warning",
                    field_name=field.field_name,
                )
            )
            return issues

        if field.status == "missing":
            issues.append(
                ValidationIssue(
                    code="required_field_unresolved" if definition and definition.required else "field_missing",
                    message=f"Required field '{field.field_name}' is not safely resolved."
                    if definition and definition.required
                    else f"Field '{field.field_name}' is marked as missing.",
                    severity="error" if definition and definition.required else "warning",
                    field_name=field.field_name,
                    metadata={"status": field.status} if definition and definition.required else {},
                )
            )
            return issues

        if definition and definition.required and not self._has_value(field.value):
            issues.append(
                ValidationIssue(
                    code="required_field_unresolved",
                    message=f"Required field '{field.field_name}' is not safely resolved.",
                    severity="error",
                    field_name=field.field_name,
                    metadata={"status": field.status},
                )
            )
            return issues

        if not self._has_value(field.value):
            return issues

        value = str(field.value).strip()
        type_name = definition.type if definition is not None else None

        if type_name == "document_id":
            issues.extend(self._validate_document_id(field.field_name, value, required=bool(definition and definition.required)))
        if type_name == "date":
            issues.extend(self._validate_date(field.field_name, value, document))
        if type_name == "decimal":
            issues.extend(self._validate_decimal(field.field_name, value, required=bool(definition and definition.required)))
        if type_name == "percentage":
            issues.extend(self._validate_percentage(field.field_name, value))

        if field.field_name == "service_code":
            issues.extend(self._validate_service_code(value))
        if field.field_name.endswith("_email"):
            issues.extend(self._validate_email(field.field_name, value))
        if field.field_name.endswith("_phone"):
            issues.extend(self._validate_phone(field.field_name, value))
        if field.field_name.endswith("_uf"):
            issues.extend(self._validate_uf(field.field_name, value))

        return issues

    def _validate_relationships(self, field_index: dict[str, ResolvedField]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        gross = self._reliable_decimal(field_index.get("gross_amount"))
        unconditional_discount = self._reliable_decimal(field_index.get("unconditional_discount"), default=Decimal("0"))
        conditional_discount = self._reliable_decimal(field_index.get("conditional_discount"), default=Decimal("0"))
        retention_values = [self._reliable_decimal(field_index.get(field_name), default=Decimal("0")) for field_name in _RETENTION_FIELDS]
        net = self._reliable_decimal(field_index.get("net_amount"))

        if None not in (gross, unconditional_discount, conditional_discount, net) and all(value is not None for value in retention_values):
            expected_net = gross - unconditional_discount - conditional_discount - sum(retention_values)
            if abs(expected_net - net) > Decimal("0.02"):
                issues.append(
                    ValidationIssue(
                        code="net_amount_inconsistent",
                        message="Net amount is inconsistent with gross amount, discounts, and retentions.",
                        severity="error",
                        field_name="net_amount",
                        metadata={
                            "gross_amount": str(gross),
                            "expected_net_amount": str(expected_net),
                            "net_amount": str(net),
                        },
                    )
                )

        taxable_amount = self._reliable_decimal(field_index.get("taxable_amount"))
        iss_amount = self._reliable_decimal(field_index.get("iss_amount"))
        iss_rate = self._reliable_percentage(field_index.get("iss_rate"))
        if taxable_amount is not None and iss_amount is not None and iss_rate is not None:
            expected_iss = (taxable_amount * iss_rate / Decimal("100")).quantize(Decimal("0.01"))
            if abs(expected_iss - iss_amount) > Decimal("0.02"):
                issues.append(
                    ValidationIssue(
                        code="iss_amount_inconsistent",
                        message="ISS amount is inconsistent with taxable amount and ISS rate.",
                        severity="warning",
                        field_name="iss_amount",
                        metadata={
                            "taxable_amount": str(taxable_amount),
                            "iss_rate": str(iss_rate),
                            "expected_iss_amount": str(expected_iss),
                            "iss_amount": str(iss_amount),
                        },
                    )
                )

        provider_document = self._resolved_text(field_index.get("provider_document"))
        recipient_document = self._resolved_text(field_index.get("recipient_document"))
        if provider_document and recipient_document and self._digits_only(provider_document) == self._digits_only(recipient_document):
            issues.append(
                ValidationIssue(
                    code="provider_recipient_document_equal",
                    message="Provider and recipient documents are identical.",
                    severity="warning",
                    field_name="recipient_document",
                )
            )

        return issues

    def _validate_document_id(self, field_name: str, value: str, *, required: bool) -> list[ValidationIssue]:
        digits = self._digits_only(value)
        if len(digits) == 11 and self._is_valid_cpf(digits):
            return []
        if len(digits) == 14 and self._is_valid_cnpj(digits):
            return []
        return [
            ValidationIssue(
                code="invalid_document_id",
                message=f"Field '{field_name}' does not contain a valid CPF or CNPJ.",
                severity="error" if required else "warning",
                field_name=field_name,
                metadata={"value": value},
            )
        ]

    def _validate_email(self, field_name: str, value: str) -> list[ValidationIssue]:
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            return []
        return [
            ValidationIssue(
                code="invalid_email",
                message=f"Field '{field_name}' does not contain a valid email format.",
                severity="warning",
                field_name=field_name,
                metadata={"value": value},
            )
        ]

    def _validate_phone(self, field_name: str, value: str) -> list[ValidationIssue]:
        digits = self._digits_only(value)
        if 10 <= len(digits) <= 13:
            return []
        return [
            ValidationIssue(
                code="invalid_phone",
                message=f"Field '{field_name}' does not contain a plausible phone number.",
                severity="warning",
                field_name=field_name,
                metadata={"value": value},
            )
        ]

    def _validate_uf(self, field_name: str, value: str) -> list[ValidationIssue]:
        if value.strip().upper() in _UF_CODES:
            return []
        return [
            ValidationIssue(
                code="invalid_uf",
                message=f"Field '{field_name}' does not contain a valid Brazilian UF.",
                severity="warning",
                field_name=field_name,
                metadata={"value": value},
            )
        ]

    def _validate_date(self, field_name: str, value: str, document: Document) -> list[ValidationIssue]:
        parsed = self._parse_date(value)
        if parsed is None:
            return [
                ValidationIssue(
                    code="invalid_date_format",
                    message=f"Field '{field_name}' does not contain a valid date.",
                    severity="error",
                    field_name=field_name,
                    metadata={"value": value},
                )
            ]

        issues: list[ValidationIssue] = []
        today = datetime.now(UTC).date()
        if parsed > today:
            issues.append(
                ValidationIssue(
                    code="implausible_future_date",
                    message=f"Field '{field_name}' is later than the current processing date.",
                    severity="warning",
                    field_name=field_name,
                    metadata={"value": value, "document_id": document.document_id},
                )
            )
        if parsed.year < 2000:
            issues.append(
                ValidationIssue(
                    code="implausible_historic_date",
                    message=f"Field '{field_name}' is earlier than expected for NFS-e records.",
                    severity="warning",
                    field_name=field_name,
                    metadata={"value": value, "document_id": document.document_id},
                )
            )
        return issues

    def _validate_service_code(self, value: str) -> list[ValidationIssue]:
        if re.fullmatch(r"\d{1,4}(?:\.\d{1,2})?", value.strip()):
            return []
        return [
            ValidationIssue(
                code="invalid_service_code",
                message="Service code does not match the expected municipal or LC116-style format.",
                severity="warning",
                field_name="service_code",
                metadata={"value": value},
            )
        ]

    def _validate_decimal(self, field_name: str, value: str, *, required: bool) -> list[ValidationIssue]:
        parsed = self._parse_decimal(value)
        if parsed is None:
            return [
                ValidationIssue(
                    code="invalid_monetary_value",
                    message=f"Field '{field_name}' does not contain a valid monetary value.",
                    severity="error" if required else "warning",
                    field_name=field_name,
                    metadata={"value": value},
                )
            ]
        if parsed < 0:
            return [
                ValidationIssue(
                    code="negative_monetary_value",
                    message=f"Field '{field_name}' must not be negative.",
                    severity="error" if required else "warning",
                    field_name=field_name,
                    metadata={"value": value},
                )
            ]
        return []

    def _validate_percentage(self, field_name: str, value: str) -> list[ValidationIssue]:
        parsed = self._parse_percentage(value)
        if parsed is None:
            return [
                ValidationIssue(
                    code="invalid_percentage",
                    message=f"Field '{field_name}' does not contain a valid percentage.",
                    severity="warning",
                    field_name=field_name,
                    metadata={"value": value},
                )
            ]
        if parsed < 0 or parsed > 100:
            return [
                ValidationIssue(
                    code="implausible_percentage",
                    message=f"Field '{field_name}' has a percentage outside the expected range.",
                    severity="warning",
                    field_name=field_name,
                    metadata={"value": value},
                )
            ]
        return []

    @staticmethod
    def _has_value(value: str | None) -> bool:
        return value is not None and str(value).strip() != ""

    @staticmethod
    def _digits_only(value: str) -> str:
        return re.sub(r"\D", "", value)

    @classmethod
    def _resolved_text(cls, field: ResolvedField | None) -> str | None:
        if field is None or field.status != "resolved" or not cls._has_value(field.value):
            return None
        return str(field.value).strip()

    @classmethod
    def _resolved_decimal(cls, field: ResolvedField | None) -> Decimal | None:
        text = cls._resolved_text(field)
        if text is None:
            return None
        return cls._parse_decimal(text)

    @classmethod
    def _resolved_percentage(cls, field: ResolvedField | None) -> Decimal | None:
        text = cls._resolved_text(field)
        if text is None:
            return None
        return cls._parse_percentage(text)

    @classmethod
    def _reliable_decimal(cls, field: ResolvedField | None, *, default: Decimal | None = None) -> Decimal | None:
        if field is None:
            return default
        text = cls._resolved_text(field)
        if text is None:
            return default
        return cls._parse_decimal(text)

    @classmethod
    def _reliable_percentage(cls, field: ResolvedField | None) -> Decimal | None:
        if field is None:
            return None
        text = cls._resolved_text(field)
        if text is None:
            return None
        return cls._parse_percentage(text)

    @staticmethod
    def _parse_decimal(value: str) -> Decimal | None:
        normalized = value.strip().replace("R$", "").replace(" ", "")
        if "," in normalized and "." in normalized:
            normalized = normalized.replace(".", "").replace(",", ".")
        elif "," in normalized:
            normalized = normalized.replace(",", ".")
        try:
            return Decimal(normalized)
        except (InvalidOperation, ValueError):
            return None

    @classmethod
    def _parse_percentage(cls, value: str) -> Decimal | None:
        normalized = value.replace("%", "").strip()
        return cls._parse_decimal(normalized)

    @staticmethod
    def _parse_date(value: str) -> date | None:
        cleaned = value.strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        if re.fullmatch(r"\d{2}/\d{4}", cleaned):
            month, year = cleaned.split("/")
            try:
                return date(int(year), int(month), 1)
            except ValueError:
                return None
        return None

    @classmethod
    def _is_valid_cpf(cls, digits: str) -> bool:
        if len(digits) != 11 or digits == digits[0] * 11:
            return False
        first = cls._cpf_digit(digits[:9], start=10)
        second = cls._cpf_digit(digits[:10], start=11)
        return digits[-2:] == f"{first}{second}"

    @staticmethod
    def _cpf_digit(base: str, *, start: int) -> int:
        total = sum(int(digit) * factor for digit, factor in zip(base, range(start, 1, -1), strict=False))
        remainder = (total * 10) % 11
        return 0 if remainder == 10 else remainder

    @classmethod
    def _is_valid_cnpj(cls, digits: str) -> bool:
        if len(digits) != 14 or digits == digits[0] * 14:
            return False
        first = cls._cnpj_digit(digits[:12])
        second = cls._cnpj_digit(digits[:12] + str(first))
        return digits[-2:] == f"{first}{second}"

    @staticmethod
    def _cnpj_digit(base: str) -> int:
        if len(base) == 12:
            factors = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
        else:
            factors = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
        total = sum(int(digit) * factor for digit, factor in zip(base, factors, strict=False))
        remainder = total % 11
        return 0 if remainder < 2 else 11 - remainder
