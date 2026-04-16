"""Abstract interfaces for engine-agnostic extraction workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import (
    DecisionResult,
    Document,
    ExtractedElement,
    FieldCandidate,
    ResolvedField,
    ValidationIssue,
)


class ExtractionEngine(ABC):
    """Produces raw extracted elements from a document."""

    @abstractmethod
    def extract(self, document: Document) -> list[ExtractedElement]:
        """Extract atomic elements from the input document."""


class OutputNormalizer(ABC):
    """Converts raw extracted elements into canonical field candidates."""

    @abstractmethod
    def normalize(
        self,
        document: Document,
        elements: list[ExtractedElement],
    ) -> list[FieldCandidate]:
        """Normalize engine output into canonical field candidates."""


class FieldResolver(ABC):
    """Resolves competing candidates into canonical fields."""

    @abstractmethod
    def resolve(
        self,
        document: Document,
        candidates: list[FieldCandidate],
    ) -> list[ResolvedField]:
        """Resolve candidate values into canonical fields."""


class Validator(ABC):
    """Validates resolved fields and returns structured issues."""

    @abstractmethod
    def validate(
        self,
        document: Document,
        fields: list[ResolvedField],
    ) -> list[ValidationIssue]:
        """Validate resolved fields for a document."""


class DecisionEngine(ABC):
    """Decides whether a resolved result should be accepted or preferred."""

    @abstractmethod
    def decide(
        self,
        document: Document,
        fields: list[ResolvedField],
        issues: list[ValidationIssue],
    ) -> DecisionResult:
        """Produce a decision result for the document."""


class Exporter(ABC):
    """Exports a decision result into a transport-friendly representation."""

    @abstractmethod
    def export(
        self,
        document: Document,
        result: DecisionResult,
    ) -> dict[str, Any] | str:
        """Export the decision result."""
