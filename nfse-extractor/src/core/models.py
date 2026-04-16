"""Canonical typed contracts for document extraction workflows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


IssueSeverity = Literal["info", "warning", "error"]
ResolutionStatus = Literal["resolved", "missing", "conflict"]
DecisionStatus = Literal["auto_approved", "approved_with_warning", "manual_review_required", "rejected"]


class Document(BaseModel):
    """Input document reference used across the pipeline."""

    document_id: str
    source_uri: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedElement(BaseModel):
    """Atomic content extracted from a document by an engine."""

    element_id: str
    element_type: str
    text: str
    page_number: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FieldCandidate(BaseModel):
    """Potential structured field value derived from extracted elements."""

    candidate_id: str
    field_name: str
    value: str
    source_element_ids: list[str] = Field(default_factory=list)
    source_name: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolvedField(BaseModel):
    """Canonical field after resolution across competing candidates."""

    field_name: str
    value: str | None = None
    status: ResolutionStatus = "resolved"
    confidence: float | None = None
    source_candidate_ids: list[str] = Field(default_factory=list)
    resolver_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    """Validation issue found during post-resolution checks."""

    code: str
    message: str
    severity: IssueSeverity
    field_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    """Decision output used to compare or accept a structured result."""

    document_id: str
    decision_status: DecisionStatus = "manual_review_required"
    selected_source: str | None = None
    score: float | None = None
    rationale: str | None = None
    resolved_fields: list[ResolvedField] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
