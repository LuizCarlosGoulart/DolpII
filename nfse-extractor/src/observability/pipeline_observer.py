"""Lightweight observability helpers for experiment-friendly pipeline runs."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator
import json

from src.core import DecisionResult, Document, ResolvedField, ValidationIssue


class PipelineObserver:
    """Collect per-document stage timings and compact status summaries."""

    def __init__(self, *, engine_id: str, document: Document) -> None:
        self.engine_id = engine_id
        self.document = document
        self.stage_timings_ms: dict[str, float] = {}

    @contextmanager
    def measure_stage(self, stage_name: str) -> Iterator[None]:
        """Measure one pipeline stage duration in milliseconds."""
        started_at = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            self.stage_timings_ms[stage_name] = round(elapsed_ms, 3)

    def build_summary(
        self,
        *,
        resolved_fields: list[ResolvedField],
        validation_issues: list[ValidationIssue],
        decision_result: DecisionResult | None = None,
    ) -> dict[str, Any]:
        """Build a concise structured summary for one processed document."""
        conflict_count = sum(1 for field in resolved_fields if field.status == "conflict")
        missing_count = sum(1 for field in resolved_fields if field.status == "missing")
        resolved_count = sum(1 for field in resolved_fields if field.status == "resolved")
        issue_counts = {
            "info": sum(1 for issue in validation_issues if issue.severity == "info"),
            "warning": sum(1 for issue in validation_issues if issue.severity == "warning"),
            "error": sum(1 for issue in validation_issues if issue.severity == "error"),
        }
        decision_status = decision_result.decision_status if decision_result is not None else None

        return {
            "document_id": self.document.document_id,
            "engine_id": self.engine_id,
            "document_status": self._document_status(decision_result, issue_counts["error"], conflict_count),
            "decision_status": decision_status,
            "manual_review_required_count": 1 if decision_status == "manual_review_required" else 0,
            "resolved_field_count": resolved_count,
            "missing_field_count": missing_count,
            "conflict_count": conflict_count,
            "validation_issue_counts": issue_counts,
            "stage_timings_ms": dict(self.stage_timings_ms),
        }

    @staticmethod
    def _document_status(
        decision_result: DecisionResult | None,
        error_count: int,
        conflict_count: int,
    ) -> str:
        if decision_result is not None:
            return decision_result.decision_status
        if error_count > 0:
            return "error"
        if conflict_count > 0:
            return "conflict"
        return "processed"


def format_structured_log(event_name: str, payload: dict[str, Any]) -> str:
    """Render one concise JSON log line for notebook or local experiment logs."""
    return json.dumps(
        {
            "event": event_name,
            **payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
