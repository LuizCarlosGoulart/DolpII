"""Config-driven decision engine for resolved NFS-e outputs."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core import DecisionEngine, DecisionResult, Document, ResolvedField, ValidationIssue


class ConfigDrivenDecisionEngine(DecisionEngine):
    """Classify processed documents using validation state, coverage, and confidence."""

    def __init__(self, *, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
        self.thresholds = self._load_yaml("decision_thresholds.yaml").get("thresholds", {})
        self.critical_fields = tuple(self._load_yaml("critical_fields.yaml").get("critical_fields", []))

    def decide(
        self,
        document: Document,
        fields: list[ResolvedField],
        issues: list[ValidationIssue],
    ) -> DecisionResult:
        field_index = {field.field_name: field for field in fields}
        resolved_with_values = [field for field in fields if field.status == "resolved" and self._has_value(field.value)]
        unresolved_conflicts = [field.field_name for field in fields if field.status == "conflict"]
        critical_present = [
            field_name
            for field_name in self.critical_fields
            if (field := field_index.get(field_name)) is not None and field.status == "resolved" and self._has_value(field.value)
        ]

        critical_coverage = len(critical_present) / len(self.critical_fields) if self.critical_fields else 1.0
        confidence_values = [float(field.confidence) for field in resolved_with_values if field.confidence is not None]
        average_field_confidence = self._average(confidence_values) if confidence_values else None
        completeness_score = self._document_completeness_score(document)

        score_components = [critical_coverage]
        if average_field_confidence is not None:
            score_components.append(average_field_confidence)
        if completeness_score is not None:
            score_components.append(completeness_score)
        decision_score = self._average(score_components)

        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")

        minimum_decision_score = float(self.thresholds.get("minimum_decision_score", 0.70))
        minimum_field_confidence = float(self.thresholds.get("minimum_field_confidence", 0.60))
        minimum_critical_coverage = float(self.thresholds.get("minimum_critical_field_coverage", 0.80))
        minimum_completeness_score = float(self.thresholds.get("minimum_completeness_score", minimum_critical_coverage))
        maximum_error_issues = int(self.thresholds.get("maximum_error_issues", 0))
        maximum_warning_issues = int(self.thresholds.get("maximum_warning_issues", 3))

        if error_count > maximum_error_issues:
            decision_status = "rejected"
            rationale = "Blocking validation issues were found."
            triggered_rule = "blocking_validation_issues"
        elif critical_coverage < minimum_critical_coverage:
            decision_status = "rejected"
            rationale = "Critical field coverage is below the configured minimum."
            triggered_rule = "critical_field_coverage_below_minimum"
        elif unresolved_conflicts or (
            average_field_confidence is not None and average_field_confidence < minimum_field_confidence
        ):
            decision_status = "manual_review_required"
            rationale = "Unresolved conflicts or low field confidence require manual review."
            triggered_rule = "unresolved_conflicts_or_low_field_confidence"
        elif completeness_score is not None and completeness_score < minimum_completeness_score:
            decision_status = "manual_review_required"
            rationale = "Available completeness score is below the configured minimum."
            triggered_rule = "completeness_below_minimum"
        elif decision_score < minimum_decision_score:
            decision_status = "manual_review_required"
            rationale = "Overall decision score is below the configured minimum."
            triggered_rule = "decision_score_below_minimum"
        elif warning_count > maximum_warning_issues:
            decision_status = "manual_review_required"
            rationale = "Warning volume is above the configured tolerance."
            triggered_rule = "warning_volume_above_tolerance"
        elif warning_count > 0:
            decision_status = "approved_with_warning"
            rationale = "Document passed blocking checks with non-blocking validation warnings."
            triggered_rule = "non_blocking_warnings_present"
        else:
            decision_status = "auto_approved"
            rationale = "Document meets configured confidence, coverage, and validation thresholds."
            triggered_rule = "all_thresholds_satisfied"

        return DecisionResult(
            document_id=document.document_id,
            decision_status=decision_status,
            selected_source=document.metadata.get("selected_source"),
            score=decision_score,
            rationale=rationale,
            resolved_fields=fields,
            validation_issues=issues,
            metadata={
                "critical_field_coverage": critical_coverage,
                "critical_fields_present": critical_present,
                "critical_fields_expected": list(self.critical_fields),
                "average_field_confidence": average_field_confidence,
                "completeness_score": completeness_score,
                "warning_count": warning_count,
                "error_count": error_count,
                "unresolved_conflicts": unresolved_conflicts,
                "triggered_rule": triggered_rule,
                "thresholds_used": {
                    "minimum_decision_score": minimum_decision_score,
                    "minimum_field_confidence": minimum_field_confidence,
                    "minimum_critical_field_coverage": minimum_critical_coverage,
                    "minimum_completeness_score": minimum_completeness_score,
                    "maximum_error_issues": maximum_error_issues,
                    "maximum_warning_issues": maximum_warning_issues,
                },
            },
        )

    @staticmethod
    def _has_value(value: str | None) -> bool:
        return value is not None and str(value).strip() != ""

    @staticmethod
    def _average(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _document_completeness_score(document: Document) -> float | None:
        raw_value = document.metadata.get("completeness_score")
        if not isinstance(raw_value, (int, float)):
            return None
        return max(0.0, min(float(raw_value), 1.0))

    def _load_yaml(self, filename: str) -> dict:
        with (self.config_dir / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
