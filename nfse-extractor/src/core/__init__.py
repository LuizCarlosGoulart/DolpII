"""Core domain models and shared contracts."""

from .field_dictionary import (
    CanonicalFieldDefinition,
    FieldDictionary,
    default_field_dictionary_path,
    load_field_dictionary,
)
from .interfaces import (
    DecisionEngine,
    Exporter,
    ExtractionEngine,
    FieldResolver,
    OutputNormalizer,
    Validator,
)
from .models import (
    DecisionResult,
    Document,
    ExtractedElement,
    FieldCandidate,
    ResolvedField,
    ValidationIssue,
)

__all__ = [
    "CanonicalFieldDefinition",
    "DecisionEngine",
    "DecisionResult",
    "Document",
    "ExperimentComparisonRunner",
    "Exporter",
    "ExtractedElement",
    "ExtractionEngine",
    "FieldDictionary",
    "FieldCandidate",
    "FieldResolver",
    "OutputNormalizer",
    "ResolvedField",
    "ValidationIssue",
    "Validator",
    "default_field_dictionary_path",
    "load_field_dictionary",
]


def __getattr__(name: str):
    if name == "ExperimentComparisonRunner":
        from .experiment_runner import ExperimentComparisonRunner

        return ExperimentComparisonRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
