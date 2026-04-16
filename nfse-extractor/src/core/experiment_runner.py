"""Experiment runner for comparing extraction engines on a shared pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from src.export import persist_processing_bundle, write_extracted_elements_json
from src.ingestion import load_documents
from src.normalization import normalize_raw_elements
from src.observability import PipelineObserver
from src.preprocessing import preprocess_document

from .field_dictionary import load_field_dictionary
from .interfaces import DecisionEngine, FieldResolver, OutputNormalizer, Validator
from .models import DecisionResult, Document, ResolvedField


FieldCorrectnessHook = Callable[[str, ResolvedField | None, Any, Document], bool | None]


class ExperimentComparisonRunner:
    """Run multiple extraction engines through the same upper-layer pipeline."""

    def __init__(
        self,
        *,
        engines: dict[str, Any],
        output_normalizer: OutputNormalizer,
        field_resolver: FieldResolver,
        validator: Validator,
        decision_engine: DecisionEngine,
        output_root: str | Path,
        preprocessor: Callable[[Document], Any] | None = None,
        pdf_converter: Any | None = None,
        normalization_hooks: list[Any] | None = None,
        field_correctness_hook: FieldCorrectnessHook | None = None,
        field_dictionary_path: str | Path | None = None,
    ) -> None:
        self.engines = engines
        self.output_normalizer = output_normalizer
        self.field_resolver = field_resolver
        self.validator = validator
        self.decision_engine = decision_engine
        self.output_root = Path(output_root)
        self.field_correctness_hook = field_correctness_hook
        self.field_dictionary = load_field_dictionary(field_dictionary_path)
        self.field_names = tuple(self.field_dictionary.by_internal_name())

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda document: preprocess_document(
                document,
                pdf_converter=pdf_converter,
                normalization_hooks=normalization_hooks,
            )

    def run(
        self,
        dataset_paths: list[str | Path],
        *,
        experiment_name: str = "engine_comparison",
        ground_truth_by_document: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        experiment_root = self.output_root / experiment_name
        experiment_root.mkdir(parents=True, exist_ok=True)

        ordered_dataset_paths = [Path(path) for path in dataset_paths]
        documents = load_documents(dataset_paths)
        document_metrics: list[dict[str, Any]] = []
        field_stats = self._initialize_field_stats()

        for sample_index, document in enumerate(documents):
            preprocessing_ms, preprocessed_document = self._preprocess_document(document)
            document_truth = (ground_truth_by_document or {}).get(document.document_id, {})

            for engine_id in sorted(self.engines):
                engine = self.engines[engine_id]
                observer = PipelineObserver(engine_id=engine_id, document=document)
                observer.stage_timings_ms["preprocessing"] = preprocessing_ms

                with observer.measure_stage("extraction"):
                    raw_elements = self._extract_preprocessed(engine, preprocessed_document)

                with observer.measure_stage("normalization"):
                    normalized_artifacts = normalize_raw_elements(raw_elements)
                    candidates = self.output_normalizer.normalize(document, raw_elements)

                with observer.measure_stage("resolution"):
                    resolved_fields = self.field_resolver.resolve(document, candidates)

                with observer.measure_stage("validation"):
                    validation_issues = self.validator.validate(document, resolved_fields)

                with observer.measure_stage("decision"):
                    decision_result = self.decision_engine.decide(document, resolved_fields, validation_issues)

                with observer.measure_stage("export"):
                    engine_output_root = experiment_root / engine_id
                    files = persist_processing_bundle(
                        document=document,
                        normalized_artifacts=normalized_artifacts,
                        resolved_fields=resolved_fields,
                        validation_issues=validation_issues,
                        decision_result=decision_result,
                        output_root=engine_output_root,
                    )
                    bundle_dir = files["manifest"].parent
                    raw_elements_path = write_extracted_elements_json(raw_elements, bundle_dir / "raw_elements.json")

                summary = observer.build_summary(
                    resolved_fields=resolved_fields,
                    validation_issues=validation_issues,
                    decision_result=decision_result,
                )
                engine_processing_time_ms = round(
                    sum(
                        duration
                        for stage_name, duration in observer.stage_timings_ms.items()
                        if stage_name != "preprocessing"
                    ),
                    3,
                )
                summary["sample_index"] = sample_index
                summary["shared_preprocessing_time_ms"] = preprocessing_ms
                summary["engine_processing_time_ms"] = engine_processing_time_ms
                summary["processing_time_ms"] = round(preprocessing_ms + engine_processing_time_ms, 3)
                summary["fill_rate"] = self._document_fill_rate(resolved_fields)
                summary["bundle_dir"] = str(bundle_dir)
                summary["raw_elements_path"] = str(raw_elements_path)
                document_metrics.append(summary)

                self._accumulate_field_metrics(
                    field_stats=field_stats,
                    engine_id=engine_id,
                    document=document,
                    resolved_fields=resolved_fields,
                    expected_fields=document_truth,
                )

        field_metrics = self._build_field_metrics(field_stats, total_documents=len(documents))
        experiment_summary = self._build_experiment_summary(
            experiment_name=experiment_name,
            document_metrics=document_metrics,
            total_documents=len(documents),
        )

        document_metrics_path = self._write_json(experiment_root / "document_metrics.json", document_metrics)
        field_metrics_path = self._write_json(experiment_root / "field_metrics.json", field_metrics)
        experiment_summary_path = self._write_json(experiment_root / "experiment_summary.json", experiment_summary)
        experiment_manifest_path = self._write_json(
            experiment_root / "experiment_manifest.json",
            {
                "experiment_name": experiment_name,
                "dataset_paths": [str(path) for path in ordered_dataset_paths],
                "engine_ids": sorted(self.engines),
                "field_names": list(self.field_names),
                "total_documents": len(documents),
            },
        )

        return {
            "experiment_root": experiment_root,
            "document_metrics_path": document_metrics_path,
            "field_metrics_path": field_metrics_path,
            "experiment_summary_path": experiment_summary_path,
            "experiment_manifest_path": experiment_manifest_path,
            "document_metrics": document_metrics,
            "field_metrics": field_metrics,
            "experiment_summary": experiment_summary,
        }

    def _preprocess_document(self, document: Document) -> tuple[float, Any]:
        started_at = perf_counter()
        preprocessed_document = self.preprocessor(document)
        elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)
        return elapsed_ms, preprocessed_document

    @staticmethod
    def _extract_preprocessed(engine: Any, preprocessed_document: Any) -> list[Any]:
        extractor = getattr(engine, "extract_preprocessed", None)
        if extractor is None:
            raise ValueError("Shared comparison requires engines that support extract_preprocessed().")
        return extractor(preprocessed_document)

    def _initialize_field_stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            engine_id: {
                field_name: {
                    "filled_count": 0,
                    "conflict_count": 0,
                    "correct_count": 0,
                    "incorrect_count": 0,
                    "correctness_evaluated_count": 0,
                }
                for field_name in self.field_names
            }
            for engine_id in sorted(self.engines)
        }

    def _accumulate_field_metrics(
        self,
        *,
        field_stats: dict[str, dict[str, dict[str, int]]],
        engine_id: str,
        document: Document,
        resolved_fields: list[ResolvedField],
        expected_fields: dict[str, Any],
    ) -> None:
        field_index = {field.field_name: field for field in resolved_fields}
        for field_name in self.field_names:
            stats = field_stats[engine_id][field_name]
            field = field_index.get(field_name)
            if field is not None and field.status == "resolved" and self._has_value(field.value):
                stats["filled_count"] += 1
            if field is not None and field.status == "conflict":
                stats["conflict_count"] += 1
            if self.field_correctness_hook is None:
                continue
            correctness = self.field_correctness_hook(
                field_name,
                field,
                expected_fields.get(field_name),
                document,
            )
            if correctness is None:
                continue
            stats["correctness_evaluated_count"] += 1
            if correctness:
                stats["correct_count"] += 1
            else:
                stats["incorrect_count"] += 1

    def _build_field_metrics(
        self,
        field_stats: dict[str, dict[str, dict[str, int]]],
        *,
        total_documents: int,
    ) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        denominator = total_documents or 1
        for engine_id in sorted(field_stats):
            engine_stats = field_stats[engine_id]
            for field_name in self.field_names:
                stats = engine_stats[field_name]
                evaluated = stats["correctness_evaluated_count"]
                metrics.append(
                    {
                        "engine_id": engine_id,
                        "field_name": field_name,
                        "fill_rate": stats["filled_count"] / denominator,
                        "conflict_rate": stats["conflict_count"] / denominator,
                        "correct_count": stats["correct_count"],
                        "incorrect_count": stats["incorrect_count"],
                        "correctness_evaluated_count": evaluated,
                        "correctness_rate": (
                            stats["correct_count"] / evaluated if evaluated > 0 else None
                        ),
                    }
                )
        return metrics

    @staticmethod
    def _build_experiment_summary(
        *,
        experiment_name: str,
        document_metrics: list[dict[str, Any]],
        total_documents: int,
    ) -> dict[str, Any]:
        by_engine: dict[str, list[dict[str, Any]]] = {}
        for metric in document_metrics:
            by_engine.setdefault(metric["engine_id"], []).append(metric)

        engine_summaries: dict[str, dict[str, Any]] = {}
        for engine_id, metrics in by_engine.items():
            count = len(metrics) or 1
            engine_summaries[engine_id] = {
                "documents_processed": len(metrics),
                "manual_review_rate": sum(item["manual_review_required_count"] for item in metrics) / count,
                "average_shared_preprocessing_time_ms": (
                    sum(item["shared_preprocessing_time_ms"] for item in metrics) / count
                ),
                "average_engine_processing_time_ms": (
                    sum(item["engine_processing_time_ms"] for item in metrics) / count
                ),
                "average_processing_time_ms": sum(item["processing_time_ms"] for item in metrics) / count,
                "status_counts": _status_counts(metrics),
            }

        return {
            "experiment_name": experiment_name,
            "total_documents": total_documents,
            "engines": sorted(by_engine),
            "engine_summaries": engine_summaries,
        }

    def _document_fill_rate(self, resolved_fields: list[ResolvedField]) -> float:
        resolved_count = sum(
            1
            for field in resolved_fields
            if field.status == "resolved" and self._has_value(field.value)
        )
        denominator = len(self.field_names) or 1
        return resolved_count / denominator

    @staticmethod
    def _has_value(value: str | None) -> bool:
        return value is not None and str(value).strip() != ""

    @staticmethod
    def _write_json(path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return path


def _status_counts(metrics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in metrics:
        status = str(item["document_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts
