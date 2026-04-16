# Experiment Plan

## Scope

Current experiments are designed to compare Tesseract and Dolphin under the same dataset, preprocessing path, resolver, validation layer, and decision logic.

## Recommended Execution Flow

1. Prepare a dataset directory with local sample files.
2. Run environment validation with `notebooks/00_environment_check.ipynb` when setting up a new runtime.
3. Use `notebooks/02_tesseract_baseline.ipynb` for isolated Tesseract checks.
4. Use `notebooks/03_dolphin_pipeline.ipynb` for isolated Dolphin checks.
5. Use `notebooks/04_comparison.ipynb` for shared comparison runs.

## Required Inputs

- dataset path or sample path
- runtime/bootstrap choice
- Dolphin runtime factory path when testing Dolphin
- injected `OutputNormalizer` path for full comparison runs
- optional output root override for local or Colab execution

## Artifact Layout

Comparison runs should write to one experiment root with:

- `experiment_summary.json`
- `document_metrics.json`
- `field_metrics.json`
- `experiment_manifest.json`

Per engine and per document, the runner writes bundles with:

- `document.json`
- `raw_elements.json`
- `normalized_artifacts.json`
- `resolved_fields.json`
- `validation_issues.json`
- `decision_result.json`
- `summary.json`
- `manifest.json`

If manual review is needed, keep review artifacts next to the experiment outputs or under a sibling review root.

## Metrics To Inspect First

- final document status
- fill rate
- conflict count / conflict rate
- manual review rate
- shared preprocessing time vs engine processing time
- field-level correctness placeholders when a correctness hook is available

## Practical Limits

- The repository does not yet ship a built-in production `OutputNormalizer`, so comparison runs must inject one explicitly.
- Experimental persistence is JSON-on-filesystem, not database-backed.
- Notebooks are orchestration-only and assume the reusable pipeline already exists in `src/`.
- Colab execution is supported, but path configuration still matters; `NFSE_PROJECT_ROOT` should be set when not using `/content/nfse-extractor`.
