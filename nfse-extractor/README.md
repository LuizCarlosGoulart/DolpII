# NFS-e Extractor

Modular Python project for structured NFS-e extraction and comparison between Tesseract and Dolphin.

## Goals

- Keep production code modular and reusable.
- Separate source code from experiment notebooks.
- Support VS Code + Codex for development.
- Support Google Colab for heavier OCR execution and benchmarking.

## Project Layout

- `src/`: production modules only
- `notebooks/`: thin orchestration entrypoints
- `configs/`: field, scoring, runtime, and decision configuration
- `docs/`: implementation-facing project notes
- `tests/`: unit, integration, and fixtures
- `scripts/`: local packaging and Colab bootstrap helpers

## Quick Start

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

## Local Development Notes

- Keep reusable logic inside `src/`.
- Use notebooks only for setup, orchestration, inspection, and comparison.
- Current local workflow is module-first: adapters, resolver, validation, decision, export, and experiment runner are implemented in `src/`.
- The comparison path still requires an injected `OutputNormalizer`; the repository does not yet ship a production normalizer implementation.

## Colab Execution Notes

- Package the repository with `scripts/package_for_colab.sh`.
- Upload or extract it into `/content/nfse-extractor`, or set `NFSE_PROJECT_ROOT` explicitly.
- Run `scripts/colab_bootstrap.sh` to install the base runtime.
- For Tesseract baseline, use `notebooks/02_tesseract_baseline.ipynb`.
- For Dolphin execution, use `notebooks/03_dolphin_pipeline.ipynb`.
- For side-by-side comparison, use `notebooks/04_comparison.ipynb`.

## Experiment Execution

The minimal experiment path is:

1. Prepare a dataset directory with images or PDFs.
2. Configure a shared preprocessing path.
3. Inject the same `OutputNormalizer`, resolver, validator, and decision engine for both OCR engines.
4. Run `ExperimentComparisonRunner` directly or through `notebooks/04_comparison.ipynb`.
5. Inspect `experiment_summary.json`, `document_metrics.json`, `field_metrics.json`, and per-document bundles.

More detailed experiment notes are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Artifact Conventions

- Per-document processing bundles are written under the selected output root.
- Comparison experiments use:
  - `experiment_summary.json`
  - `document_metrics.json`
  - `field_metrics.json`
  - `experiment_manifest.json`
- Per-document bundles include:
  - `document.json`
  - `raw_elements.json`
  - `normalized_artifacts.json`
  - `resolved_fields.json`
  - `validation_issues.json`
  - `decision_result.json`
  - `summary.json`
  - `manifest.json`
- Manual review artifacts, when generated, include:
  - `manual_review.json`
  - `manual_corrections.template.json`

## Limitations And Known Assumptions

- The project is ready for experimental execution, not production deployment.
- The repository currently depends on an injected `OutputNormalizer` for full end-to-end comparison runs.
- Manual review is file-based and notebook-assisted; there is no full review UI.
- Persistence is filesystem-based and JSON-oriented by design.
- Validation and decision rules are config-driven where practical, but not all pipeline behavior is externalized yet.
