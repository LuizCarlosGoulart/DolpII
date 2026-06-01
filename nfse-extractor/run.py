"""Command-line entry point for the NFS-e extraction pipeline.

Usage
-----
    # Tesseract (default, no GPU needed):
    python run.py path/to/document.pdf

    # Dolphin VLM (requires GPU + model download):
    python run.py path/to/document.pdf --engine dolphin

    # Custom output and config directories:
    python run.py path/to/document.pdf --output-dir ./results --config-dir ./configs

    # Increase verbosity:
    python run.py path/to/document.pdf -v

Exit codes: 0 = success, 1 = extraction/validation error, 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _build_pipeline(config_dir: Path):
    """Instantiate normalizer, resolver, validator, and decision engine."""
    from src.normalization import ConfigDrivenOutputNormalizer
    from src.resolver import ConfigDrivenFieldResolver
    from src.validation import ConfigDrivenValidator
    from src.decision import ConfigDrivenDecisionEngine

    kw = {"config_dir": config_dir}
    return (
        ConfigDrivenOutputNormalizer(**kw),
        ConfigDrivenFieldResolver(**kw),
        ConfigDrivenValidator(**kw),
        ConfigDrivenDecisionEngine(**kw),
    )


def _configure_tesseract() -> None:
    """Point pytesseract at the system Tesseract binary when not on PATH."""
    import shutil
    import sys
    import pytesseract

    if shutil.which("tesseract"):
        return  # already on PATH

    candidates: list[str] = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif sys.platform == "darwin":
        candidates = ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
    else:
        candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]

    for candidate in candidates:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return

    raise RuntimeError(
        "Tesseract binary not found. Install it and add it to PATH, "
        "or set pytesseract.pytesseract.tesseract_cmd manually."
    )


def _build_engine(engine_name: str, model_path: str | None, device: str | None):
    if engine_name == "tesseract":
        _configure_tesseract()
        from src.engines import TesseractExtractionAdapter
        return TesseractExtractionAdapter(language="por")

    if engine_name == "dolphin":
        from src.engines import DolphinExtractionAdapter, load_dolphin_runtime
        return DolphinExtractionAdapter(
            runtime_factory=load_dolphin_runtime,
            model_path=model_path,
            device=device,
        )

    raise ValueError(f"Unknown engine: {engine_name!r}. Choose 'tesseract' or 'dolphin'.")


def _run(args: argparse.Namespace) -> int:
    log = logging.getLogger("run")

    config_dir = Path(args.config_dir).resolve()
    if not config_dir.is_dir():
        log.error("Config directory does not exist: %s", config_dir)
        return 2

    # Startup config integrity check
    from src.validation import validate_config_integrity
    issues = validate_config_integrity(config_dir)
    if issues:
        for issue in issues:
            log.error("Config integrity: %s", issue)
        return 2

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        log.error("Input file does not exist: %s", input_path)
        return 2

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from src.ingestion import load_document
    from src.preprocessing import preprocess_document
    from src.preprocessing.pdf import PyMuPdfPdfToImageConverter
    from src.normalization import normalize_raw_elements
    from src.export import persist_processing_bundle, write_extracted_elements_json
    from src.observability import PipelineObserver

    log.info("Loading document: %s", input_path)
    document = load_document(input_path)

    log.info("Preprocessing …")
    preprocessed = preprocess_document(
        document,
        pdf_converter=PyMuPdfPdfToImageConverter(),
    )
    log.info("Pages preprocessed: %d", len(preprocessed.pages))

    engine_name = args.engine
    log.info("Building engine: %s", engine_name)
    engine = _build_engine(engine_name, args.model_path, args.device)

    normalizer, resolver, validator, decision_engine = _build_pipeline(config_dir)

    observer = PipelineObserver(engine_id=engine_name, document=document)

    log.info("Extracting …")
    with observer.measure_stage("extraction"):
        raw_elements = engine.extract_preprocessed(preprocessed)
    log.info("Extracted %d elements", len(raw_elements))

    log.info("Normalizing …")
    with observer.measure_stage("normalization"):
        normalized_artifacts = normalize_raw_elements(raw_elements)
        candidates = normalizer.normalize(document, raw_elements)
    log.info("Produced %d field candidates", len(candidates))

    log.info("Resolving fields …")
    with observer.measure_stage("resolution"):
        resolved_fields = resolver.resolve(document, candidates)

    log.info("Validating …")
    with observer.measure_stage("validation"):
        validation_issues = validator.validate(document, resolved_fields)

    log.info("Deciding …")
    with observer.measure_stage("decision"):
        decision_result = decision_engine.decide(document, resolved_fields, validation_issues)

    log.info("Exporting …")
    with observer.measure_stage("export"):
        files = persist_processing_bundle(
            document=document,
            normalized_artifacts=normalized_artifacts,
            resolved_fields=resolved_fields,
            validation_issues=validation_issues,
            decision_result=decision_result,
            output_root=output_root,
        )
        bundle_dir = files["manifest"].parent
        write_extracted_elements_json(raw_elements, bundle_dir / "raw_elements.json")

    summary = observer.build_summary(
        resolved_fields=resolved_fields,
        validation_issues=validation_issues,
        decision_result=decision_result,
    )

    # Human-readable results to stdout
    resolved_with_values = {
        f.field_name: f.value
        for f in resolved_fields
        if f.status == "resolved" and f.value
    }
    print(json.dumps({
        "document_id": document.document_id,
        "file": str(input_path),
        "engine": engine_name,
        "decision_status": decision_result.decision_status,
        "decision_score": decision_result.score,
        "rationale": decision_result.rationale,
        "resolved_fields": resolved_with_values,
        "validation_issues": [
            {"code": i.code, "severity": i.severity, "message": i.message}
            for i in validation_issues
        ],
        "bundle_dir": str(bundle_dir),
        "stage_timings_ms": summary.get("stage_timings_ms", {}),
    }, indent=2, ensure_ascii=False))

    log.info("Done. Bundle written to: %s", bundle_dir)
    return 0


def main() -> None:
    # Derive the default config dir relative to this script
    _here = Path(__file__).resolve().parent
    _default_config = _here / "configs"

    parser = argparse.ArgumentParser(
        description="Extract structured fields from an NFS-e PDF or image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input PDF or image file.")
    parser.add_argument(
        "--engine",
        choices=["tesseract", "dolphin"],
        default="tesseract",
        help="OCR engine to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Root directory for output bundles.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(_default_config),
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Dolphin: HuggingFace model ID or local snapshot path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Dolphin: 'cuda', 'cpu', or None (auto-detect).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    sys.exit(_run(args))


if __name__ == "__main__":
    main()
