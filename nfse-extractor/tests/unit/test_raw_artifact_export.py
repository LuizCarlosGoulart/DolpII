from pathlib import Path

from src.core import ExtractedElement
from src.export import (
    serialize_extracted_elements,
    write_extracted_elements_json,
    write_text_log,
)


def test_serialize_extracted_elements_returns_json_friendly_dicts() -> None:
    elements = [
        ExtractedElement(
            element_id="doc-1:tesseract:0",
            element_type="text",
            text="NFSE",
            page_number=1,
            confidence=0.95,
            metadata={"source_engine": "tesseract"},
        )
    ]

    serialized = serialize_extracted_elements(elements)

    assert serialized[0]["element_id"] == "doc-1:tesseract:0"
    assert serialized[0]["metadata"]["source_engine"] == "tesseract"


def test_write_extracted_elements_json_writes_output_file(tmp_path: Path) -> None:
    elements = [
        ExtractedElement(
            element_id="doc-1:tesseract:0",
            element_type="text",
            text="NFSE",
            metadata={"source_engine": "tesseract"},
        )
    ]

    output_path = write_extracted_elements_json(elements, tmp_path / "artifacts" / "raw.json")

    assert output_path.exists()
    assert "\"source_engine\": \"tesseract\"" in output_path.read_text(encoding="utf-8")


def test_write_text_log_writes_plain_text_file(tmp_path: Path) -> None:
    output_path = write_text_log("pipeline ok", tmp_path / "logs" / "run.log")

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "pipeline ok"
