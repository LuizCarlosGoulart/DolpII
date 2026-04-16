from src.core import ExtractedElement
from src.normalization import normalize_raw_elements


def test_normalize_raw_elements_preserves_tesseract_hints() -> None:
    elements = [
        ExtractedElement(
            element_id="doc-1:tesseract:0",
            element_type="text",
            text="NFSE",
            page_number=1,
            bounding_box=(10.0, 20.0, 30.0, 12.0),
            confidence=0.95,
            metadata={
                "source_engine": "tesseract",
                "block_num": 2,
                "line_num": 3,
                "word_num": 4,
            },
        )
    ]

    normalized = normalize_raw_elements(elements)

    assert len(normalized) == 1
    assert normalized[0].source_engine == "tesseract"
    assert normalized[0].source_element_type == "text"
    assert normalized[0].raw_text == "NFSE"
    assert normalized[0].confidence == 0.95
    assert normalized[0].page_number == 1
    assert normalized[0].bounding_box == (10.0, 20.0, 30.0, 12.0)
    assert normalized[0].block_hint == 2
    assert normalized[0].line_hint == 3
    assert normalized[0].word_hint == 4
    assert normalized[0].trace["element_type"] == "text"


def test_normalize_raw_elements_preserves_dolphin_hints_and_traceability() -> None:
    elements = [
        ExtractedElement(
            element_id="doc-1:dolphin:0",
            element_type="token",
            text="Prestador",
            page_number=2,
            bounding_box=(1.0, 2.0, 3.0, 4.0),
            confidence=0.87,
            metadata={
                "source_engine": "dolphin",
                "raw_label": "word",
                "device": "cuda",
                "model_path": "/content/models/dolphin",
            },
        )
    ]

    normalized = normalize_raw_elements(elements)

    assert len(normalized) == 1
    assert normalized[0].source_element_id == "doc-1:dolphin:0"
    assert normalized[0].source_engine == "dolphin"
    assert normalized[0].source_element_type == "token"
    assert normalized[0].raw_text == "Prestador"
    assert normalized[0].label_hint == "word"
    assert normalized[0].trace["source_metadata"]["device"] == "cuda"
    assert normalized[0].trace["source_metadata"]["model_path"] == "/content/models/dolphin"


def test_normalize_raw_elements_copies_trace_metadata() -> None:
    source_metadata = {"source_engine": "tesseract", "block_num": 2}
    elements = [
        ExtractedElement(
            element_id="doc-1:tesseract:1",
            element_type="text",
            text="12345",
            metadata=source_metadata,
        )
    ]

    normalized = normalize_raw_elements(elements)
    source_metadata["block_num"] = 99

    assert normalized[0].trace["source_metadata"]["block_num"] == 2
