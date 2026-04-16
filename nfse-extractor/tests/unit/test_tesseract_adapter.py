from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import Document
from src.engines import TesseractExtractionAdapter
from src.preprocessing import PreprocessedDocument, PreprocessedPage


def test_tesseract_adapter_returns_raw_elements_with_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "note.png"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-1",
        source_uri=image_path.resolve().as_uri(),
        media_type="image/png",
    )
    raw_output = {
        "text": ["NFSE", "12345", ""],
        "conf": ["95", "80", "-1"],
        "page_num": [1, 1, 1],
        "left": [10, 30, 0],
        "top": [20, 40, 0],
        "width": [50, 60, 0],
        "height": [12, 14, 0],
        "block_num": [1, 1, 1],
        "line_num": [1, 1, 1],
        "word_num": [1, 2, 3],
    }

    with (
        patch("src.engines.tesseract_adapter.Image.open") as image_open,
        patch("src.engines.tesseract_adapter.pytesseract.image_to_data", return_value=raw_output) as image_to_data,
    ):
        image_open.return_value.__enter__.return_value = MagicMock(name="image")
        adapter = TesseractExtractionAdapter(language="por")
        elements = adapter.extract(document)

    assert len(elements) == 2
    assert elements[0].text == "NFSE"
    assert elements[0].confidence == 0.95
    assert elements[0].bounding_box == (10.0, 20.0, 50.0, 12.0)
    assert elements[0].metadata["source_engine"] == "tesseract"
    assert elements[1].text == "12345"
    image_to_data.assert_called_once()


def test_tesseract_adapter_rejects_pdf_documents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = Document(
        document_id="doc-2",
        source_uri=pdf_path.resolve().as_uri(),
        media_type="application/pdf",
    )

    adapter = TesseractExtractionAdapter()

    with pytest.raises(ValueError, match="Convert PDFs to page images"):
        adapter.extract(document)


def test_tesseract_adapter_requires_source_uri() -> None:
    adapter = TesseractExtractionAdapter()
    document = Document(document_id="doc-3")

    with pytest.raises(ValueError, match="source_uri"):
        adapter.extract(document)


def test_tesseract_adapter_can_infer_image_media_type_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "note.jpg"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-infer",
        source_uri=image_path.resolve().as_uri(),
    )
    raw_output = {
        "text": ["NFSE"],
        "conf": ["95"],
        "page_num": [1],
        "left": [10],
        "top": [20],
        "width": [30],
        "height": [12],
        "block_num": [1],
        "line_num": [1],
        "word_num": [1],
    }

    with (
        patch("src.engines.tesseract_adapter.Image.open") as image_open,
        patch("src.engines.tesseract_adapter.pytesseract.image_to_data", return_value=raw_output),
    ):
        image_open.return_value.__enter__.return_value = MagicMock(name="image")
        adapter = TesseractExtractionAdapter()
        elements = adapter.extract(document)

    assert len(elements) == 1
    assert elements[0].metadata["source_engine"] == "tesseract"


def test_tesseract_adapter_extracts_from_preprocessed_pages() -> None:
    document = Document(document_id="doc-4", media_type="image/png")
    preprocessed = PreprocessedDocument(
        document=document,
        pages=[
            PreprocessedPage(
                page_number=1,
                image=MagicMock(name="page-image"),
                metadata={"normalization_steps": ["grayscale"]},
            )
        ],
        metadata={"page_count": 1},
    )
    raw_output = {
        "text": ["Prestador"],
        "conf": ["90"],
        "page_num": [""],
        "left": [5],
        "top": [7],
        "width": [20],
        "height": [8],
        "block_num": [1],
        "line_num": [1],
        "word_num": [1],
    }

    with patch("src.engines.tesseract_adapter.pytesseract.image_to_data", return_value=raw_output):
        adapter = TesseractExtractionAdapter()
        elements = adapter.extract_preprocessed(preprocessed)

    assert len(elements) == 1
    assert elements[0].page_number == 1
    assert elements[0].metadata["source_engine"] == "tesseract"
    assert elements[0].metadata["preprocessing_metadata"] == {"normalization_steps": ["grayscale"]}
