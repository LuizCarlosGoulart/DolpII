from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core import Document
from src.engines import DolphinExtractionAdapter
from src.preprocessing import PreprocessedDocument, PreprocessedPage


def test_dolphin_adapter_normalizes_mocked_predictions_into_shared_elements(tmp_path: Path) -> None:
    image_path = tmp_path / "note.png"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-1",
        source_uri=image_path.resolve().as_uri(),
        media_type="image/png",
    )
    raw_output = {
        "predictions": [
            {
                "text": "NFSE",
                "score": 0.92,
                "bbox": [10, 20, 30, 12],
                "page": 1,
                "label": "word",
            },
            {
                "content": "12345",
                "confidence": 87,
                "bounding_box": [50, 60, 20, 10],
                "page_number": 1,
                "type": "token",
            },
        ]
    }

    predictor = MagicMock(return_value=raw_output)
    with patch("PIL.Image.open") as image_open:
        image_open.return_value.__enter__.return_value = MagicMock(name="image")
        adapter = DolphinExtractionAdapter(predictor=predictor, model_path="/models/dolphin")
        elements = adapter.extract(document)

    assert len(elements) == 2
    assert elements[0].text == "NFSE"
    assert elements[0].confidence == 0.92
    assert elements[0].bounding_box == (10.0, 20.0, 30.0, 12.0)
    assert elements[0].metadata["source_engine"] == "dolphin"
    assert elements[1].text == "12345"
    assert elements[1].confidence == 0.87


def test_dolphin_adapter_uses_runtime_factory_once_for_preprocessed_pages() -> None:
    predictor = MagicMock(
        return_value={"elements": [{"text": "Prestador", "score": 0.9, "bbox": [1, 2, 3, 4]}]}
    )
    runtime_factory = MagicMock(return_value=predictor)
    preprocessed = PreprocessedDocument(
        document=Document(document_id="doc-2", media_type="image/png"),
        pages=[
            PreprocessedPage(page_number=1, image=MagicMock(name="page-1"), metadata={"normalization_steps": ["deskew"]}),
            PreprocessedPage(page_number=2, image=MagicMock(name="page-2"), metadata={"normalization_steps": ["deskew"]}),
        ],
        metadata={"page_count": 2},
    )

    adapter = DolphinExtractionAdapter(
        runtime_factory=runtime_factory,
        model_path="/content/models/dolphin",
        device="cuda",
    )
    elements = adapter.extract_preprocessed(preprocessed)

    assert len(elements) == 2
    assert elements[0].page_number == 1
    assert elements[1].page_number == 2
    assert elements[0].metadata["preprocessing_metadata"] == {"normalization_steps": ["deskew"]}
    runtime_factory.assert_called_once_with(model_path="/content/models/dolphin", device="cuda")


def test_dolphin_adapter_supports_runtime_factory_without_named_parameters() -> None:
    predictor = MagicMock(return_value=[{"text": "Tomador"}])
    runtime_factory = MagicMock(return_value=predictor)
    preprocessed = PreprocessedDocument(
        document=Document(document_id="doc-2b", media_type="image/png"),
        pages=[PreprocessedPage(page_number=1, image=MagicMock(name="page"))],
    )

    adapter = DolphinExtractionAdapter(runtime_factory=runtime_factory)
    elements = adapter.extract_preprocessed(preprocessed)

    assert len(elements) == 1
    runtime_factory.assert_called_once_with()


def test_dolphin_adapter_normalizes_list_of_strings_output() -> None:
    preprocessed = PreprocessedDocument(
        document=Document(document_id="doc-strings", media_type="image/png"),
        pages=[PreprocessedPage(page_number=1, image=MagicMock(name="page"))],
    )
    adapter = DolphinExtractionAdapter(predictor=MagicMock(return_value=["NFSE", "12345"]))

    elements = adapter.extract_preprocessed(preprocessed)

    assert [element.text for element in elements] == ["NFSE", "12345"]
    assert all(element.metadata["source_engine"] == "dolphin" for element in elements)


def test_dolphin_adapter_rejects_pdf_documents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = Document(
        document_id="doc-3",
        source_uri=pdf_path.resolve().as_uri(),
        media_type="application/pdf",
    )

    adapter = DolphinExtractionAdapter(predictor=MagicMock())

    with pytest.raises(ValueError, match="Convert PDFs to page images"):
        adapter.extract(document)


def test_dolphin_adapter_can_infer_image_media_type_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "note.jpg"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-infer",
        source_uri=image_path.resolve().as_uri(),
    )

    predictor = MagicMock(return_value={"elements": [{"text": "NFSE"}]})
    with patch("PIL.Image.open") as image_open:
        image_open.return_value.__enter__.return_value = MagicMock(name="image")
        adapter = DolphinExtractionAdapter(predictor=predictor)
        elements = adapter.extract(document)

    assert len(elements) == 1
    assert elements[0].metadata["source_engine"] == "dolphin"


def test_dolphin_adapter_requires_predictor_or_runtime_factory() -> None:
    adapter = DolphinExtractionAdapter()
    preprocessed = PreprocessedDocument(
        document=Document(document_id="doc-4", media_type="image/png"),
        pages=[PreprocessedPage(page_number=1, image=MagicMock(name="page"))],
    )

    with pytest.raises(ValueError, match="Configure a Dolphin predictor or runtime_factory"):
        adapter.extract_preprocessed(preprocessed)
