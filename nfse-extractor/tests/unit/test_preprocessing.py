from pathlib import Path
from unittest.mock import patch

import pytest

from src.core import Document
from src.preprocessing import (
    ImageNormalizationHook,
    PdfToImageConverter,
    preprocess_document,
)


class FakeImage:
    def __init__(self, label: str) -> None:
        self.label = label


class StubPdfConverter(PdfToImageConverter):
    def convert(self, pdf_path: Path) -> list[FakeImage]:
        return [FakeImage(f"{pdf_path.name}-p1"), FakeImage(f"{pdf_path.name}-p2")]


class AppendHook(ImageNormalizationHook):
    def __init__(self, name: str, suffix: str) -> None:
        self.name = name
        self.suffix = suffix

    def apply(self, image: FakeImage) -> FakeImage:
        return FakeImage(f"{image.label}{self.suffix}")


def test_preprocess_document_uses_pdf_converter_and_records_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = Document(
        document_id="doc-1",
        source_uri=pdf_path.resolve().as_uri(),
        media_type="application/pdf",
    )

    result = preprocess_document(document, pdf_converter=StubPdfConverter())

    assert result.metadata["page_count"] == 2
    assert result.metadata["media_type"] == "application/pdf"
    assert result.metadata["source_kind"] == "pdf"
    assert result.metadata["used_pdf_converter"] is True
    assert result.pages[0].metadata["media_type"] == "application/pdf"
    assert result.pages[0].metadata["source_kind"] == "pdf"
    assert result.pages[1].page_number == 2


def test_preprocess_document_applies_hooks_to_image_documents(tmp_path: Path) -> None:
    image_path = tmp_path / "note.png"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-2",
        source_uri=image_path.resolve().as_uri(),
        media_type="image/png",
    )

    hooks = [AppendHook("grayscale", "-gray"), AppendHook("contrast", "-contrast")]
    with patch("src.preprocessing.pipeline._load_image_from_path", return_value=FakeImage("base")):
        result = preprocess_document(document, normalization_hooks=hooks)

    assert result.metadata["page_count"] == 1
    assert result.metadata["media_type"] == "image/png"
    assert result.metadata["source_kind"] == "image"
    assert result.metadata["normalization_hooks"] == ["grayscale", "contrast"]
    assert result.pages[0].image.label == "base-gray-contrast"
    assert result.pages[0].metadata["media_type"] == "image/png"
    assert result.pages[0].metadata["normalization_steps"] == ["grayscale", "contrast"]


def test_preprocess_document_can_infer_media_type_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "note.jpg"
    image_path.write_bytes(b"fake-image")
    document = Document(
        document_id="doc-infer",
        source_uri=image_path.resolve().as_uri(),
    )

    with patch("src.preprocessing.pipeline._load_image_from_path", return_value=FakeImage("base")):
        result = preprocess_document(document)

    assert result.metadata["media_type"] == "image/jpeg"
    assert result.metadata["source_kind"] == "image"


def test_preprocess_document_requires_converter_for_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = Document(
        document_id="doc-3",
        source_uri=pdf_path.resolve().as_uri(),
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="PDF converter"):
        preprocess_document(document)


def test_preprocess_document_requires_source_uri() -> None:
    document = Document(document_id="doc-4")

    with pytest.raises(ValueError, match="source_uri"):
        preprocess_document(document)


def test_preprocess_document_rejects_unsupported_media_type(tmp_path: Path) -> None:
    data_path = tmp_path / "note.bin"
    data_path.write_bytes(b"binary")
    document = Document(
        document_id="doc-5",
        source_uri=data_path.resolve().as_uri(),
        media_type="application/octet-stream",
    )

    with pytest.raises(ValueError, match="Unsupported media_type"):
        preprocess_document(document)
