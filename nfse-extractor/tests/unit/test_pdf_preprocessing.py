from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.preprocessing import PyMuPdfPdfToImageConverter


class FakePixmap:
    width = 2
    height = 1
    samples = b"\x00\x00\x00\xff\xff\xff"


class FakePage:
    def get_pixmap(self, *, matrix, alpha: bool) -> FakePixmap:
        assert matrix is not None
        assert alpha is False
        return FakePixmap()


class FakeDocument:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "FakeDocument":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def __iter__(self):
        return iter(self.pages)


def test_pymupdf_converter_renders_pages_to_images(tmp_path: Path) -> None:
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    fake_fitz = SimpleNamespace(
        Matrix=lambda scale_x, scale_y: (scale_x, scale_y),
        open=lambda path: FakeDocument([FakePage(), FakePage()]),
    )

    with patch.dict("sys.modules", {"fitz": fake_fitz}):
        images = PyMuPdfPdfToImageConverter(dpi=144).convert(pdf_path)

    assert len(images) == 2
    assert images[0].mode == "RGB"
    assert images[0].size == (2, 1)


def test_pymupdf_converter_rejects_empty_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    fake_fitz = SimpleNamespace(
        Matrix=lambda scale_x, scale_y: (scale_x, scale_y),
        open=lambda path: FakeDocument([]),
    )

    with patch.dict("sys.modules", {"fitz": fake_fitz}):
        with pytest.raises(ValueError, match="no pages"):
            PyMuPdfPdfToImageConverter().convert(pdf_path)
