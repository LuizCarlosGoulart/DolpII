"""PDF rendering helpers for shared preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import PdfToImageConverter


class PyMuPdfPdfToImageConverter(PdfToImageConverter):
    """Render PDF pages into PIL images using PyMuPDF."""

    def __init__(self, *, dpi: int = 200) -> None:
        self.dpi = dpi

    def convert(self, pdf_path: Path) -> list[Any]:
        try:
            import fitz
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PDF preprocessing requires PyMuPDF and Pillow. "
                "Run the project bootstrap or install requirements.txt."
            ) from exc

        images: list[Any] = []
        scale = self.dpi / 72
        matrix = fitz.Matrix(scale, scale)

        with fitz.open(pdf_path) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                images.append(image)

        if not images:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        return images
