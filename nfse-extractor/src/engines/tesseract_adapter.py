"""Baseline Tesseract extraction adapter."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

from src.core import Document, ExtractedElement, ExtractionEngine
from src.preprocessing import PreprocessedDocument, PreprocessedPage


class TesseractExtractionAdapter(ExtractionEngine):
    """Thin Tesseract adapter that returns raw extracted elements."""

    def __init__(
        self,
        *,
        language: str = "por",
        config: str = "",
    ) -> None:
        self.language = language
        self.config = config

    def extract(self, document: Document) -> list[ExtractedElement]:
        from PIL import Image
        import pytesseract

        path = self._document_path(document)
        media_type = self._resolve_media_type(document, path)
        if media_type == "application/pdf":
            raise ValueError(
                "TesseractExtractionAdapter only accepts image documents directly. "
                "Convert PDFs to page images during preprocessing first."
            )
        if media_type is None or not media_type.startswith("image/"):
            raise ValueError(f"Unsupported media_type for Tesseract extraction: {media_type!r}")

        with Image.open(path) as image:
            raw = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

        return self._build_elements(document, raw)

    def extract_preprocessed(
        self,
        preprocessed_document: PreprocessedDocument,
    ) -> list[ExtractedElement]:
        """Extract raw elements from preprocessed page images."""
        import pytesseract

        elements: list[ExtractedElement] = []
        for page in preprocessed_document.pages:
            raw = pytesseract.image_to_data(
                page.image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
            elements.extend(
                self._build_elements(
                    preprocessed_document.document,
                    raw,
                    preprocessed_page=page,
                )
            )
        return elements

    @staticmethod
    def _resolve_media_type(document: Document, path: Path) -> str | None:
        if document.media_type:
            return document.media_type

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix in {".tif", ".tiff"}:
            return "image/tiff"
        return None

    def _document_path(self, document: Document) -> Path:
        if not document.source_uri:
            raise ValueError("Document source_uri is required for extraction.")

        parsed = urlparse(document.source_uri)
        if parsed.scheme not in ("", "file"):
            raise ValueError(f"Unsupported document source URI scheme: {parsed.scheme}")

        raw_path = (
            url2pathname(unquote(parsed.path))
            if parsed.scheme == "file"
            else document.source_uri
        )
        if parsed.scheme == "file" and parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"

        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document path does not exist: {path}")
        return path

    def _build_elements(
        self,
        document: Document,
        raw: dict[str, list],
        *,
        preprocessed_page: PreprocessedPage | None = None,
    ) -> list[ExtractedElement]:
        element_count = len(raw.get("text", []))
        elements: list[ExtractedElement] = []

        for index in range(element_count):
            text = str(raw["text"][index]).strip()
            if not text:
                continue

            confidence = self._parse_confidence(raw.get("conf", [None])[index])
            page_number = self._parse_optional_int(raw.get("page_num", [None])[index])
            if page_number is None and preprocessed_page is not None:
                page_number = preprocessed_page.page_number
            left = self._parse_optional_float(raw.get("left", [None])[index])
            top = self._parse_optional_float(raw.get("top", [None])[index])
            width = self._parse_optional_float(raw.get("width", [None])[index])
            height = self._parse_optional_float(raw.get("height", [None])[index])

            bounding_box = None
            if None not in (left, top, width, height):
                bounding_box = (left, top, width, height)

            elements.append(
                ExtractedElement(
                    element_id=f"{document.document_id}:tesseract:{index}",
                    element_type="text",
                    text=text,
                    page_number=page_number,
                    bounding_box=bounding_box,
                    confidence=confidence,
                    metadata={
                        "source_engine": "tesseract",
                        "language": self.language,
                        "block_num": self._parse_optional_int(raw.get("block_num", [None])[index]),
                        "line_num": self._parse_optional_int(raw.get("line_num", [None])[index]),
                        "word_num": self._parse_optional_int(raw.get("word_num", [None])[index]),
                        "preprocessing_metadata": preprocessed_page.metadata if preprocessed_page else {},
                    },
                )
            )

        return elements

    @staticmethod
    def _parse_confidence(value: object) -> float | None:
        parsed = TesseractExtractionAdapter._parse_optional_float(value)
        if parsed is None or parsed < 0:
            return None
        return parsed / 100.0

    @staticmethod
    def _parse_optional_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _parse_optional_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        return float(value)
