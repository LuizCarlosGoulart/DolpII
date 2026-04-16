"""Dolphin-based extraction adapter."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

from src.core import Document, ExtractedElement, ExtractionEngine
from src.preprocessing import PreprocessedDocument, PreprocessedPage


class DolphinExtractionAdapter(ExtractionEngine):
    """Thin adapter that normalizes Dolphin output into shared raw elements."""

    def __init__(
        self,
        *,
        predictor: Callable[[object], object] | None = None,
        runtime_factory: Callable[..., Callable[[object], object]] | None = None,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._predictor = predictor
        self._runtime_factory = runtime_factory
        self.model_path = model_path
        self.device = device

    def extract(self, document: Document) -> list[ExtractedElement]:
        from PIL import Image

        path = self._document_path(document)
        media_type = self._resolve_media_type(document, path)
        if media_type == "application/pdf":
            raise ValueError(
                "DolphinExtractionAdapter only accepts image documents directly. "
                "Convert PDFs to page images during preprocessing first."
            )
        if media_type is None or not media_type.startswith("image/"):
            raise ValueError(f"Unsupported media_type for Dolphin extraction: {media_type!r}")

        with Image.open(path) as image:
            raw_output = self._predict(image)

        return self._build_elements(document, raw_output)

    def extract_preprocessed(
        self,
        preprocessed_document: PreprocessedDocument,
    ) -> list[ExtractedElement]:
        """Extract raw elements from preprocessed page images."""
        elements: list[ExtractedElement] = []
        for page in preprocessed_document.pages:
            raw_output = self._predict(page.image)
            elements.extend(
                self._build_elements(
                    preprocessed_document.document,
                    raw_output,
                    preprocessed_page=page,
                )
            )
        return elements

    def _predict(self, image: object) -> object:
        predictor = self._get_predictor()
        return predictor(image)

    def _get_predictor(self) -> Callable[[object], object]:
        if self._predictor is not None:
            return self._predictor
        if self._runtime_factory is None:
            raise ValueError(
                "Configure a Dolphin predictor or runtime_factory before extraction."
            )
        signature = inspect.signature(self._runtime_factory)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepts_named_args = {"model_path", "device"}.issubset(signature.parameters)

        if accepts_kwargs or accepts_named_args:
            self._predictor = self._runtime_factory(
                model_path=self.model_path,
                device=self.device,
            )
        else:
            self._predictor = self._runtime_factory()
        return self._predictor

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
        raw_output: object,
        *,
        preprocessed_page: PreprocessedPage | None = None,
    ) -> list[ExtractedElement]:
        items = self._normalize_output_items(raw_output)
        elements: list[ExtractedElement] = []

        for index, item in enumerate(items):
            text = self._extract_text(item).strip()
            if not text:
                continue

            page_number = self._extract_page_number(item)
            if page_number is None and preprocessed_page is not None:
                page_number = preprocessed_page.page_number

            elements.append(
                ExtractedElement(
                    element_id=f"{document.document_id}:dolphin:{index}",
                    element_type=self._extract_element_type(item),
                    text=text,
                    page_number=page_number,
                    bounding_box=self._extract_bounding_box(item),
                    confidence=self._extract_confidence(item),
                    metadata={
                        "source_engine": "dolphin",
                        "model_path": self.model_path,
                        "device": self.device,
                        "raw_label": item.get("label") or item.get("type"),
                        "preprocessing_metadata": preprocessed_page.metadata if preprocessed_page else {},
                    },
                )
            )

        return elements

    @staticmethod
    def _normalize_output_items(raw_output: object) -> list[dict]:
        if isinstance(raw_output, str):
            return [{"text": raw_output}]
        if isinstance(raw_output, list):
            items: list[dict] = []
            for item in raw_output:
                if isinstance(item, dict):
                    items.append(item)
                elif isinstance(item, str):
                    items.append({"text": item})
            return items
        if isinstance(raw_output, dict):
            for key in ("elements", "predictions", "tokens", "items"):
                value = raw_output.get(key)
                if isinstance(value, list):
                    return DolphinExtractionAdapter._normalize_output_items(value)
            return [raw_output]
        raise ValueError(f"Unsupported Dolphin raw output type: {type(raw_output)!r}")

    @staticmethod
    def _extract_text(item: dict) -> str:
        for key in ("text", "value", "content", "raw_text"):
            value = item.get(key)
            if value is not None:
                return str(value)
        return ""

    @staticmethod
    def _extract_confidence(item: dict) -> float | None:
        for key in ("confidence", "score", "probability"):
            value = item.get(key)
            if value in (None, ""):
                continue
            parsed = float(value)
            return parsed if parsed <= 1 else parsed / 100.0
        return None

    @staticmethod
    def _extract_page_number(item: dict) -> int | None:
        for key in ("page_number", "page", "page_num"):
            value = item.get(key)
            if value in (None, ""):
                continue
            return int(value)
        return None

    @staticmethod
    def _extract_bounding_box(item: dict) -> tuple[float, float, float, float] | None:
        for key in ("bounding_box", "bbox", "box"):
            value = item.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                return tuple(float(part) for part in value)
        return None

    @staticmethod
    def _extract_element_type(item: dict) -> str:
        return str(item.get("element_type") or item.get("type") or item.get("label") or "text")
