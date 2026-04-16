"""Shared preprocessing pipeline for document images."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

from src.core import Document

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class PreprocessedPage:
    """One page or image prepared for downstream OCR engines."""

    page_number: int
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedDocument:
    """Shared preprocessing result for downstream extraction engines."""

    document: Document
    pages: list[PreprocessedPage]
    metadata: dict[str, Any] = field(default_factory=dict)


class PdfToImageConverter(ABC):
    """Abstract PDF-to-image conversion interface."""

    @abstractmethod
    def convert(self, pdf_path: Path) -> list[Any]:
        """Convert a PDF file into page images."""


class ImageNormalizationHook(ABC):
    """Small, reusable image normalization step."""

    name: str

    @abstractmethod
    def apply(self, image: Any) -> Any:
        """Apply one normalization step and return the updated image."""


def _document_path(document: Document) -> Path:
    if not document.source_uri:
        raise ValueError("Document source_uri is required for preprocessing.")

    parsed = urlparse(document.source_uri)
    if parsed.scheme in ("", "file"):
        raw_path = (
            url2pathname(unquote(parsed.path))
            if parsed.scheme == "file"
            else document.source_uri
        )
        if parsed.scheme == "file" and parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        return Path(raw_path).resolve()

    raise ValueError(f"Unsupported document source URI scheme: {parsed.scheme}")


def _load_image_from_path(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


def _resolve_media_type(document: Document, path: Path) -> str | None:
    if document.media_type:
        return document.media_type

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in IMAGE_SUFFIXES:
        return f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else 'tiff' if suffix in {'.tif', '.tiff'} else 'png'}"
    return None


def preprocess_document(
    document: Document,
    *,
    pdf_converter: PdfToImageConverter | None = None,
    normalization_hooks: list[ImageNormalizationHook] | None = None,
) -> PreprocessedDocument:
    """Prepare a document into normalized page images for downstream OCR."""
    path = _document_path(document)
    hooks = normalization_hooks or []
    media_type = _resolve_media_type(document, path)

    if media_type == "application/pdf":
        if pdf_converter is None:
            raise ValueError("A PDF converter is required for PDF preprocessing.")
        raw_images = pdf_converter.convert(path)
        source_kind = "pdf"
    elif media_type and media_type.startswith("image/"):
        raw_images = [_load_image_from_path(path)]
        source_kind = "image"
    else:
        raise ValueError(f"Unsupported media_type for preprocessing: {media_type!r}")

    pages: list[PreprocessedPage] = []
    for index, image in enumerate(raw_images, start=1):
        steps_applied: list[str] = []
        current_image = image
        for hook in hooks:
            current_image = hook.apply(current_image)
            steps_applied.append(hook.name)

        pages.append(
            PreprocessedPage(
                page_number=index,
                image=current_image,
                metadata={
                    "media_type": media_type,
                    "source_kind": source_kind,
                    "normalization_steps": steps_applied,
                },
            )
        )

    return PreprocessedDocument(
        document=document,
        pages=pages,
        metadata={
            "page_count": len(pages),
            "media_type": media_type,
            "source_kind": source_kind,
            "used_pdf_converter": pdf_converter is not None if source_kind == "pdf" else False,
            "normalization_hooks": [hook.name for hook in hooks],
        },
    )
