"""Document ingestion utilities for local files."""

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from src.core import Document


SUPPORTED_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _normalize_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Document path must point to a file: {path}")
    return path


def _detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = SUPPORTED_MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported document type: {suffix or '<no extension>'}")
    return media_type


def load_document(file_path: str | Path) -> Document:
    """Load a local image or PDF into the canonical document contract."""
    path = _normalize_path(file_path)
    media_type = _detect_media_type(path)

    return Document(
        document_id=str(uuid5(NAMESPACE_URL, path.as_uri())),
        source_uri=path.as_uri(),
        media_type=media_type,
        metadata={
            "file_name": path.name,
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "file_extension": path.suffix.lower(),
        },
    )


def load_documents(file_paths: list[str | Path]) -> list[Document]:
    """Load multiple local documents preserving input order."""
    return [load_document(file_path) for file_path in file_paths]
