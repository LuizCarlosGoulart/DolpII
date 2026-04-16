from pathlib import Path

import pytest

from src.ingestion import load_document, load_documents


def test_load_document_builds_canonical_document_from_pdf(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    document = load_document(file_path)

    assert document.document_id
    assert document.source_uri == file_path.resolve().as_uri()
    assert document.media_type == "application/pdf"
    assert document.metadata["file_name"] == "sample.pdf"
    assert document.metadata["file_extension"] == ".pdf"
    assert document.metadata["file_size_bytes"] == len(b"%PDF-1.4\n")


def test_load_document_accepts_image_extensions(tmp_path: Path) -> None:
    file_path = tmp_path / "note.png"
    file_path.write_bytes(b"fake-image")

    document = load_document(file_path)

    assert document.media_type == "image/png"


def test_load_document_rejects_missing_or_unsupported_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pdf"
    unsupported_path = tmp_path / "note.txt"
    unsupported_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_document(missing_path)

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(unsupported_path)


def test_load_documents_preserves_input_order(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"%PDF-1.4\n")
    second.write_bytes(b"fake-jpeg")

    documents = load_documents([first, second])

    assert [document.metadata["file_name"] for document in documents] == ["a.pdf", "b.jpg"]


def test_load_document_assigns_stable_document_id_for_same_path(tmp_path: Path) -> None:
    file_path = tmp_path / "stable.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    first = load_document(file_path)
    second = load_document(file_path)

    assert first.document_id == second.document_id
