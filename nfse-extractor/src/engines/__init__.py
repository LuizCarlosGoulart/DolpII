"""OCR engine adapters for Tesseract and Dolphin."""

from .dolphin_adapter import DolphinExtractionAdapter
from .tesseract_adapter import TesseractExtractionAdapter

__all__ = ["DolphinExtractionAdapter", "TesseractExtractionAdapter"]
