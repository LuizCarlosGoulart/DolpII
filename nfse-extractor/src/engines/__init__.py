"""OCR engine adapters for Tesseract and Dolphin."""

from .dolphin_adapter import DolphinExtractionAdapter
from .dolphin_runtime import load_dolphin_runtime
from .tesseract_adapter import TesseractExtractionAdapter

__all__ = ["DolphinExtractionAdapter", "load_dolphin_runtime", "TesseractExtractionAdapter"]
