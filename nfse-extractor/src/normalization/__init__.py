"""Normalization from raw OCR output into structured fields."""

from .output_normalizer import ConfigDrivenOutputNormalizer
from .raw_output import NormalizedRawArtifact, normalize_raw_elements

__all__ = [
    "ConfigDrivenOutputNormalizer",
    "NormalizedRawArtifact",
    "normalize_raw_elements",
]
