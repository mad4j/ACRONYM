"""ACRONYM – Micro AI for extracting acronyms and definitions from .docx files.

Supports language-specific training and detection for English (en) and Italian (it).

Quick start:
    from acronym.trainer import train_from_file
    from acronym.detector import detect_acronyms

    # Train a model
    train_from_file("data/en/samples.json", lang="en")

    # Detect acronyms in a document
    results = detect_acronyms("document.docx", lang="en")
    for r in results:
        print(r["acronym"], "→", r["definition"])

    # Detect standalone ALL-CAPS acronyms without explicit definitions
    from acronym.detector import detect_standalone_acronyms_from_text
    results = detect_standalone_acronyms_from_text("The CPU handles all computation.", lang="en")
"""

from .detector import (
    detect_acronyms,
    detect_acronyms_from_text,
    detect_standalone_acronyms_from_text,
)
from .model import AcronymModel
from .trainer import train_from_file, train_from_samples

__all__ = [
    "AcronymModel",
    "detect_acronyms",
    "detect_acronyms_from_text",
    "detect_standalone_acronyms_from_text",
    "train_from_file",
    "train_from_samples",
]
