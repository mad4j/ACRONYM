"""Detection API – extract acronyms and definitions from .docx files or text.

A trained :class:`~acronym.model.AcronymModel` is required.  Use
:func:`~acronym.trainer.train_from_file` or
:func:`~acronym.trainer.train_from_samples` to create one first.
"""

import os
from typing import Dict, List, Optional

from .model import AcronymModel
from .patterns import find_candidates
from .reader import read_docx
from .trainer import DEFAULT_MODEL_DIR, get_model_path

#: Default probability threshold above which a candidate is accepted.
DEFAULT_THRESHOLD: float = 0.5

# Result dict keys: "acronym", "definition", "confidence"
Result = Dict[str, object]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(lang: str, model_dir: Optional[str] = None) -> AcronymModel:
    """Load the trained model for *lang* from disk.

    Args:
        lang:      Language code (``"en"`` or ``"it"``).
        model_dir: Override for the models directory.

    Raises:
        FileNotFoundError: if no model has been trained for *lang* yet.
    """
    model_path = get_model_path(lang, model_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for language '{lang}' at '{model_path}'. "
            "Train a model first with: acronym train <data.json> --lang "
            + lang
        )
    return AcronymModel.load(model_path)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------


def _score_and_filter(
    candidates: list,
    model: AcronymModel,
    threshold: float,
) -> List[Result]:
    """Run *model* on *candidates* and return those above *threshold*."""
    if not candidates:
        return []

    probas = model.predict_proba(candidates)

    results: List[Result] = []
    seen: set = set()

    for (acronym, definition, _pattern_type), confidence in zip(candidates, probas):
        if confidence >= threshold:
            key = (acronym.upper(), definition.lower())
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "acronym": acronym,
                        "definition": definition,
                        "confidence": round(float(confidence), 3),
                    }
                )

    results.sort(key=lambda x: str(x["acronym"]))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_acronyms(
    docx_path: str,
    lang: str = "en",
    model: Optional[AcronymModel] = None,
    model_dir: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Result]:
    """Detect acronyms and their definitions in a *.docx* file.

    Args:
        docx_path:  Path to the ``.docx`` document.
        lang:       Language of the document (``"en"`` or ``"it"``).
        model:      Pre-loaded :class:`~acronym.model.AcronymModel`.  When
                    ``None`` the default model for *lang* is loaded from disk.
        model_dir:  Override for the directory that contains trained models.
        threshold:  Minimum confidence score ``[0, 1]`` for a candidate to be
                    included in the output.  Defaults to ``0.5``.

    Returns:
        List of result dicts, each with keys ``"acronym"``, ``"definition"``,
        and ``"confidence"`` (float in ``[0, 1]``), sorted alphabetically by
        acronym.

    Raises:
        FileNotFoundError: if *docx_path* does not exist or no model is found.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Document not found: {docx_path}")

    if model is None:
        model = load_model(lang, model_dir)

    text = read_docx(docx_path)
    candidates = find_candidates(text, lang)
    return _score_and_filter(candidates, model, threshold)


def detect_acronyms_from_text(
    text: str,
    lang: str = "en",
    model: Optional[AcronymModel] = None,
    model_dir: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Result]:
    """Detect acronyms and definitions in a plain-text string.

    Args:
        text:       Input text.
        lang:       Language of the text (``"en"`` or ``"it"``).
        model:      Pre-loaded model.  When ``None`` the default model for
                    *lang* is loaded from disk.
        model_dir:  Override for the models directory.
        threshold:  Confidence threshold.

    Returns:
        Same format as :func:`detect_acronyms`.
    """
    if model is None:
        model = load_model(lang, model_dir)

    candidates = find_candidates(text, lang)
    return _score_and_filter(candidates, model, threshold)
