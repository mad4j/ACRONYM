"""Training API for language-specific acronym-detection models.

Training data is stored as JSON files with the following schema::

    [
        {
            "acronym":      "NATO",
            "definition":   "North Atlantic Treaty Organization",
            "pattern_type": "before",
            "label":        1
        },
        ...
    ]

``label`` is ``1`` for a valid acronym-definition pair and ``0`` for a
negative / noise example.  ``pattern_type`` is ``"before"`` when the acronym
precedes its definition in parentheses and ``"after"`` when it follows.

Pass a previously trained :class:`~acronym.model.AcronymModel` as
``base_model`` to :func:`train_from_samples` or :func:`train_from_file` to
**warm-start** the new training run from that model's existing weights.  This
allows the model to build on already-learned knowledge rather than discarding
it and starting from scratch.
"""

import json
import os
from typing import List, Optional, Tuple, Union

from .model import AcronymModel

# (acronym, definition, pattern_type) or (acronym, definition, pattern_type, context)
Sample = Union[Tuple[str, str, str], Tuple[str, str, str, str]]

# Default directories (relative to the project root, resolved at runtime)
_PKG_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PKG_DIR)
DEFAULT_MODEL_DIR: str = os.path.join(_PROJECT_ROOT, "models")
DEFAULT_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_path(lang: str, model_dir: Optional[str] = None) -> str:
    """Return the canonical path for the serialised model of *lang*.

    Args:
        lang:       Language code (e.g. ``"en"`` or ``"it"``).
        model_dir:  Override for the models directory.
    """
    directory = model_dir or DEFAULT_MODEL_DIR
    return os.path.join(directory, f"model_{lang}.pkl")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(data_path: str) -> Tuple[List[Sample], List[int]]:
    """Load labelled training examples from a JSON file.

    Args:
        data_path: Path to the JSON file.

    Returns:
        Tuple of ``(samples, labels)`` where *samples* is a list of
        ``(acronym, definition, pattern_type)`` tuples and *labels* is a
        parallel list of integers (``1`` / ``0``).

    Raises:
        FileNotFoundError: if *data_path* does not exist.
        ValueError:        if the JSON structure is invalid.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Training data JSON must be a top-level list of objects.")

    samples: List[Sample] = []
    labels: List[int] = []

    for i, item in enumerate(data):
        try:
            acronym = str(item["acronym"])
            definition = str(item["definition"])
            pattern_type = str(item.get("pattern_type", "before"))
            label = int(item["label"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid record at index {i}: {exc}") from exc
        samples.append((acronym, definition, pattern_type))
        labels.append(label)

    return samples, labels


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_from_file(
    data_path: str,
    lang: str,
    model_dir: Optional[str] = None,
    base_model: Optional[AcronymModel] = None,
) -> AcronymModel:
    """Train a model from a JSON data file and persist it to disk.

    Args:
        data_path:   Path to the training data JSON file.
        lang:        Language code (``"en"`` or ``"it"``).
        model_dir:   Directory where the model file is saved.  Defaults to
                     ``models/`` in the project root.
        base_model:  Optional pre-trained :class:`~acronym.model.AcronymModel`
                     to warm-start from.  When provided the new training run
                     begins from the existing model weights rather than from
                     random initialisation.

    Returns:
        The trained :class:`~acronym.model.AcronymModel`.
    """
    samples, labels = load_training_data(data_path)
    return train_from_samples(
        samples, labels, lang=lang, model_dir=model_dir, base_model=base_model
    )


def train_from_samples(
    samples: List[Sample],
    labels: List[int],
    lang: str,
    model_dir: Optional[str] = None,
    base_model: Optional[AcronymModel] = None,
) -> AcronymModel:
    """Train a model from in-memory samples and persist it to disk.

    Args:
        samples:    List of ``(acronym, definition, pattern_type[, context])``
                    tuples.
        labels:     Parallel list of integer labels (``1`` / ``0``).
        lang:       Language code (``"en"`` or ``"it"``).
        model_dir:  Directory where the model file is saved.
        base_model: Optional pre-trained :class:`~acronym.model.AcronymModel`
                    to warm-start from.  When provided, :meth:`~AcronymModel.update`
                    is called on the base model so that previous weights are
                    used as the starting point for optimisation.

    Returns:
        The trained :class:`~acronym.model.AcronymModel`.
    """
    if base_model is not None and base_model.is_trained():
        model = base_model
        model.update(samples, labels)
    else:
        model = AcronymModel(lang=lang)
        model.train(samples, labels)

    model_path = get_model_path(lang, model_dir)
    model.save(model_path)
    print(f"[acronym] Model for '{lang}' trained on {len(samples)} samples → {model_path}")
    return model
