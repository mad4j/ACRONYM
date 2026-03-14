"""Micro AI model for classifying acronym-definition candidate pairs.

The model is a scikit-learn ``Pipeline`` that normalises features with
:class:`~sklearn.preprocessing.StandardScaler` and classifies them with
:class:`~sklearn.linear_model.LogisticRegression`.  One model instance is
trained per language.

The classifier uses ``warm_start=True`` so that :meth:`AcronymModel.update`
can incrementally refine an already-trained model without discarding the
knowledge embedded in its existing weights.
"""

import os
import pickle
from typing import List, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features

# (acronym, definition, pattern_type) – legacy 3-tuple
# (acronym, definition, pattern_type, context) – context-aware 4-tuple
Sample = Union[Tuple[str, str, str], Tuple[str, str, str, str]]


class AcronymModel:
    """Language-aware binary classifier for acronym-definition pairs.

    Args:
        lang: Language code the model is trained for (``"en"`` or ``"it"``).
    """

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._trained = False
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        class_weight="balanced",
                        warm_start=True,
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_X(self, samples: List[Sample]) -> np.ndarray:
        rows = []
        for item in samples:
            # Accept both 3-tuple (acronym, definition, pattern_type) and
            # 4-tuple (acronym, definition, pattern_type, context).
            if len(item) == 4:
                acr, defn, pt, ctx = item[0], item[1], item[2], item[3]
            else:
                acr, defn, pt, ctx = item[0], item[1], item[2], ""
            rows.append(extract_features(acr, defn, self.lang, pt, ctx))
        return np.array(rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, samples: List[Sample], labels: List[int]) -> "AcronymModel":
        """Fit the model on labelled samples.

        Args:
            samples: List of ``(acronym, definition, pattern_type)`` tuples or
                     ``(acronym, definition, pattern_type, context)`` 4-tuples.
            labels:  Parallel list of integer labels (``1`` = valid pair,
                     ``0`` = invalid / noise).

        Returns:
            ``self`` (for chaining).
        """
        if not samples:
            raise ValueError("samples must not be empty")
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have the same length")
        X = self._build_X(samples)
        self._pipeline.fit(X, labels)
        self._trained = True
        return self

    def update(self, samples: List[Sample], labels: List[int]) -> "AcronymModel":
        """Incrementally update the model with new labelled samples.

        When the model has already been trained this method performs a
        **warm-start** fit: the existing ``LogisticRegression`` coefficients
        are used as initial values and are refined on the combined knowledge
        brought by *samples*.  If the model has not been trained yet, this
        method falls back to :meth:`train`.

        Args:
            samples: New ``(acronym, definition, pattern_type[, context])``
                     tuples to learn from.
            labels:  Parallel list of integer labels.

        Returns:
            ``self`` (for chaining).
        """
        if not self._trained:
            return self.train(samples, labels)
        if not samples:
            raise ValueError("samples must not be empty")
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have the same length")
        X = self._build_X(samples)
        # warm_start=True (set at construction) reuses the previous coef_ as
        # the initial estimate for the next fit() call.
        self._pipeline.fit(X, labels)
        return self

    def predict(self, samples: List[Sample]) -> np.ndarray:
        """Return binary predictions (0 or 1) for *samples*.

        Raises:
            RuntimeError: if the model has not been trained yet.
        """
        self._check_trained()
        X = self._build_X(samples)
        return self._pipeline.predict(X)

    def predict_proba(self, samples: List[Sample]) -> np.ndarray:
        """Return the probability of class ``1`` for each sample.

        Raises:
            RuntimeError: if the model has not been trained yet.
        """
        self._check_trained()
        X = self._build_X(samples)
        return self._pipeline.predict_proba(X)[:, 1]

    def is_trained(self) -> bool:
        """Return ``True`` if the model has been fitted."""
        return self._trained

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the model to *path* using :mod:`pickle`.

        Parent directories are created automatically.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"pipeline": self._pipeline, "lang": self.lang, "trained": self._trained},
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "AcronymModel":
        """Deserialise a model previously saved with :meth:`save`.

        Args:
            path: Filesystem path to the pickle file.

        Returns:
            Loaded :class:`AcronymModel` instance.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(lang=data["lang"])
        obj._pipeline = data["pipeline"]
        obj._trained = data["trained"]
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. "
                "Call train() or load a pre-trained model first."
            )
