"""Micro AI model for classifying acronym-definition candidate pairs.

The model is a scikit-learn ``Pipeline`` that normalises features with
:class:`~sklearn.preprocessing.StandardScaler` and classifies them with
:class:`~sklearn.linear_model.LogisticRegression`.  One model instance is
trained per language.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features

# (acronym, definition, pattern_type)
Sample = Tuple[str, str, str]


class AcronymModel:
    """Language-aware binary classifier for acronym-definition pairs.

    Args:
        lang:       Language code the model is trained for (``"en"`` or ``"it"``).
        warm_start: When ``True`` the underlying :class:`~sklearn.linear_model.LogisticRegression`
                    will reuse the coefficients from the previous :meth:`train` call as the
                    starting point for the next optimisation.  This allows incremental
                    training: pass ``warm_start=True`` together with a pre-loaded model in
                    :func:`~acronym.trainer.train_from_samples` to continue training an
                    existing model instead of starting from scratch.
    """

    def __init__(self, lang: str = "en", warm_start: bool = False) -> None:
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
                        warm_start=warm_start,
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_X(self, samples: List[Sample]) -> np.ndarray:
        return np.array(
            [extract_features(acr, defn, self.lang, pt) for acr, defn, pt in samples]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, samples: List[Sample], labels: List[int]) -> "AcronymModel":
        """Fit the model on labelled samples.

        Args:
            samples: List of ``(acronym, definition, pattern_type)`` tuples.
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
