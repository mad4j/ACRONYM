"""Tests for acronym.model – AcronymModel."""

import os
import tempfile

import numpy as np
import pytest

from acronym.model import AcronymModel


# ---------------------------------------------------------------------------
# Minimal training fixtures
# ---------------------------------------------------------------------------

POSITIVE_EN = [
    ("NATO", "North Atlantic Treaty Organization", "before"),
    ("AI", "Artificial Intelligence", "before"),
    ("CPU", "Central Processing Unit", "before"),
    ("WHO", "World Health Organization", "before"),
    ("PDF", "Portable Document Format", "before"),
    ("API", "Application Programming Interface", "before"),
    ("ML", "Machine Learning", "before"),
    ("NLP", "Natural Language Processing", "before"),
]

NEGATIVE_EN = [
    ("IT", "Italy", "before"),
    ("EX", "a former employee of the company", "before"),
    ("ON", "based on the agreed premise stated above", "before"),
    ("BY", "located by the river in the western district", "before"),
    ("NO", "this is not applicable to the current case", "before"),
]


def _make_samples_labels():
    samples = POSITIVE_EN + NEGATIVE_EN
    labels = [1] * len(POSITIVE_EN) + [0] * len(NEGATIVE_EN)
    return samples, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAcronymModel:

    def test_init_default_lang(self):
        m = AcronymModel()
        assert m.lang == "en"

    def test_init_custom_lang(self):
        m = AcronymModel(lang="it")
        assert m.lang == "it"

    def test_not_trained_initially(self):
        m = AcronymModel()
        assert not m.is_trained()

    def test_trained_after_fit(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        assert m.is_trained()

    def test_predict_returns_binary(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        preds = m.predict(samples)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_in_range(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        probas = m.predict_proba(samples)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_proba_length(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        probas = m.predict_proba(samples)
        assert len(probas) == len(samples)

    def test_clear_acronym_scored_high(self):
        """A textbook acronym like NATO should score above 0.5."""
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        p = m.predict_proba([("NATO", "North Atlantic Treaty Organization", "before")])
        assert p[0] >= 0.5

    def test_noise_scored_low(self):
        """Obvious noise should score lower than a valid acronym."""
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)
        p_valid = m.predict_proba([("API", "Application Programming Interface", "before")])
        p_noise = m.predict_proba([("ON", "based on the agreed premise stated above", "before")])
        assert p_valid[0] > p_noise[0]

    def test_train_empty_samples_raises(self):
        m = AcronymModel(lang="en")
        with pytest.raises(ValueError):
            m.train([], [])

    def test_train_mismatched_lengths_raises(self):
        m = AcronymModel(lang="en")
        with pytest.raises(ValueError):
            m.train([("A", "Alpha", "before")], [1, 0])

    def test_predict_before_train_raises(self):
        m = AcronymModel(lang="en")
        with pytest.raises(RuntimeError):
            m.predict([("NATO", "North Atlantic Treaty Organization", "before")])

    def test_predict_proba_before_train_raises(self):
        m = AcronymModel(lang="en")
        with pytest.raises(RuntimeError):
            m.predict_proba([("NATO", "North Atlantic Treaty Organization", "before")])

    def test_save_and_load(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_en.pkl")
            m.save(path)
            assert os.path.exists(path)

            loaded = AcronymModel.load(path)
            assert loaded.lang == "en"
            assert loaded.is_trained()

            original_probas = m.predict_proba(samples)
            loaded_probas = loaded.predict_proba(samples)
            np.testing.assert_array_almost_equal(original_probas, loaded_probas)

    def test_save_creates_parent_dirs(self):
        samples, labels = _make_samples_labels()
        m = AcronymModel(lang="en")
        m.train(samples, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "model_en.pkl")
            m.save(path)
            assert os.path.exists(path)
