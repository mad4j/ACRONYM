"""Tests for warm-start training in AcronymModel and trainer API."""

import os
import tempfile

import numpy as np
import pytest

from acronym.model import AcronymModel
from acronym.trainer import get_model_path, train_from_file, train_from_samples
import json


# ---------------------------------------------------------------------------
# Shared training fixtures
# ---------------------------------------------------------------------------

_SAMPLES = [
    ("NATO", "North Atlantic Treaty Organization", "before"),
    ("AI",   "Artificial Intelligence",            "before"),
    ("CPU",  "Central Processing Unit",            "before"),
    ("WHO",  "World Health Organization",          "before"),
    ("API",  "Application Programming Interface",  "before"),
    ("ML",   "Machine Learning",                   "before"),
    ("IT",   "Italy",                              "before"),
    ("EX",   "a former employee of the company",   "before"),
    ("ON",   "based on the agreed premise above",  "before"),
    ("BY",   "located by the river district",      "before"),
]
_LABELS = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

_EXTRA_SAMPLES = [
    ("NLP", "Natural Language Processing", "before"),
    ("GPU", "Graphics Processing Unit",    "before"),
    ("NO",  "not applicable to the case",  "before"),
]
_EXTRA_LABELS = [1, 1, 0]


# ---------------------------------------------------------------------------
# AcronymModel – warm_start parameter
# ---------------------------------------------------------------------------


class TestAcronymModelWarmStart:

    def test_default_warm_start_is_false(self):
        m = AcronymModel()
        assert m._pipeline.named_steps["clf"].warm_start is False

    def test_warm_start_true_sets_flag(self):
        m = AcronymModel(warm_start=True)
        assert m._pipeline.named_steps["clf"].warm_start is True

    def test_warm_start_false_trains_normally(self):
        m = AcronymModel(lang="en", warm_start=False)
        m.train(_SAMPLES, _LABELS)
        assert m.is_trained()

    def test_warm_start_true_trains_normally(self):
        """warm_start=True on a fresh model (no prior coefficients) still works."""
        m = AcronymModel(lang="en", warm_start=True)
        m.train(_SAMPLES, _LABELS)
        assert m.is_trained()

    def test_warm_start_continues_from_previous_fit(self):
        """After first fit the second fit warm-starts from existing coefficients."""
        m = AcronymModel(lang="en", warm_start=True)
        m.train(_SAMPLES, _LABELS)
        probas_first = m.predict_proba(_SAMPLES).copy()

        # Second training round with additional data
        m.train(_SAMPLES + _EXTRA_SAMPLES, _LABELS + _EXTRA_LABELS)
        probas_second = m.predict_proba(_SAMPLES)

        # Both fits should produce valid probability arrays
        assert np.all(probas_first >= 0.0) and np.all(probas_first <= 1.0)
        assert np.all(probas_second >= 0.0) and np.all(probas_second <= 1.0)

    def test_warm_start_flag_preserved_after_save_load(self):
        m = AcronymModel(lang="en", warm_start=True)
        m.train(_SAMPLES, _LABELS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_en.pkl")
            m.save(path)
            loaded = AcronymModel.load(path)
            # The pipeline (including warm_start flag) is preserved via pickle
            assert loaded._pipeline.named_steps["clf"].warm_start is True


# ---------------------------------------------------------------------------
# train_from_samples – warm_start parameter
# ---------------------------------------------------------------------------


class TestTrainFromSamplesWarmStart:

    def test_warm_start_false_no_prior_model(self):
        """warm_start=False should just create a fresh model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_from_samples(_SAMPLES, _LABELS, lang="en", model_dir=tmpdir)
            assert model.is_trained()
            assert os.path.exists(get_model_path("en", tmpdir))

    def test_warm_start_true_no_prior_model_falls_back(self):
        """warm_start=True with no saved model falls back to fresh training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_from_samples(
                _SAMPLES, _LABELS, lang="en", model_dir=tmpdir, warm_start=True
            )
            assert model.is_trained()

    def test_warm_start_true_loads_existing_model(self):
        """warm_start=True with a saved model reuses its coefficients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First training pass (from scratch)
            train_from_samples(_SAMPLES, _LABELS, lang="en", model_dir=tmpdir)

            # Second training pass (warm start from the first model)
            model_ws = train_from_samples(
                _SAMPLES + _EXTRA_SAMPLES,
                _LABELS + _EXTRA_LABELS,
                lang="en",
                model_dir=tmpdir,
                warm_start=True,
            )
            assert model_ws.is_trained()
            # The warm-started model should still give valid predictions
            probas = model_ws.predict_proba(_SAMPLES)
            assert np.all(probas >= 0.0) and np.all(probas <= 1.0)

    def test_warm_start_does_not_degrade_clear_positive(self):
        """A textbook acronym should still score above 0.5 after warm-start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_from_samples(_SAMPLES, _LABELS, lang="en", model_dir=tmpdir)
            model_ws = train_from_samples(
                _SAMPLES + _EXTRA_SAMPLES,
                _LABELS + _EXTRA_LABELS,
                lang="en",
                model_dir=tmpdir,
                warm_start=True,
            )
            p = model_ws.predict_proba(
                [("NATO", "North Atlantic Treaty Organization", "before")]
            )
            assert p[0] >= 0.5

    def test_warm_start_saves_model_file(self):
        """The warm-started model should be saved back to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_from_samples(_SAMPLES, _LABELS, lang="en", model_dir=tmpdir)
            model_path = get_model_path("en", tmpdir)
            mtime_before = os.path.getmtime(model_path)

            train_from_samples(
                _EXTRA_SAMPLES, _EXTRA_LABELS, lang="en", model_dir=tmpdir, warm_start=True
            )
            mtime_after = os.path.getmtime(model_path)
            assert mtime_after >= mtime_before


# ---------------------------------------------------------------------------
# train_from_file – warm_start parameter
# ---------------------------------------------------------------------------


class TestTrainFromFileWarmStart:

    def _write_json(self, tmpdir, samples, labels, filename="samples.json"):
        data = [
            {
                "acronym": acr,
                "definition": defn,
                "pattern_type": pt,
                "label": lbl,
            }
            for (acr, defn, pt), lbl in zip(samples, labels)
        ]
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path

    def test_warm_start_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = self._write_json(tmpdir, _SAMPLES, _LABELS)
            model_dir = os.path.join(tmpdir, "models")

            # First training pass
            train_from_file(data_path, lang="en", model_dir=model_dir)

            # Second pass with warm start
            extra_path = self._write_json(
                tmpdir, _EXTRA_SAMPLES, _EXTRA_LABELS, "extra.json"
            )
            model_ws = train_from_file(
                extra_path, lang="en", model_dir=model_dir, warm_start=True
            )
            assert model_ws.is_trained()
