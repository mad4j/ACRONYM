"""Tests for acronym.trainer – training API."""

import json
import os
import tempfile

import pytest

from acronym.trainer import (
    get_model_path,
    load_training_data,
    train_from_file,
    train_from_samples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DATA = [
    {"acronym": "NATO", "definition": "North Atlantic Treaty Organization", "pattern_type": "before", "label": 1},
    {"acronym": "AI",   "definition": "Artificial Intelligence",            "pattern_type": "before", "label": 1},
    {"acronym": "CPU",  "definition": "Central Processing Unit",            "pattern_type": "before", "label": 1},
    {"acronym": "WHO",  "definition": "World Health Organization",          "pattern_type": "before", "label": 1},
    {"acronym": "API",  "definition": "Application Programming Interface",  "pattern_type": "before", "label": 1},
    {"acronym": "ML",   "definition": "Machine Learning",                   "pattern_type": "before", "label": 1},
    {"acronym": "IT",   "definition": "Italy",                              "pattern_type": "before", "label": 0},
    {"acronym": "EX",   "definition": "a former employee of the company",   "pattern_type": "before", "label": 0},
    {"acronym": "ON",   "definition": "based on the agreed premise above",  "pattern_type": "before", "label": 0},
    {"acronym": "BY",   "definition": "located by the river district",      "pattern_type": "before", "label": 0},
]


def _write_json(tmpdir, data, filename="samples.json"):
    path = os.path.join(tmpdir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


# ---------------------------------------------------------------------------
# Tests: get_model_path
# ---------------------------------------------------------------------------


class TestGetModelPath:

    def test_default_dir_contains_lang(self):
        path = get_model_path("en")
        assert "model_en.pkl" in path

    def test_custom_dir(self):
        path = get_model_path("it", model_dir="/tmp/mymodels")
        assert path == "/tmp/mymodels/model_it.pkl"


# ---------------------------------------------------------------------------
# Tests: load_training_data
# ---------------------------------------------------------------------------


class TestLoadTrainingData:

    def test_loads_samples_and_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(tmpdir, _VALID_DATA)
            samples, labels = load_training_data(path)
        assert len(samples) == len(_VALID_DATA)
        assert len(labels) == len(_VALID_DATA)

    def test_samples_are_triples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(tmpdir, _VALID_DATA)
            samples, _ = load_training_data(path)
        for s in samples:
            assert len(s) == 3

    def test_labels_are_ints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(tmpdir, _VALID_DATA)
            _, labels = load_training_data(path)
        assert all(isinstance(l, int) for l in labels)

    def test_default_pattern_type(self):
        data = [{"acronym": "AI", "definition": "Artificial Intelligence", "label": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(tmpdir, data)
            samples, _ = load_training_data(path)
        assert samples[0][2] == "before"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_training_data("/nonexistent/path/samples.json")

    def test_invalid_json_structure_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as fh:
                json.dump({"not": "a list"}, fh)
            with pytest.raises(ValueError):
                load_training_data(path)

    def test_missing_required_key_raises(self):
        bad = [{"acronym": "AI", "label": 1}]  # missing 'definition'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(tmpdir, bad)
            with pytest.raises(ValueError):
                load_training_data(path)


# ---------------------------------------------------------------------------
# Tests: train_from_file
# ---------------------------------------------------------------------------


class TestTrainFromFile:

    def test_trains_and_saves_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = _write_json(tmpdir, _VALID_DATA)
            model_dir = os.path.join(tmpdir, "models")
            model = train_from_file(data_path, lang="en", model_dir=model_dir)
            assert model.is_trained()
            assert os.path.exists(os.path.join(model_dir, "model_en.pkl"))

    def test_trains_italian(self):
        it_data = [
            {"acronym": "IVA",  "definition": "Imposta sul Valore Aggiunto",     "pattern_type": "before", "label": 1},
            {"acronym": "PIL",  "definition": "Prodotto Interno Lordo",           "pattern_type": "before", "label": 1},
            {"acronym": "PEC",  "definition": "Posta Elettronica Certificata",    "pattern_type": "before", "label": 1},
            {"acronym": "INPS", "definition": "Istituto Nazionale Previdenza",    "pattern_type": "before", "label": 1},
            {"acronym": "UE",   "definition": "Unione Europea",                   "pattern_type": "before", "label": 1},
            {"acronym": "EX",   "definition": "l'ex presidente della regione",    "pattern_type": "before", "label": 0},
            {"acronym": "ST",   "definition": "una strada del centro storico",    "pattern_type": "before", "label": 0},
            {"acronym": "NB",   "definition": "si prega di notare il paragrafo",  "pattern_type": "before", "label": 0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = _write_json(tmpdir, it_data)
            model_dir = os.path.join(tmpdir, "models")
            model = train_from_file(data_path, lang="it", model_dir=model_dir)
            assert model.is_trained()
            assert model.lang == "it"


# ---------------------------------------------------------------------------
# Tests: train_from_samples
# ---------------------------------------------------------------------------


class TestTrainFromSamples:

    def test_trains_and_saves(self):
        samples = [(d["acronym"], d["definition"], d["pattern_type"]) for d in _VALID_DATA]
        labels = [d["label"] for d in _VALID_DATA]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models")
            model = train_from_samples(samples, labels, lang="en", model_dir=model_dir)
            assert model.is_trained()

    def test_warm_start_from_base_model(self):
        """Passing base_model warm-starts training and keeps the model trained."""
        samples = [(d["acronym"], d["definition"], d["pattern_type"]) for d in _VALID_DATA]
        labels = [d["label"] for d in _VALID_DATA]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models")
            # First train
            base = train_from_samples(samples, labels, lang="en", model_dir=model_dir)
            assert base.is_trained()
            # Second train with warm start
            model_dir2 = os.path.join(tmpdir, "models2")
            updated = train_from_samples(
                samples, labels, lang="en", model_dir=model_dir2, base_model=base
            )
            assert updated.is_trained()
            assert updated is base  # same object returned

    def test_warm_start_untrained_base_falls_back(self):
        """Passing an untrained base_model falls back to a fresh train."""
        from acronym.model import AcronymModel
        samples = [(d["acronym"], d["definition"], d["pattern_type"]) for d in _VALID_DATA]
        labels = [d["label"] for d in _VALID_DATA]
        untrained = AcronymModel(lang="en")
        assert not untrained.is_trained()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models")
            model = train_from_samples(
                samples, labels, lang="en", model_dir=model_dir, base_model=untrained
            )
            assert model.is_trained()

