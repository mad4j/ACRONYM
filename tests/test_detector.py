"""Tests for acronym.detector – detection API."""

import os
import tempfile

import pytest
import docx

from acronym.detector import detect_acronyms, detect_acronyms_from_text, load_model
from acronym.trainer import train_from_samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TRAIN_EN = [
    ("NATO", "North Atlantic Treaty Organization", "before"),
    ("AI",   "Artificial Intelligence",            "before"),
    ("CPU",  "Central Processing Unit",            "before"),
    ("WHO",  "World Health Organization",          "before"),
    ("API",  "Application Programming Interface",  "before"),
    ("ML",   "Machine Learning",                   "before"),
    ("NLP",  "Natural Language Processing",        "before"),
    ("IT",   "Italy",                              "before"),
    ("EX",   "a former employee of the company",   "before"),
    ("ON",   "based on the agreed premise above",  "before"),
    ("BY",   "located by the river district",      "before"),
    ("NO",   "not applicable to the current case", "before"),
]
_LABELS_EN = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

_TRAIN_IT = [
    ("IVA",   "Imposta sul Valore Aggiunto",                   "before"),
    ("PIL",   "Prodotto Interno Lordo",                        "before"),
    ("UE",    "Unione Europea",                                "before"),
    ("INPS",  "Istituto Nazionale della Previdenza Sociale",   "before"),
    ("PEC",   "Posta Elettronica Certificata",                 "before"),
    ("ASL",   "Azienda Sanitaria Locale",                      "before"),
    ("EX",    "l'ex presidente della regione nominato",        "before"),
    ("NB",    "si prega di notare quanto indicato",            "before"),
    ("ID",    "il codice identificativo univoco del record",   "before"),
]
_LABELS_IT = [1, 1, 1, 1, 1, 1, 0, 0, 0]


def _train_model(tmpdir, lang):
    if lang == "en":
        samples, labels = _TRAIN_EN, _LABELS_EN
    else:
        samples, labels = _TRAIN_IT, _LABELS_IT
    return train_from_samples(samples, labels, lang=lang, model_dir=tmpdir)


def _make_docx(tmpdir, text: str, filename: str = "test.docx") -> str:
    """Create a minimal .docx containing *text*."""
    doc = docx.Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    path = os.path.join(tmpdir, filename)
    doc.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------


class TestLoadModel:

    def test_load_trained_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _train_model(tmpdir, "en")
            model = load_model("en", model_dir=tmpdir)
            assert model.is_trained()

    def test_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_model("en", model_dir=tmpdir)


# ---------------------------------------------------------------------------
# Tests: detect_acronyms_from_text
# ---------------------------------------------------------------------------


class TestDetectAcronymsFromText:

    def test_detects_valid_acronym(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = "The CPU (Central Processing Unit) drives performance."
            results = detect_acronyms_from_text(text, lang="en", model=model)
            acronyms = [r["acronym"] for r in results]
            assert "CPU" in acronyms

    def test_result_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = "The API (Application Programming Interface) is documented."
            results = detect_acronyms_from_text(text, lang="en", model=model)
            assert len(results) >= 1
            for r in results:
                assert "acronym" in r
                assert "definition" in r
                assert "confidence" in r

    def test_confidence_in_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = "The API (Application Programming Interface) is documented."
            results = detect_acronyms_from_text(text, lang="en", model=model)
            for r in results:
                assert 0.0 <= float(r["confidence"]) <= 1.0

    def test_empty_text_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            results = detect_acronyms_from_text("", lang="en", model=model)
            assert results == []

    def test_threshold_filters_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = "The API (Application Programming Interface) is documented."
            low = detect_acronyms_from_text(text, lang="en", model=model, threshold=0.0)
            high = detect_acronyms_from_text(text, lang="en", model=model, threshold=1.0)
            assert len(low) >= len(high)

    def test_sorted_by_acronym(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = (
                "The CPU (Central Processing Unit) and API (Application Programming Interface) "
                "and ML (Machine Learning) are key concepts."
            )
            results = detect_acronyms_from_text(text, lang="en", model=model, threshold=0.0)
            acronyms = [r["acronym"] for r in results]
            assert acronyms == sorted(acronyms)

    def test_italian_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "it")
            text = "L'IVA (Imposta sul Valore Aggiunto) è un'imposta indiretta sulle vendite."
            results = detect_acronyms_from_text(text, lang="it", model=model)
            acronyms = [r["acronym"] for r in results]
            assert "IVA" in acronyms

    def test_no_model_and_no_saved_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                detect_acronyms_from_text(
                    "Some text AI (Artificial Intelligence).",
                    lang="en",
                    model_dir=tmpdir,
                )


# ---------------------------------------------------------------------------
# Tests: detect_acronyms (from .docx)
# ---------------------------------------------------------------------------


class TestDetectAcronyms:

    def test_detects_from_docx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            docx_path = _make_docx(
                tmpdir,
                "The CPU (Central Processing Unit) drives performance.",
            )
            results = detect_acronyms(docx_path, lang="en", model=model)
            acronyms = [r["acronym"] for r in results]
            assert "CPU" in acronyms

    def test_missing_docx_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            with pytest.raises(FileNotFoundError):
                detect_acronyms("/nonexistent/file.docx", lang="en", model=model)

    def test_empty_docx_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            docx_path = _make_docx(tmpdir, "No acronyms here at all.")
            results = detect_acronyms(docx_path, lang="en", model=model)
            assert isinstance(results, list)

    def test_multiple_acronyms_in_docx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _train_model(tmpdir, "en")
            text = (
                "The API (Application Programming Interface) allows integration.\n"
                "Machine Learning (ML) is used for predictions.\n"
                "The CPU (Central Processing Unit) is the brain."
            )
            docx_path = _make_docx(tmpdir, text)
            results = detect_acronyms(docx_path, lang="en", model=model)
            acronyms = [r["acronym"] for r in results]
            assert len(acronyms) >= 2
