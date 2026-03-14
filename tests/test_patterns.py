"""Tests for acronym.patterns – candidate extraction."""

import pytest

from acronym.patterns import find_candidates


# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------


class TestFindCandidatesEnglish:
    """Test candidate extraction on English text."""

    def test_acronym_before_definition(self):
        text = "The NATO (North Atlantic Treaty Organization) was founded in 1949."
        candidates = find_candidates(text, lang="en")
        assert len(candidates) >= 1
        acronyms = [c[0] for c in candidates]
        assert "NATO" in acronyms

    def test_definition_before_acronym(self):
        text = "We use Artificial Intelligence (AI) to solve complex problems."
        candidates = find_candidates(text, lang="en")
        assert len(candidates) >= 1
        acronyms = [c[0] for c in candidates]
        assert "AI" in acronyms

    def test_multiple_acronyms(self):
        text = (
            "The CPU (Central Processing Unit) and GPU (Graphics Processing Unit) "
            "work together."
        )
        candidates = find_candidates(text, lang="en")
        acronyms = [c[0] for c in candidates]
        assert "CPU" in acronyms
        assert "GPU" in acronyms

    def test_pattern_type_before(self):
        text = "NATO (North Atlantic Treaty Organization) was founded."
        candidates = find_candidates(text, lang="en")
        before = [c for c in candidates if c[2] == "before"]
        assert len(before) >= 1

    def test_pattern_type_after(self):
        text = "North Atlantic Treaty Organization (NATO) was founded."
        candidates = find_candidates(text, lang="en")
        after = [c for c in candidates if c[2] == "after"]
        assert len(after) >= 1

    def test_no_false_acronym_from_number(self):
        text = "The value is 42 (the answer to everything)."
        candidates = find_candidates(text, lang="en")
        # Pure numeric parentheticals should not match our patterns
        assert all(c[0][0].isupper() for c in candidates)

    def test_empty_text(self):
        assert find_candidates("", lang="en") == []

    def test_no_duplicates(self):
        text = "NATO (North Atlantic Treaty Organization) and NATO (North Atlantic Treaty Organization)."
        candidates = find_candidates(text, lang="en")
        keys = [(c[0].upper(), c[1].lower()) for c in candidates]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Italian
# ---------------------------------------------------------------------------


class TestFindCandidatesItalian:
    """Test candidate extraction on Italian text."""

    def test_acronym_before_definition(self):
        text = "L'IVA (Imposta sul Valore Aggiunto) è un'imposta indiretta."
        candidates = find_candidates(text, lang="it")
        acronyms = [c[0] for c in candidates]
        assert "IVA" in acronyms

    def test_definition_before_acronym(self):
        text = "La Posta Elettronica Certificata (PEC) è obbligatoria."
        candidates = find_candidates(text, lang="it")
        acronyms = [c[0] for c in candidates]
        assert "PEC" in acronyms

    def test_multiple_acronyms(self):
        text = (
            "L'INPS (Istituto Nazionale della Previdenza Sociale) e "
            "l'ASL (Azienda Sanitaria Locale) collaborano."
        )
        candidates = find_candidates(text, lang="it")
        acronyms = [c[0] for c in candidates]
        assert "INPS" in acronyms
        assert "ASL" in acronyms


# ---------------------------------------------------------------------------
# find_candidates_with_context
# ---------------------------------------------------------------------------


class TestFindCandidatesWithContext:
    """Tests for the context-aware variant of candidate extraction."""

    def test_returns_four_tuples(self):
        from acronym.patterns import find_candidates_with_context
        text = "The NATO (North Atlantic Treaty Organization) was founded in 1949."
        candidates = find_candidates_with_context(text, lang="en")
        assert len(candidates) >= 1
        # Each entry must be a 4-tuple
        for c in candidates:
            assert len(c) == 4

    def test_context_contains_match(self):
        from acronym.patterns import find_candidates_with_context
        text = "The CPU (Central Processing Unit) runs at 3 GHz."
        candidates = find_candidates_with_context(text, lang="en")
        acr_map = {c[0]: c[3] for c in candidates}
        assert "CPU" in acr_map
        # The context window should contain the acronym itself
        assert "CPU" in acr_map["CPU"]

    def test_consistent_with_find_candidates(self):
        """find_candidates_with_context stripped of context equals find_candidates."""
        from acronym.patterns import find_candidates_with_context
        text = (
            "The API (Application Programming Interface) and "
            "Machine Learning (ML) are important."
        )
        with_ctx = find_candidates_with_context(text, lang="en")
        without_ctx = find_candidates(text, lang="en")
        assert [(c[0], c[1], c[2]) for c in with_ctx] == list(without_ctx)

    def test_empty_text_returns_empty(self):
        from acronym.patterns import find_candidates_with_context
        assert find_candidates_with_context("", lang="en") == []
