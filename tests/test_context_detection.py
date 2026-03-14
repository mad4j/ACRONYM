"""Tests for contextual acronym detection.

Covers:
- ``find_standalone_candidates`` in ``acronym.patterns``
- ``extract_context_features`` and ``score_standalone_candidate`` in ``acronym.features``
- ``detect_standalone_acronyms_from_text`` in ``acronym.detector``
"""

import numpy as np
import pytest

from acronym.patterns import find_standalone_candidates
from acronym.features import (
    CONTEXT_FEATURE_NAMES,
    extract_context_features,
    score_standalone_candidate,
)
from acronym.detector import detect_standalone_acronyms_from_text


# ---------------------------------------------------------------------------
# Tests: find_standalone_candidates
# ---------------------------------------------------------------------------


class TestFindStandaloneCandidates:

    def test_finds_standalone_uppercase_word(self):
        text = "The CPU handles all calculations in the system."
        candidates = find_standalone_candidates(text)
        acronyms = [a for a, _ in candidates]
        assert "CPU" in acronyms

    def test_excludes_explicit_pairs(self):
        """Words that appear with an explicit definition must not be returned."""
        text = "The CPU (Central Processing Unit) is fast. CPU is used everywhere."
        candidates = find_standalone_candidates(text)
        acronyms = [a for a, _ in candidates]
        assert "CPU" not in acronyms

    def test_context_window_included(self):
        text = "The CPU handles all calculations."
        candidates = find_standalone_candidates(text, context_window=50)
        assert len(candidates) >= 1
        acronym, context = candidates[0]
        assert acronym == "CPU"
        assert "CPU" in context

    def test_deduplication(self):
        """Each acronym should appear at most once."""
        text = "The CPU is fast. The CPU is reliable. CPU works well."
        candidates = find_standalone_candidates(text)
        acronyms = [a for a, _ in candidates]
        assert acronyms.count("CPU") == 1

    def test_empty_text_returns_empty(self):
        assert find_standalone_candidates("") == []

    def test_no_uppercase_words(self):
        text = "this text has no uppercase words at all."
        candidates = find_standalone_candidates(text)
        assert candidates == []

    def test_multiple_standalone_words(self):
        text = "The CPU and GPU are both important for ML workloads."
        candidates = find_standalone_candidates(text)
        acronyms = [a for a, _ in candidates]
        assert len(acronyms) >= 2
        assert "CPU" in acronyms
        assert "GPU" in acronyms

    def test_italian_standalone(self):
        text = "L'IVA è applicata su tutti i prodotti. IVA viene calcolata in percentuale."
        # "IVA" has no parenthetical definition here
        candidates = find_standalone_candidates(text, lang="it")
        acronyms = [a for a, _ in candidates]
        assert "IVA" in acronyms

    def test_does_not_return_single_letter(self):
        """Single-letter words should not be returned (min length is 2)."""
        text = "The A stands for something in A context."
        candidates = find_standalone_candidates(text)
        acronyms = [a for a, _ in candidates]
        assert "A" not in acronyms


# ---------------------------------------------------------------------------
# Tests: extract_context_features
# ---------------------------------------------------------------------------


class TestExtractContextFeatures:

    def test_returns_correct_length(self):
        feats = extract_context_features("CPU", "The CPU handles all calculations.")
        assert len(feats) == len(CONTEXT_FEATURE_NAMES)

    def test_returns_numpy_array(self):
        feats = extract_context_features("CPU", "The CPU handles all calculations.")
        assert isinstance(feats, np.ndarray)

    def test_all_caps_is_one_for_uppercase(self):
        feats = extract_context_features("NATO", "NATO summit was held in Brussels.")
        idx = CONTEXT_FEATURE_NAMES.index("is_all_caps")
        assert feats[idx] == 1.0

    def test_all_caps_is_zero_for_mixed(self):
        feats = extract_context_features("Nato", "Nato summit was held in Brussels.")
        idx = CONTEXT_FEATURE_NAMES.index("is_all_caps")
        assert feats[idx] == 0.0

    def test_is_all_alpha_for_letters_only(self):
        feats = extract_context_features("NATO", "some context")
        idx = CONTEXT_FEATURE_NAMES.index("is_all_alpha")
        assert feats[idx] == 1.0

    def test_is_not_all_alpha_with_digits(self):
        feats = extract_context_features("H2O", "some context")
        idx = CONTEXT_FEATURE_NAMES.index("is_all_alpha")
        assert feats[idx] == 0.0

    def test_len_in_range_for_typical_acronym(self):
        feats = extract_context_features("CPU", "Central Processing Unit context")
        idx = CONTEXT_FEATURE_NAMES.index("len_in_range")
        assert feats[idx] == 1.0

    def test_len_not_in_range_for_long_word(self):
        feats = extract_context_features("ABCDEFGHIJ", "some context")
        idx = CONTEXT_FEATURE_NAMES.index("len_in_range")
        assert feats[idx] == 0.0

    def test_is_common_word_for_known_non_acronym(self):
        feats = extract_context_features("THE", "some text here", lang="en")
        idx = CONTEXT_FEATURE_NAMES.index("is_common_word")
        assert feats[idx] == 1.0

    def test_is_not_common_word_for_acronym(self):
        feats = extract_context_features("NATO", "some text here", lang="en")
        idx = CONTEXT_FEATURE_NAMES.index("is_common_word")
        assert feats[idx] == 0.0

    def test_initial_match_ratio_high_when_initials_match(self):
        """CPU context containing 'Central Processing Unit' should have high ratio."""
        feats = extract_context_features(
            "CPU", "Central Processing Unit handles calculations", lang="en"
        )
        idx = CONTEXT_FEATURE_NAMES.index("initial_match_ratio")
        assert feats[idx] > 0.5

    def test_initial_match_ratio_low_without_matching_words(self):
        feats = extract_context_features(
            "CPU", "the system runs smoothly", lang="en"
        )
        idx = CONTEXT_FEATURE_NAMES.index("initial_match_ratio")
        assert feats[idx] < 0.5

    def test_empty_context(self):
        feats = extract_context_features("CPU", "", lang="en")
        assert len(feats) == len(CONTEXT_FEATURE_NAMES)
        idx = CONTEXT_FEATURE_NAMES.index("initial_match_ratio")
        assert feats[idx] == 0.0

    def test_italian_common_word(self):
        feats = extract_context_features("NO", "some text here", lang="it")
        idx = CONTEXT_FEATURE_NAMES.index("is_common_word")
        assert feats[idx] == 1.0


# ---------------------------------------------------------------------------
# Tests: score_standalone_candidate
# ---------------------------------------------------------------------------


class TestScoreStandaloneCandidate:

    def test_score_in_range(self):
        score = score_standalone_candidate("CPU", "Central Processing Unit context")
        assert 0.0 <= score <= 1.0

    def test_clear_acronym_with_matching_initials_scores_high(self):
        """CPU with 'Central Processing Unit' nearby should score above threshold."""
        score = score_standalone_candidate(
            "CPU", "Central Processing Unit is a key component", lang="en"
        )
        assert score >= 0.5

    def test_acronym_without_matching_initials_has_base_score(self):
        """An ALL-CAPS word that is not a common word should get at least base score."""
        score = score_standalone_candidate("NATO", "a famous international alliance", lang="en")
        assert score >= 0.5

    def test_common_word_scores_low(self):
        """Known common words like 'THE' or 'NO' should score very low."""
        score_the = score_standalone_candidate("THE", "some sentence here", lang="en")
        score_no = score_standalone_candidate("NO", "some sentence here", lang="en")
        assert score_the <= 0.1
        assert score_no <= 0.1

    def test_mixed_case_scores_zero(self):
        """A word that is not all-caps should score 0."""
        score = score_standalone_candidate("Nato", "some context", lang="en")
        assert score == 0.0

    def test_italian_common_caps_scores_low(self):
        score = score_standalone_candidate("IL", "some Italian text", lang="it")
        assert score <= 0.1

    def test_long_out_of_range_word_scores_lower(self):
        """Very long ALL-CAPS strings get a lower base score."""
        short = score_standalone_candidate("NATO", "context", lang="en")
        long_ = score_standalone_candidate("ABCDEFGHIJ", "context", lang="en")
        assert short >= long_


# ---------------------------------------------------------------------------
# Tests: detect_standalone_acronyms_from_text
# ---------------------------------------------------------------------------


class TestDetectStandaloneAcronymsFromText:

    def test_detects_standalone_acronym(self):
        text = "The CPU handles all computation in the system."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        acronyms = [r["acronym"] for r in results]
        assert "CPU" in acronyms

    def test_result_keys(self):
        text = "The NATO alliance met yesterday."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        assert len(results) >= 1
        for r in results:
            assert "acronym" in r
            assert "definition" in r
            assert "confidence" in r

    def test_definition_is_empty_string(self):
        text = "The CPU handles computation."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        for r in results:
            assert r["definition"] == ""

    def test_confidence_in_range(self):
        text = "The CPU and GPU work together."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        for r in results:
            assert 0.0 <= float(r["confidence"]) <= 1.0

    def test_threshold_filters_results(self):
        text = "The CPU handles computation."
        low = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        high = detect_standalone_acronyms_from_text(text, lang="en", threshold=1.0)
        assert len(low) >= len(high)

    def test_sorted_by_acronym(self):
        text = "The CPU and GPU and ML and NLP are all important topics."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        acronyms = [r["acronym"] for r in results]
        assert acronyms == sorted(acronyms)

    def test_empty_text_returns_empty(self):
        results = detect_standalone_acronyms_from_text("", lang="en")
        assert results == []

    def test_no_uppercase_words_returns_empty(self):
        results = detect_standalone_acronyms_from_text(
            "no uppercase words here", lang="en"
        )
        assert results == []

    def test_excludes_words_with_explicit_definitions(self):
        """Acronyms already present as explicit pairs are excluded from standalone results."""
        text = "The CPU (Central Processing Unit) is fast. CPU is used everywhere."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.0)
        acronyms = [r["acronym"] for r in results]
        assert "CPU" not in acronyms

    def test_common_words_filtered_by_default_threshold(self):
        """Common non-acronym words (THE, NO, ...) should not pass the 0.5 threshold."""
        text = "NO THE BY IN ON do not matter here."
        results = detect_standalone_acronyms_from_text(text, lang="en", threshold=0.5)
        for r in results:
            assert r["acronym"] not in {"NO", "THE", "BY", "IN", "ON"}

    def test_italian_detection(self):
        text = "Il PIL è cresciuto nell'ultimo trimestre."
        results = detect_standalone_acronyms_from_text(text, lang="it", threshold=0.0)
        acronyms = [r["acronym"] for r in results]
        assert "PIL" in acronyms

    def test_context_window_parameter(self):
        """A narrower context window should still produce results."""
        text = "The CPU handles all computation."
        results_wide = detect_standalone_acronyms_from_text(
            text, lang="en", threshold=0.0, context_window=300
        )
        results_narrow = detect_standalone_acronyms_from_text(
            text, lang="en", threshold=0.0, context_window=10
        )
        # Both should find CPU; confidence may differ
        assert any(r["acronym"] == "CPU" for r in results_wide)
        assert any(r["acronym"] == "CPU" for r in results_narrow)
