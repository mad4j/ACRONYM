"""Tests for acronym.features – feature extraction."""

import numpy as np
import pytest

from acronym.features import FEATURE_NAMES, extract_features


class TestExtractFeatures:
    """Unit tests for the feature vector produced by extract_features."""

    def test_returns_ndarray(self):
        feat = extract_features("NATO", "North Atlantic Treaty Organization", "en", "before")
        assert isinstance(feat, np.ndarray)

    def test_vector_length_matches_feature_names(self):
        feat = extract_features("NATO", "North Atlantic Treaty Organization", "en", "before")
        assert feat.shape == (len(FEATURE_NAMES),)

    def test_all_caps_flag(self):
        feat_caps = extract_features("NATO", "North Atlantic Treaty Organization", "en", "before")
        feat_mixed = extract_features("Nato", "North Atlantic Treaty Organization", "en", "before")
        idx = FEATURE_NAMES.index("is_all_caps")
        assert feat_caps[idx] == 1.0
        assert feat_mixed[idx] == 0.0

    def test_first_letter_ratio_perfect_match(self):
        # N-A-T-O matches North-Atlantic-Treaty-Organization perfectly
        feat = extract_features("NATO", "North Atlantic Treaty Organization", "en", "before")
        idx = FEATURE_NAMES.index("first_letter_ratio")
        assert feat[idx] == pytest.approx(1.0)

    def test_first_letter_ratio_poor_match(self):
        # Random noise – letters don't match definition initials
        feat = extract_features("XYZ", "Apple Banana Cherry", "en", "before")
        idx = FEATURE_NAMES.index("first_letter_ratio")
        assert feat[idx] == pytest.approx(0.0)

    def test_pattern_before_flag(self):
        feat_before = extract_features("AI", "Artificial Intelligence", "en", "before")
        feat_after = extract_features("AI", "Artificial Intelligence", "en", "after")
        idx = FEATURE_NAMES.index("pattern_before")
        assert feat_before[idx] == 1.0
        assert feat_after[idx] == 0.0

    def test_italian_stopwords_ignored(self):
        # "Imposta sul Valore Aggiunto" – "sul" is an Italian stop word;
        # significant words are Imposta, Valore, Aggiunto → IVA still matches
        feat = extract_features("IVA", "Imposta sul Valore Aggiunto", "it", "before")
        idx = FEATURE_NAMES.index("sig_first_letter_ratio")
        assert feat[idx] == pytest.approx(1.0)

    def test_english_stopwords_ignored(self):
        # "Application Programming Interface" has no stopwords,
        # but "API" matches A-P-I perfectly
        feat = extract_features("API", "Application Programming Interface", "en", "before")
        idx = FEATURE_NAMES.index("first_letter_ratio")
        assert feat[idx] == pytest.approx(1.0)

    def test_acr_len_feature(self):
        feat = extract_features("NATO", "North Atlantic Treaty Organization", "en", "before")
        idx = FEATURE_NAMES.index("acr_len")
        assert feat[idx] == 4.0

    def test_word_count_feature(self):
        feat = extract_features("API", "Application Programming Interface", "en", "before")
        idx = FEATURE_NAMES.index("word_count")
        assert feat[idx] == 3.0

    def test_empty_definition(self):
        # Should not raise; edge case
        feat = extract_features("AI", "", "en", "before")
        assert feat.shape == (len(FEATURE_NAMES),)
