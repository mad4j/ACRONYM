"""Feature extraction for the acronym-definition ML classifier.

Each candidate ``(acronym, definition, pattern_type)`` is converted into a
fixed-length numeric feature vector used by the :class:`~acronym.model.AcronymModel`.
"""

import re
from typing import FrozenSet, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Language-specific stop words
# ---------------------------------------------------------------------------

_EN_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with",
    }
)

_IT_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "ai", "al", "alla", "alle", "allo", "che", "chi", "ci", "col",
        "con", "da", "dagli", "dai", "dal", "dalla", "dalle", "dallo", "degli",
        "dei", "del", "dell", "della", "delle", "dello", "di", "e", "ed", "è",
        "gli", "i", "il", "in", "la", "le", "lo", "nei", "nel", "nella",
        "nelle", "nello", "o", "per", "su", "sul", "sulla", "sulle", "sullo",
        "un", "una", "uno",
    }
)

LANG_STOPWORDS: dict = {
    "en": _EN_STOPWORDS,
    "it": _IT_STOPWORDS,
}

# ---------------------------------------------------------------------------
# Feature names (kept in sync with extract_features return order)
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "acr_len",
    "word_count",
    "sig_word_count",
    "is_all_caps",
    "is_all_alpha",
    "first_letter_ratio",
    "sig_first_letter_ratio",
    "length_ratio",
    "sig_length_ratio",
    "acronym_in_initials",
    "def_char_len",
    "pattern_before",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _first_letter_ratio(acronym: str, words: List[str]) -> float:
    """Fraction of acronym letters that appear in sequence among *words* initials."""
    if not acronym or not words:
        return 0.0
    initials = "".join(w[0].upper() for w in words if w and w[0].isalpha())
    acr_upper = acronym.upper()
    matched = 0
    j = 0
    for ch in acr_upper:
        while j < len(initials):
            if initials[j] == ch:
                matched += 1
                j += 1
                break
            j += 1
    return matched / len(acr_upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    acronym: str,
    definition: str,
    lang: str = "en",
    pattern_type: str = "before",
) -> np.ndarray:
    """Extract a numeric feature vector for a candidate pair.

    Args:
        acronym:      The candidate acronym string (e.g. ``"NATO"``).
        definition:   The candidate definition string.
        lang:         Language code (``"en"`` or ``"it"``).
        pattern_type: ``"before"`` if the acronym preceded its definition in
                      the source text, ``"after"`` otherwise.

    Returns:
        1-D ``numpy`` array of length ``len(FEATURE_NAMES)``.
    """
    stopwords = LANG_STOPWORDS.get(lang, _EN_STOPWORDS)

    all_words = [w for w in re.split(r"\s+", definition.strip()) if w]
    sig_words = [w for w in all_words if w.lower() not in stopwords]
    if not sig_words:
        sig_words = all_words

    acr_len = len(acronym)
    word_count = len(all_words)
    sig_word_count = len(sig_words)
    is_all_caps = int(acronym.isupper())
    is_all_alpha = int(acronym.isalpha())

    first_letter_ratio = _first_letter_ratio(acronym, all_words)
    sig_first_letter_ratio = _first_letter_ratio(acronym, sig_words)

    length_ratio = acr_len / word_count if word_count else 0.0
    sig_length_ratio = acr_len / sig_word_count if sig_word_count else 0.0

    # Does the acronym appear verbatim in the concatenated word initials?
    all_initials = "".join(w[0].upper() for w in all_words if w and w[0].isalpha())
    acronym_in_initials = int(acronym.upper() in all_initials)

    def_char_len = len(definition)
    pattern_before = int(pattern_type == "before")

    return np.array(
        [
            acr_len,
            word_count,
            sig_word_count,
            is_all_caps,
            is_all_alpha,
            first_letter_ratio,
            sig_first_letter_ratio,
            length_ratio,
            sig_length_ratio,
            acronym_in_initials,
            def_char_len,
            pattern_before,
        ],
        dtype=float,
    )
