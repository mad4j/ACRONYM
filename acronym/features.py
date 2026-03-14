"""Feature extraction for the acronym-definition ML classifier.

Each candidate ``(acronym, definition, pattern_type)`` is converted into a
fixed-length numeric feature vector used by the :class:`~acronym.model.AcronymModel`.

A second, lighter feature set – :func:`extract_context_features` – is used by
the contextual detector (:func:`~acronym.detector.detect_standalone_acronyms_from_text`)
to score standalone ALL-CAPS words that appear without an explicit definition.
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
# Common short all-caps words that are NOT acronyms
# (used only by the contextual detector)
# ---------------------------------------------------------------------------

_EN_COMMON_CAPS: FrozenSet[str] = frozenset(
    {
        "A", "I", "BY", "IN", "ON", "AT", "TO", "OF", "OR", "AND", "THE",
        "NO", "AS", "IS", "IT", "US", "HE", "SHE", "WE", "DO", "GO", "BE",
        "AN", "IF", "SO", "MY", "UP",
    }
)

_IT_COMMON_CAPS: FrozenSet[str] = frozenset(
    {
        "IL", "LA", "DI", "IN", "DA", "PER", "CON", "SU", "NO", "SI",
        "TU", "IO", "LE", "LO", "UN", "ED", "MA", "SE", "MI", "TI",
    }
)

LANG_COMMON_CAPS: dict = {
    "en": _EN_COMMON_CAPS,
    "it": _IT_COMMON_CAPS,
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

CONTEXT_FEATURE_NAMES: List[str] = [
    "acr_len",
    "is_all_caps",
    "is_all_alpha",
    "len_in_range",
    "initial_match_ratio",
    "is_common_word",
    "context_word_count",
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


def extract_context_features(
    acronym: str,
    context: str,
    lang: str = "en",
) -> np.ndarray:
    """Extract a feature vector for a standalone uppercase candidate and its context.

    Unlike :func:`extract_features`, no explicit definition is required.  The
    surrounding *context* window is used to derive contextual signals such as
    whether the acronym letters appear as initials of nearby words.

    Args:
        acronym: The candidate uppercase word (e.g. ``"CPU"``).
        context: A window of surrounding text from which contextual features
                 are derived.
        lang:    Language code (``"en"`` or ``"it"``).

    Returns:
        1-D ``numpy`` array of length ``len(CONTEXT_FEATURE_NAMES)``.
    """
    common_caps = LANG_COMMON_CAPS.get(lang, _EN_COMMON_CAPS)

    acr_len = len(acronym)
    is_all_caps = int(acronym.isupper())
    is_all_alpha = int(acronym.isalpha())
    len_in_range = int(2 <= acr_len <= 8)
    is_common_word = int(acronym.upper() in common_caps)

    context_words = [w for w in re.split(r"\s+", context.strip()) if w and w[0].isalpha()]
    context_word_count = len(context_words)
    initial_match_ratio = _first_letter_ratio(acronym, context_words) if context_words else 0.0

    return np.array(
        [
            acr_len,
            is_all_caps,
            is_all_alpha,
            len_in_range,
            initial_match_ratio,
            is_common_word,
            context_word_count,
        ],
        dtype=float,
    )


def score_standalone_candidate(
    acronym: str,
    context: str,
    lang: str = "en",
) -> float:
    """Compute a heuristic confidence score for a standalone uppercase candidate.

    This function does not require an explicit definition; it derives a
    confidence score purely from the *acronym* string and its surrounding
    *context* window.  It is used by
    :func:`~acronym.detector.detect_standalone_acronyms_from_text` to decide
    whether a word that appears in ALL CAPS – but without a parenthetical
    definition – is likely to be an acronym in the given context.

    Args:
        acronym: The candidate uppercase word.
        context: A window of surrounding text.
        lang:    Language code (``"en"`` or ``"it"``).

    Returns:
        A float in ``[0.0, 1.0]`` representing how likely the word is an
        acronym in context.  Returns ``0.1`` for known common non-acronym words
        (e.g. ``"THE"``, ``"NO"``) and ``0.0`` for words that are not fully
        uppercase.
    """
    feats = extract_context_features(acronym, context, lang)
    (
        _acr_len,
        is_all_caps,
        _is_all_alpha,
        len_in_range,
        initial_match_ratio,
        is_common_word,
        _ctx_words,
    ) = feats

    # A standalone acronym candidate must be written in all capitals.
    if not is_all_caps:
        return 0.0

    # Known common short words used in ALL CAPS are very unlikely to be acronyms.
    if is_common_word:
        return 0.1

    # Base confidence from structural properties (length in typical range).
    base = 0.55 if len_in_range else 0.3

    # Context boost: if the acronym's letters appear as initials of nearby
    # words it is strong evidence that the word is being used as an acronym.
    context_boost = initial_match_ratio * 0.45

    return min(1.0, base + context_boost)
