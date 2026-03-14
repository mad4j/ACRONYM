"""Feature extraction for the acronym-definition ML classifier.

Each candidate ``(acronym, definition, pattern_type)`` is converted into a
fixed-length numeric feature vector used by the :class:`~acronym.model.AcronymModel`.

Two contextual features are included:

* **is_common_word** – ``1`` when the acronym lowercased is a well-known
  function word (e.g. ``"it"``, ``"on"``), signalling it is probably *not* an
  acronym.
* **context_has_def_marker** – ``1`` when the surrounding sentence contains a
  definitional phrase such as *"stands for"* or *"also known as"*, which
  strongly signals an explicit acronym introduction.
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
# Language-specific common words (short words that are NOT acronyms)
# ---------------------------------------------------------------------------

#: English words whose uppercase form is often *mistaken* for an acronym.
#: This list covers common function/preposition words drawn from standard
#: English frequency lists.  Extend ``LANG_COMMON_WORDS`` at runtime if you
#: need to add domain-specific false-positive words without modifying this file.
_EN_COMMON_WORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "as", "at", "be", "by", "do", "ex", "go", "he",
        "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok",
        "on", "or", "so", "to", "up", "us", "we",
    }
)

#: Italian words whose uppercase form is often *mistaken* for an acronym.
#: Drawn from standard Italian frequency lists covering prepositions, articles,
#: pronouns, and common verbs.
_IT_COMMON_WORDS: FrozenSet[str] = frozenset(
    {
        "da", "di", "ed", "il", "la", "le", "lo", "ma", "mi", "ne",
        "no", "si", "su", "te", "tu", "va",
    }
)

LANG_COMMON_WORDS: dict = {
    "en": _EN_COMMON_WORDS,
    "it": _IT_COMMON_WORDS,
}

# ---------------------------------------------------------------------------
# Definitional context markers per language
# ---------------------------------------------------------------------------

#: Phrases that strongly indicate an acronym is being introduced in English.
_EN_DEF_MARKERS: List[str] = [
    "stands for",
    "stand for",
    "known as",
    "refers to",
    "abbreviated",
    "short for",
    "abbreviation",
    "i.e.",
    "namely",
    "meaning",
    "also called",
]

#: Phrases that strongly indicate an acronym is being introduced in Italian.
_IT_DEF_MARKERS: List[str] = [
    "ossia",
    "ovvero",
    "vale a dire",
    "abbreviazione",
    "acronimo",
    "sta per",
    "si intende",
    "denominato",
    "conosciuto come",
    "anche detto",
]

LANG_DEF_MARKERS: dict = {
    "en": _EN_DEF_MARKERS,
    "it": _IT_DEF_MARKERS,
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
    "is_common_word",
    "context_has_def_marker",
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
    context: str = "",
) -> np.ndarray:
    """Extract a numeric feature vector for a candidate pair.

    Args:
        acronym:      The candidate acronym string (e.g. ``"NATO"``).
        definition:   The candidate definition string.
        lang:         Language code (``"en"`` or ``"it"``).
        pattern_type: ``"before"`` if the acronym preceded its definition in
                      the source text, ``"after"`` otherwise.
        context:      The surrounding sentence or text window where the
                      candidate was found.  Used to compute the contextual
                      features ``is_common_word`` and
                      ``context_has_def_marker``.

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

    # -----------------------------------------------------------------------
    # Contextual features
    # -----------------------------------------------------------------------

    # Feature: is the acronym also a common function/dictionary word?
    # A high-frequency word that happens to be in uppercase is likely NOT an
    # acronym (e.g. "IT" meaning "it", "ON" meaning "on").
    common_words = LANG_COMMON_WORDS.get(lang, _EN_COMMON_WORDS)
    is_common_word = int(acronym.lower() in common_words)

    # Feature: does the surrounding context contain a definitional phrase
    # (e.g. "stands for", "also known as") that explicitly introduces the
    # acronym?  Such phrases are strong positive signals.
    def_markers = LANG_DEF_MARKERS.get(lang, _EN_DEF_MARKERS)
    ctx_lower = context.lower()
    context_has_def_marker = int(any(marker in ctx_lower for marker in def_markers))

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
            is_common_word,
            context_has_def_marker,
        ],
        dtype=float,
    )
