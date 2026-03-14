"""Language-specific regex patterns for candidate acronym-definition extraction.

Two patterns are recognised in both languages:
  - *acronym_before*: ``ACRONYM (Long Form Definition)``
  - *acronym_after*:  ``Long Form Definition (ACRONYM)``

The extended function :func:`find_candidates_with_context` also captures the
surrounding sentence (context window) for each match so that contextual
features can be computed downstream.
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Compiled patterns per language
# ---------------------------------------------------------------------------

# Both English and Italian share the same surface patterns; the distinction
# between languages is used by the feature extractor (stop-word lists) and by
# the trained ML model.
_ACRONYM_BEFORE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,9})\s*\(([^)]{3,150})\)",
    re.MULTILINE,
)

_ACRONYM_AFTER = re.compile(
    r"\b([A-Za-z\u00C0-\u024F][^()\n]{3,150}?)\s*\(\s*([A-Z][A-Z0-9]{1,9})\s*\)",
    re.MULTILINE,
)

LANG_PATTERNS: dict = {
    "en": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
    "it": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
}

# Candidate: (acronym, definition, pattern_type)
Candidate = Tuple[str, str, str]

# CandidateWithContext: (acronym, definition, pattern_type, context)
CandidateWithContext = Tuple[str, str, str, str]

#: Number of characters to include before and after a match as context.
_CONTEXT_WINDOW: int = 200


def _get_context(text: str, match_start: int, match_end: int) -> str:
    """Return a text window of ``_CONTEXT_WINDOW`` chars around a regex match."""
    start = max(0, match_start - _CONTEXT_WINDOW)
    end = min(len(text), match_end + _CONTEXT_WINDOW)
    return text[start:end]


def find_candidates_with_context(
    text: str, lang: str = "en"
) -> List[CandidateWithContext]:
    """Find candidate acronym-definition pairs together with their context.

    Args:
        text: Plain text to search.
        lang: Language code (``"en"`` or ``"it"``).

    Returns:
        List of ``(acronym, definition, pattern_type, context)`` 4-tuples where
        *pattern_type* is ``"before"`` or ``"after"`` and *context* is the
        surrounding text window (up to ``_CONTEXT_WINDOW`` characters on each
        side of the match).
    """
    patterns = LANG_PATTERNS.get(lang, LANG_PATTERNS["en"])
    candidates: List[CandidateWithContext] = []
    seen: set = set()

    # Pattern 1: ACRONYM (definition)
    for m in patterns["acronym_before"].finditer(text):
        acronym = m.group(1).strip()
        definition = m.group(2).strip()
        key = (acronym.upper(), definition.lower())
        if key not in seen:
            seen.add(key)
            context = _get_context(text, m.start(), m.end())
            candidates.append((acronym, definition, "before", context))

    # Pattern 2: definition (ACRONYM)
    for m in patterns["acronym_after"].finditer(text):
        definition = m.group(1).strip().rstrip(",;: ")
        acronym = m.group(2).strip()
        key = (acronym.upper(), definition.lower())
        if key not in seen:
            seen.add(key)
            context = _get_context(text, m.start(), m.end())
            candidates.append((acronym, definition, "after", context))

    return candidates


def find_candidates(text: str, lang: str = "en") -> List[Candidate]:
    """Find candidate acronym-definition pairs in *text*.

    Args:
        text: Plain text to search.
        lang: Language code (``"en"`` or ``"it"``).

    Returns:
        List of ``(acronym, definition, pattern_type)`` triples where
        *pattern_type* is ``"before"`` (acronym precedes definition) or
        ``"after"`` (definition precedes acronym).
    """
    return [
        (acr, defn, pt)
        for acr, defn, pt, _ctx in find_candidates_with_context(text, lang)
    ]
