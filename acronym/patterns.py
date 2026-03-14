"""Language-specific regex patterns for candidate acronym-definition extraction.

Two patterns are recognised in both languages:
  - *acronym_before*: ``ACRONYM (Long Form Definition)``
  - *acronym_after*:  ``Long Form Definition (ACRONYM)``

A third, standalone pattern is used for *contextual* detection:
  - *standalone*: any ``ALL-CAPS`` token not already covered by the above pairs.
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

# Matches any standalone ALL-CAPS word of 2-10 characters (letters and/or digits).
_STANDALONE_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")

LANG_PATTERNS: dict = {
    "en": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
    "it": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
}

# Candidate: (acronym, definition, pattern_type)
Candidate = Tuple[str, str, str]

# Standalone candidate: (acronym, context_window)
StandaloneCandidate = Tuple[str, str]


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
    patterns = LANG_PATTERNS.get(lang, LANG_PATTERNS["en"])
    candidates: List[Candidate] = []
    seen: set = set()

    # Pattern 1: ACRONYM (definition)
    for m in patterns["acronym_before"].finditer(text):
        acronym = m.group(1).strip()
        definition = m.group(2).strip()
        key = (acronym.upper(), definition.lower())
        if key not in seen:
            seen.add(key)
            candidates.append((acronym, definition, "before"))

    # Pattern 2: definition (ACRONYM)
    for m in patterns["acronym_after"].finditer(text):
        definition = m.group(1).strip().rstrip(",;: ")
        acronym = m.group(2).strip()
        key = (acronym.upper(), definition.lower())
        if key not in seen:
            seen.add(key)
            candidates.append((acronym, definition, "after"))

    return candidates


def find_standalone_candidates(
    text: str,
    lang: str = "en",
    context_window: int = 150,
) -> List[StandaloneCandidate]:
    """Find standalone ALL-CAPS words that are *not* part of an explicit pair.

    This function complements :func:`find_candidates` by scanning for
    all-uppercase tokens (e.g. ``CPU``, ``NATO``) that appear in the text
    without an adjacent parenthetical definition.  Each such token is returned
    together with a window of surrounding text that can be used as contextual
    evidence.

    Args:
        text:           Plain text to search.
        lang:           Language code (``"en"`` or ``"it"``).
        context_window: Number of characters on each side of the match to
                        include in the context window.

    Returns:
        List of ``(acronym, context)`` pairs, deduplicated by acronym.
        Acronyms already discovered by :func:`find_candidates` are excluded.
    """
    # Build the set of acronyms already captured via explicit patterns so we
    # do not double-report them.
    explicit: set = {acr.upper() for acr, _, _ in find_candidates(text, lang)}

    candidates: List[StandaloneCandidate] = []
    seen: set = set()

    for m in _STANDALONE_ACRONYM.finditer(text):
        acronym = m.group(1)
        upper = acronym.upper()

        if upper in explicit or upper in seen:
            continue

        seen.add(upper)

        # Extract a window of surrounding text as contextual evidence.
        start = max(0, m.start() - context_window)
        end = min(len(text), m.end() + context_window)
        context = text[start:end].strip()

        candidates.append((acronym, context))

    return candidates
