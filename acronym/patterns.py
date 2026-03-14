"""Language-specific regex patterns for candidate acronym-definition extraction.

Two patterns are recognised in both languages:
  - *acronym_before*: ``ACRONYM (Long Form Definition)``
  - *acronym_after*:  ``Long Form Definition (ACRONYM)``
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
    r"\b([A-Za-z\u00C0-\u00FA][^()\n]{3,150}?)\s*\(\s*([A-Z][A-Z0-9]{1,9})\s*\)",
    re.MULTILINE,
)

LANG_PATTERNS: dict = {
    "en": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
    "it": {"acronym_before": _ACRONYM_BEFORE, "acronym_after": _ACRONYM_AFTER},
}

# Candidate: (acronym, definition, pattern_type)
Candidate = Tuple[str, str, str]


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
