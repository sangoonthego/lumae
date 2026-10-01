"""Hard gates for specific visual queries and AI-only annotation provenance."""

from __future__ import annotations

import math
import re

VAGUE = re.compile(r"\b(demonstrat(?:e|es|ing) the product|shows? the product|"
                   r"presents? the product|showcases? (?:its|the) features|"
                   r"something happens|explains? the benefits)\b", re.I)
SUBJECTIVE = re.compile(r"\b(amazing|best|perfect|premium|delicious|beautiful|"
                        r"healthy|effective|greatest)\b", re.I)


def query_errors(query: str) -> list[str]:
    words = re.findall(r"\b\w+\b", query)
    errors = []
    if len(words) < 6 or len(words) > 32:
        errors.append("query length outside 6..32 words")
    if VAGUE.search(query):
        errors.append("generic product phrasing")
    if SUBJECTIVE.search(query):
        errors.append("subjective claim")
    if not re.search(r"\b(?:a|an|the|one|two|three|workers|people|hands|children)\b", query, re.I):
        errors.append("missing concrete subject phrase")
    return errors


def interval_errors(start: float, end: float, duration: float) -> list[str]:
    if not all(math.isfinite(x) for x in (start, end, duration)):
        return ["nonfinite interval"]
    if not 0 <= start < end <= duration:
        return ["interval outside video"]
    if start == 0 and abs(end - duration) < 0.11:
        return ["whole-video fallback prohibited"]
    return []
