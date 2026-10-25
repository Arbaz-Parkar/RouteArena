"""A lightweight numerical grounding check for generated explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)


@dataclass(frozen=True)
class GroundingResult:
    """Whether every numeric claim can be matched to a number in the supplied context."""

    grounded: bool
    unsupported_numbers: tuple[str, ...]


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """Return source token, numeric value, and displayed decimal places."""
    found = []
    for match in _NUMBER_PATTERN.finditer(text):
        token = match.group().replace(",", "")
        if token.endswith("%"):
            token = token[:-1]
        try:
            number = float(token)
        except ValueError:
            continue
        mantissa, has_exponent, exponent = token.lower().partition("e")
        decimal_places = len(mantissa.partition(".")[2]) if "." in mantissa else 0
        if has_exponent:
            decimal_places -= int(exponent)
        found.append((match.group(), number, decimal_places))
    return found


def check_numeric_grounding(answer: str, context: str) -> GroundingResult:
    """Check answer numbers against supplied values, allowing displayed rounding.

    A context value is considered a match when it rounds to the same number of
    decimal places as the answer. This catches invented numeric claims while
    allowing a value such as 12.3 to summarize a computed 12.34.
    """
    context_numbers = [number for _, number, _ in _numbers(context)]
    unsupported: list[str] = []
    for token, number, decimals in _numbers(answer):
        if not any(
            round(candidate, decimals) == round(number, decimals) for candidate in context_numbers
        ):
            unsupported.append(token)
    return GroundingResult(not unsupported, tuple(unsupported))
