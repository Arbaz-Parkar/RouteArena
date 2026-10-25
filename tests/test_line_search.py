"""Tests for one-dimensional line search methods."""

import pytest

from routearena.optimizers.line_search import (
    fibonacci_search,
    golden_section_search,
    quadratic_fit_search,
)


@pytest.mark.parametrize("search", [golden_section_search, fibonacci_search])
def test_bounded_search_finds_quadratic_minimum(search):
    result = search(lambda x: (x - 2.5) ** 2 + 3.0, -4.0, 8.0, tol=1e-7)

    assert result.success
    assert result.x == pytest.approx(2.5, abs=2e-6)
    assert result.fun == pytest.approx(3.0, abs=1e-10)


def test_quadratic_fit_search_refines_a_bracket():
    result = quadratic_fit_search(lambda x: (x - 1.25) ** 2, 0.0, 1.0, 2.0)

    assert result.success
    assert result.x == pytest.approx(1.25, abs=1e-6)

