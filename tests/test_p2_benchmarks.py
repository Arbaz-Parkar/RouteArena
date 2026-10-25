"""Small checks for the optional final-milestone references."""

from __future__ import annotations

import numpy as np
import pytest

from routearena.logistics.routing import held_karp
from routearena.logistics.routing.ortools_reference import ortools_reference
from routearena.logistics.routing.tsplib import BURMA14_OPTIMAL_COST, burma14_distance_matrix


def test_burma14_matches_published_tsplib_optimum() -> None:
    matrix = burma14_distance_matrix()
    assert matrix.shape == (14, 14)
    np.testing.assert_allclose(matrix, matrix.T)
    result = held_karp(matrix, max_customers=13)
    assert result.cost == BURMA14_OPTIMAL_COST


def test_ortools_reference_returns_valid_closed_route_when_installed() -> None:
    pytest.importorskip("ortools")
    matrix = np.array(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 0.0, 1.0, 3.0],
            [3.0, 1.0, 0.0, 1.0],
            [2.0, 3.0, 1.0, 0.0],
        ]
    )
    result = ortools_reference(matrix, time_limit_seconds=1)
    assert result.route[0] == result.route[-1] == 0
    assert sorted(result.route[1:-1]) == [1, 2, 3]
    assert result.cost >= 0.0
    assert result.metadata["optimality_proven"] is False
