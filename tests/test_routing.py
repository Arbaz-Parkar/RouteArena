"""Correctness checks for the from-scratch routing solvers."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from routearena.logistics.routing import (
    RouteResult,
    ant_colony,
    genetic_algorithm,
    held_karp,
    nearest_neighbor,
    two_opt,
)


def _euclidean_matrix(seed: int, points: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(-5.0, 5.0, size=(points, 2))
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def _assert_valid(result: RouteResult, point_count: int) -> None:
    assert result.route[0] == result.route[-1] == 0
    assert sorted(result.route[1:-1]) == list(range(1, point_count))
    assert result.cost >= 0.0
    assert result.runtime_seconds >= 0.0


def test_solvers_return_closed_permutations() -> None:
    matrix = _euclidean_matrix(14, points=8)
    for result in (
        nearest_neighbor(matrix),
        two_opt(matrix, nearest_neighbor(matrix).route),
        held_karp(matrix),
        genetic_algorithm(matrix, population_size=30, generations=40, seed=8),
        ant_colony(matrix, ants=12, iterations=30, seed=8),
    ):
        _assert_valid(result, len(matrix))


def test_held_karp_matches_brute_force_on_small_instance() -> None:
    matrix = _euclidean_matrix(7, points=8)
    expected = float("inf")
    for order in permutations(range(1, len(matrix))):
        route = (0, *order, 0)
        cost = sum(matrix[left, right] for left, right in zip(route[:-1], route[1:], strict=True))
        expected = min(expected, cost)
    result = held_karp(matrix)
    assert result.cost == pytest.approx(expected)


def test_held_karp_refuses_more_than_documented_limit() -> None:
    matrix = _euclidean_matrix(2, points=14)
    with pytest.raises(ValueError, match="limited to 12 customer stops"):
        held_karp(matrix)


def test_seeded_heuristics_are_reproducible() -> None:
    matrix = _euclidean_matrix(42)
    ga_a = genetic_algorithm(matrix, population_size=24, generations=25, seed=11)
    ga_b = genetic_algorithm(matrix, population_size=24, generations=25, seed=11)
    aco_a = ant_colony(matrix, ants=10, iterations=20, seed=11)
    aco_b = ant_colony(matrix, ants=10, iterations=20, seed=11)
    assert ga_a.route == ga_b.route
    assert ga_a.history == ga_b.history
    assert aco_a.route == aco_b.route
    assert aco_a.history == aco_b.history


def test_two_opt_never_increases_a_seeded_route_cost() -> None:
    for seed in range(15):
        matrix = _euclidean_matrix(seed, points=10)
        initial = nearest_neighbor(matrix)
        improved = two_opt(matrix, initial.route)
        assert improved.cost <= initial.cost + 1e-10


def test_heuristics_beat_the_mean_of_random_tours_on_a_small_instance() -> None:
    matrix = _euclidean_matrix(36, points=9)
    rng = np.random.default_rng(500)
    customers = np.arange(1, len(matrix))
    random_costs = []
    for _ in range(300):
        route = (0, *rng.permutation(customers).tolist(), 0)
        random_costs.append(
            sum(matrix[left, right] for left, right in zip(route[:-1], route[1:], strict=True))
        )
    random_average = float(np.mean(random_costs))

    nearest = nearest_neighbor(matrix)
    results = (
        nearest,
        two_opt(matrix, nearest.route),
        genetic_algorithm(matrix, population_size=40, generations=80, seed=10),
        ant_colony(matrix, ants=12, iterations=60, seed=10),
    )
    assert all(result.cost < random_average for result in results)


def test_matrix_validation_rejects_asymmetric_input() -> None:
    matrix = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        nearest_neighbor(matrix)


def test_nearest_neighbor_handles_hub_only_case() -> None:
    result = nearest_neighbor(np.zeros((1, 1)))
    assert result.route == (0, 0)
    assert result.cost == 0.0
