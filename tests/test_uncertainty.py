"""Tests for Monte Carlo traffic and interval route analysis."""

from __future__ import annotations

import numpy as np
import pytest

from routearena.logistics.uncertainty import (
    TRAFFIC_MODELS,
    convergence_curve,
    interval_route,
    simulate_route,
)


def _time_matrix() -> np.ndarray:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def test_monte_carlo_is_reproducible_and_reports_ordered_percentiles() -> None:
    matrix = _time_matrix()
    route = (0, 1, 2, 3, 0)
    first = simulate_route(matrix, route, deadline_minutes=5.0, sample_count=2_000, seed=19)
    second = simulate_route(matrix, route, deadline_minutes=5.0, sample_count=2_000, seed=19)
    assert np.array_equal(first.samples, second.samples)
    assert first.percentile_50_minutes <= first.percentile_90_minutes
    assert first.percentile_90_minutes <= first.percentile_95_minutes
    assert 0.0 <= first.probability_late <= 1.0
    assert first.mean_minutes == pytest.approx(second.mean_minutes)


def test_traffic_levels_raise_the_expected_shared_multiplier() -> None:
    assert TRAFFIC_MODELS["low"].shared_mean < TRAFFIC_MODELS["moderate"].shared_mean
    assert TRAFFIC_MODELS["moderate"].shared_mean < TRAFFIC_MODELS["heavy"].shared_mean


def test_interval_bounds_have_expected_shape_and_order() -> None:
    matrix = _time_matrix()
    route = (0, 1, 2, 3, 0)
    interval = interval_route(matrix, route, lower_multiplier=0.8, upper_multiplier=1.8)
    assert len(interval.lower_leg_minutes) == len(route) - 1
    assert np.all(interval.lower_leg_minutes <= interval.upper_leg_minutes)
    assert interval.lower_total_minutes <= interval.worst_case_minutes


def test_interval_worst_case_exceeds_moderate_traffic_sample_mean() -> None:
    matrix = _time_matrix()
    route = (0, 1, 2, 3, 0)
    monte_carlo = simulate_route(
        matrix, route, deadline_minutes=100.0, traffic_level="moderate", sample_count=5_000, seed=7
    )
    interval = interval_route(matrix, route, upper_multiplier=1.8)
    assert interval.worst_case_minutes >= monte_carlo.mean_minutes


def test_invalid_routes_and_bounds_raise_clear_errors() -> None:
    matrix = _time_matrix()
    with pytest.raises(ValueError, match="start and end at hub"):
        simulate_route(matrix, (1, 2, 3, 0), deadline_minutes=5.0)
    with pytest.raises(ValueError, match="traffic_level"):
        simulate_route(matrix, (0, 1, 2, 3, 0), deadline_minutes=5.0, traffic_level="rush hour")
    with pytest.raises(ValueError, match="sample_count"):
        simulate_route(matrix, (0, 1, 2, 3, 0), deadline_minutes=5.0, sample_count=10)
    with pytest.raises(ValueError, match="multipliers"):
        interval_route(matrix, (0, 1, 2, 3, 0), lower_multiplier=1.5, upper_multiplier=1.0)


def test_convergence_curve_returns_requested_prefix_statistics() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts, means, percentile_90 = convergence_curve(values, checkpoints=[2, 5])
    assert counts.tolist() == [2, 5]
    assert means.tolist() == pytest.approx([1.5, 3.0])
    assert percentile_90[0] <= percentile_90[1]
