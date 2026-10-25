"""Expected-improvement Bayesian optimization for bounded objectives."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from typing import Callable

import numpy as np

from routearena.surrogate.models import fit_gaussian_process
from routearena.surrogate.sampling import latin_hypercube, random_sampling

Objective = Callable[[np.ndarray], float]


def expected_improvement(
    mean: np.ndarray, standard_deviation: np.ndarray, best_value: float, *, xi: float = 0.01
) -> np.ndarray:
    """Calculate expected improvement for minimizing an objective."""
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(standard_deviation, dtype=float)
    if mu.shape != sigma.shape or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("mean and standard_deviation must be same-shaped finite arrays")
    if np.any(sigma < 0.0) or not np.isfinite(best_value) or not np.isfinite(xi) or xi < 0.0:
        raise ValueError("standard deviations and xi must be nonnegative; best_value finite")
    improvement = best_value - mu - xi
    z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 0.0)
    cdf = np.fromiter(
        (0.5 * (1.0 + erf(value / sqrt(2.0))) for value in z.flat),
        dtype=float,
        count=z.size,
    ).reshape(z.shape)
    density = np.exp(-0.5 * z**2) / sqrt(2.0 * pi)
    result = improvement * cdf + sigma * density
    deterministic_improvement = np.maximum(0.0, improvement)
    return np.where(sigma > 0.0, np.maximum(0.0, result), deterministic_improvement)


@dataclass(frozen=True)
class SearchResult:
    """Evaluated designs and the best-so-far objective for one search."""

    points: np.ndarray
    values: np.ndarray
    best_history: np.ndarray

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.values))

    @property
    def best_point(self) -> np.ndarray:
        return self.points[self.best_index].copy()

    @property
    def best_value(self) -> float:
        return float(self.values[self.best_index])


def _checked_bounds(bounds: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(bounds, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("bounds must have shape (number of dimensions, 2)")
    if not np.all(np.isfinite(values)) or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("each bound must contain finite values with upper > lower")
    return values


def _evaluate(objective: Objective, points: np.ndarray) -> np.ndarray:
    values = np.asarray([objective(point.copy()) for point in points], dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("objective returned a non-finite value")
    return values


def random_search(
    objective: Objective,
    bounds: np.ndarray | list[tuple[float, float]],
    evaluations: int,
    *,
    seed: int = 0,
) -> SearchResult:
    """Evaluate uniform random points for a fixed objective budget."""
    limits = _checked_bounds(bounds)
    points = random_sampling(limits, evaluations, seed=seed)
    values = _evaluate(objective, points)
    return SearchResult(points, values, np.minimum.accumulate(values))


def bayesian_optimization(
    objective: Objective,
    bounds: np.ndarray | list[tuple[float, float]],
    *,
    initial_samples: int = 8,
    iterations: int = 12,
    candidate_count: int = 512,
    length_scale: float = 0.25,
    xi: float = 0.01,
    seed: int = 0,
) -> SearchResult:
    """Optimize a black-box objective by fitting an RBF GP and maximizing EI."""
    limits = _checked_bounds(bounds)
    if initial_samples < 2 or iterations < 0 or candidate_count < 1:
        raise ValueError("initial_samples >= 2, iterations >= 0, and candidate_count >= 1 are required")
    dimensions = limits.shape[0]
    unit_bounds = np.tile(np.array([[0.0, 1.0]]), (dimensions, 1))
    unit_points = latin_hypercube(unit_bounds, initial_samples, seed=seed)
    points = limits[:, 0] + unit_points * (limits[:, 1] - limits[:, 0])
    values = _evaluate(objective, points)
    rng = np.random.default_rng(seed + 1)

    for iteration in range(iterations):
        normalized = (points - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
        model = fit_gaussian_process(normalized, values, length_scale=length_scale)
        candidates = rng.random((candidate_count, dimensions))
        mean, standard_deviation = model.predict(candidates)
        acquisition = expected_improvement(mean, standard_deviation, float(np.min(values)), xi=xi)
        best_candidate = candidates[int(np.argmax(acquisition))]
        point = limits[:, 0] + best_candidate * (limits[:, 1] - limits[:, 0])
        value = _evaluate(objective, point.reshape(1, -1))[0]
        points = np.vstack((points, point))
        values = np.append(values, value)

    return SearchResult(points, values, np.minimum.accumulate(values))
