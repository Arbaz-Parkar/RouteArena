"""Reproducible design points and simple space-filling metrics."""

# ruff: noqa: E402
from __future__ import annotations

from itertools import product

import numpy as np


def _validated_bounds(bounds: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(bounds, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("bounds must have shape (number of dimensions, 2)")
    if not np.all(np.isfinite(values)) or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("each bound must contain finite values with upper > lower")
    return values


def random_sampling(
    bounds: np.ndarray | list[tuple[float, float]], sample_count: int, *, seed: int = 0
) -> np.ndarray:
    """Draw independent uniform samples inside the supplied parameter bounds."""
    limits = _validated_bounds(bounds)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    unit = np.random.default_rng(seed).random((sample_count, limits.shape[0]))
    return limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])


def full_factorial(bounds: np.ndarray | list[tuple[float, float]], levels: int) -> np.ndarray:
    """Return every combination of evenly spaced values on each axis."""
    limits = _validated_bounds(bounds)
    if levels < 2:
        raise ValueError("levels must be at least 2")
    axes = [np.linspace(lower, upper, levels) for lower, upper in limits]
    return np.asarray(list(product(*axes)), dtype=float)


def latin_hypercube(
    bounds: np.ndarray | list[tuple[float, float]], sample_count: int, *, seed: int = 0
) -> np.ndarray:
    """Draw one jittered point from each stratum in every dimension."""
    limits = _validated_bounds(bounds)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    dimensions = limits.shape[0]
    unit = np.empty((sample_count, dimensions), dtype=float)
    for column in range(dimensions):
        unit[:, column] = (rng.permutation(sample_count) + rng.random(sample_count)) / sample_count
    return limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])


def stratified_sampling(
    bounds: np.ndarray | list[tuple[float, float]], levels: int, *, seed: int = 0
) -> np.ndarray:
    """Jitter a full factorial grid so every rectangular cell gets one sample."""
    limits = _validated_bounds(bounds)
    if levels < 1:
        raise ValueError("levels must be positive")
    dimensions = limits.shape[0]
    cells = np.asarray(list(product(range(levels), repeat=dimensions)), dtype=int)
    unit = (cells + np.random.default_rng(seed).random(cells.shape)) / levels
    return limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])


def space_filling_metrics(points: np.ndarray) -> dict[str, float]:  # noqa: F811
    """Return minimum pairwise distance and centered discrepancy on the unit cube."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("points must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("space-filling metrics expect finite points inside [0, 1]")

    differences = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    np.fill_diagonal(distances, np.inf)

    centered_single = np.prod(
        1.0 + 0.5 * np.abs(values - 0.5) - 0.5 * np.abs(values - 0.5) ** 2,
        axis=1,
    )
    pairwise = np.abs(values[:, None, :] - values[None, :, :])
    centered_pairs = np.prod(
        1.0
        + 0.5 * np.abs(values[:, None, :] - 0.5)
        + 0.5 * np.abs(values[None, :, :] - 0.5)
        - 0.5 * pairwise,
        axis=2,
    )
    n_points, dimensions = values.shape
    discrepancy_squared = (
        (13.0 / 12.0) ** dimensions
        - 2.0 * np.sum(centered_single) / n_points
        + np.sum(centered_pairs) / n_points**2
    )
    return {
        "minimum_pairwise_distance": float(np.min(distances)),
        "centered_discrepancy": float(np.sqrt(max(0.0, discrepancy_squared))),
    }


def _centered_discrepancy(points: np.ndarray) -> float:  # noqa: F811
    """Return SciPy's squared centered L2 discrepancy on the unit cube."""
    count, dimensions = points.shape
    centered = np.abs(points - 0.5)
    first_order = np.prod(1.0 + 0.5 * centered - 0.5 * centered**2, axis=1)
    pairwise = (
        1.0
        + 0.5 * centered[:, None, :]
        + 0.5 * centered[None, :, :]
        - 0.5 * np.abs(points[:, None, :] - points[None, :, :])
    )
    second_order = np.prod(pairwise, axis=2).sum()
    squared = (
        (13.0 / 12.0) ** dimensions - (2.0 / count) * first_order.sum() + second_order / count**2
    )
    return float(max(0.0, squared))


def space_filling_metrics(points: np.ndarray) -> dict[str, float]:  # noqa: F811
    """Measure normalized minimum spacing and centered discrepancy."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("points must be a 2D array with at least two rows and one column")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values")
    if np.all((values >= 0.0) & (values <= 1.0)):
        normalized = values
    else:
        minima = np.min(values, axis=0)
        spans = np.ptp(values, axis=0)
        normalized = np.divide(
            values - minima,
            spans,
            out=np.zeros_like(values),
            where=spans > 0,
        )
    differences = normalized[:, None, :] - normalized[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return {
        "minimum_pairwise_distance": float(np.min(distances)),
        "centered_discrepancy": _centered_discrepancy(normalized),
    }
