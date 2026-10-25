"""Shared validation and result types for routing algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RouteResult:
    """A closed route, its travel-time cost, and solver diagnostics."""

    route: tuple[int, ...]
    cost: float
    runtime_seconds: float
    algorithm: str
    metadata: dict[str, Any] = field(default_factory=dict)
    history: tuple[float, ...] = ()


def validate_time_matrix(time_matrix: np.ndarray, *, max_stops: int = 30) -> np.ndarray:
    """Return a checked square, symmetric, nonnegative travel-time matrix."""
    matrix = np.asarray(time_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("travel-time matrix must be a non-empty square matrix")
    if matrix.shape[0] > max_stops + 1:
        raise ValueError(f"at most {max_stops} customer stops are supported")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("travel-time matrix must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("travel times cannot be negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-10):
        raise ValueError("travel-time matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10):
        raise ValueError("travel-time matrix diagonal must be zero")
    return matrix


def route_cost(matrix: np.ndarray, route: tuple[int, ...] | list[int]) -> float:
    """Sum the edge costs for a closed route."""
    indices = np.asarray(route, dtype=int)
    return float(np.sum(matrix[indices[:-1], indices[1:]]))


def validate_closed_route(route: tuple[int, ...] | list[int], n_points: int) -> tuple[int, ...]:
    """Require one visit to every point, with the hub (zero) at both ends."""
    closed = tuple(int(point) for point in route)
    expected = tuple(range(1, n_points))
    if len(closed) != n_points + 1 or closed[0] != 0 or closed[-1] != 0:
        raise ValueError("route must start and end at hub index 0")
    if tuple(sorted(closed[1:-1])) != expected:
        raise ValueError("route must visit each customer index exactly once")
    return closed


def make_result(
    matrix: np.ndarray,
    route: tuple[int, ...] | list[int],
    started_at: float,
    algorithm: str,
    *,
    metadata: dict[str, Any] | None = None,
    history: tuple[float, ...] | list[float] = (),
) -> RouteResult:
    """Validate a result and capture its cost and elapsed wall time."""
    closed = validate_closed_route(route, matrix.shape[0])
    return RouteResult(
        route=closed,
        cost=route_cost(matrix, closed),
        runtime_seconds=perf_counter() - started_at,
        algorithm=algorithm,
        metadata={} if metadata is None else dict(metadata),
        history=tuple(float(value) for value in history),
    )
