"""A simple nearest-unvisited-stop baseline."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import RouteResult, make_result, validate_time_matrix


def nearest_neighbor(time_matrix: np.ndarray) -> RouteResult:
    """Repeatedly visit the closest unvisited stop, then return to the hub."""
    started_at = perf_counter()
    matrix = validate_time_matrix(time_matrix)
    n_points = matrix.shape[0]
    unvisited = set(range(1, n_points))
    route = [0]
    while unvisited:
        current = route[-1]
        next_point = min(unvisited, key=lambda point: (matrix[current, point], point))
        route.append(next_point)
        unvisited.remove(next_point)
    route.append(0)
    return make_result(matrix, route, started_at, "nearest_neighbor")
