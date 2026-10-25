"""2-opt edge swaps for improving a closed symmetric route."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import (
    RouteResult,
    make_result,
    route_cost,
    validate_closed_route,
    validate_time_matrix,
)


def two_opt(time_matrix: np.ndarray, route: tuple[int, ...] | list[int]) -> RouteResult:
    """Repeatedly reverse route segments that reduce total travel time."""
    started_at = perf_counter()
    matrix = validate_time_matrix(time_matrix)
    best = list(validate_closed_route(route, matrix.shape[0]))
    best_cost = route_cost(matrix, best)
    history = [best_cost]
    improvements = 0

    changed = True
    while changed:
        changed = False
        # The hub stays fixed at both ends; only customer-order segments are reversed.
        for left in range(1, len(best) - 2):
            for right in range(left + 1, len(best) - 1):
                a, b = best[left - 1], best[left]
                c, d = best[right], best[right + 1]
                delta = matrix[a, c] + matrix[b, d] - matrix[a, b] - matrix[c, d]
                if delta < -1e-12:
                    best[left : right + 1] = reversed(best[left : right + 1])
                    best_cost += float(delta)
                    history.append(best_cost)
                    improvements += 1
                    changed = True
                    break
            if changed:
                break

    return make_result(
        matrix,
        best,
        started_at,
        "two_opt",
        metadata={"improvements": improvements},
        history=history,
    )
