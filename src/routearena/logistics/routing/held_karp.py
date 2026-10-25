"""Held-Karp dynamic programming for exact small routing instances."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import RouteResult, make_result, validate_time_matrix

MAX_EXACT_CUSTOMERS = 12


def held_karp(time_matrix: np.ndarray, *, max_customers: int = MAX_EXACT_CUSTOMERS) -> RouteResult:
    """Find an optimal closed tour in O(n² 2ⁿ) time for at most 12 customers."""
    started_at = perf_counter()
    matrix = validate_time_matrix(time_matrix)
    customer_count = matrix.shape[0] - 1
    if max_customers < 0:
        raise ValueError("max_customers must be nonnegative")
    if customer_count > max_customers:
        raise ValueError(
            f"Held-Karp is limited to {max_customers} customer stops; "
            f"received {customer_count}. Use a heuristic solver for larger routes."
        )
    if customer_count == 0:
        return make_result(matrix, [0, 0], started_at, "held_karp", metadata={"states": 1})

    # A bit at position k represents customer k + 1; each state ends at one customer.
    costs: dict[tuple[int, int], float] = {}
    parents: dict[tuple[int, int], int | None] = {}
    for customer in range(customer_count):
        mask = 1 << customer
        costs[(mask, customer)] = float(matrix[0, customer + 1])
        parents[(mask, customer)] = None

    for mask in range(1, 1 << customer_count):
        for last in range(customer_count):
            if not mask & (1 << last) or mask == (1 << last):
                continue
            prior_mask = mask ^ (1 << last)
            candidates = [
                (
                    previous,
                    costs[(prior_mask, previous)] + matrix[previous + 1, last + 1],
                )
                for previous in range(customer_count)
                if prior_mask & (1 << previous)
            ]
            best_previous, best_cost = min(candidates, key=lambda item: (item[1], item[0]))
            costs[(mask, last)] = float(best_cost)
            parents[(mask, last)] = best_previous

    full_mask = (1 << customer_count) - 1
    final_candidates = [
        (
            customer,
            costs[(full_mask, customer)] + matrix[customer + 1, 0],
        )
        for customer in range(customer_count)
    ]
    last, _ = min(final_candidates, key=lambda item: (item[1], item[0]))
    reverse_path: list[int] = []
    mask = full_mask
    while last is not None:
        reverse_path.append(last + 1)
        previous = parents[(mask, last)]
        mask ^= 1 << last
        last = previous
    route = [0, *reversed(reverse_path), 0]
    return make_result(
        matrix,
        route,
        started_at,
        "held_karp",
        metadata={"customers": customer_count, "max_customers": max_customers},
    )
