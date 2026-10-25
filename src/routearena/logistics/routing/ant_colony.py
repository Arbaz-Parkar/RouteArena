"""Ant colony optimization with symmetric pheromone updates."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import (
    RouteResult,
    make_result,
    route_cost,
    validate_time_matrix,
)


def ant_colony(
    time_matrix: np.ndarray,
    *,
    ants: int | None = None,
    iterations: int = 150,
    alpha: float = 1.0,
    beta: float = 3.0,
    evaporation_rate: float = 0.35,
    deposit_amount: float = 1.0,
    seed: int | None = 0,
) -> RouteResult:
    """Construct routes probabilistically and reinforce shorter tours with pheromone."""
    started_at = perf_counter()
    matrix = validate_time_matrix(time_matrix)
    customer_count = matrix.shape[0] - 1
    ant_count = max(4, customer_count) if ants is None else ants
    if ant_count < 1:
        raise ValueError("ants must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be nonnegative")
    if not 0.0 < evaporation_rate < 1.0:
        raise ValueError("evaporation_rate must be between 0 and 1")
    if deposit_amount <= 0.0:
        raise ValueError("deposit_amount must be positive")
    if customer_count == 0:
        return make_result(
            matrix, [0, 0], started_at, "ant_colony", metadata={"seed": seed}, history=[0.0]
        )

    rng = np.random.default_rng(seed)
    pheromone = np.ones_like(matrix)
    heuristic = 1.0 / np.maximum(matrix, 1e-12)
    np.fill_diagonal(heuristic, 0.0)
    best_route: list[int] | None = None
    best_cost = float("inf")
    history: list[float] = []

    for _ in range(iterations):
        tours: list[list[int]] = []
        tour_costs: list[float] = []
        for _ in range(ant_count):
            route = [0]
            unvisited = set(range(1, matrix.shape[0]))
            while unvisited:
                current = route[-1]
                candidates = np.asarray(sorted(unvisited), dtype=int)
                log_weight = (
                    alpha * np.log(np.maximum(pheromone[current, candidates], 1e-300))
                    + beta * np.log(np.maximum(heuristic[current, candidates], 1e-300))
                )
                probabilities = np.exp(log_weight - np.max(log_weight))
                probabilities /= probabilities.sum()
                following = int(rng.choice(candidates, p=probabilities))
                route.append(following)
                unvisited.remove(following)
            route.append(0)
            cost = route_cost(matrix, route)
            tours.append(route)
            tour_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_route = route.copy()

        pheromone *= 1.0 - evaporation_rate
        for route, cost in zip(tours, tour_costs, strict=True):
            deposit = deposit_amount / max(cost, 1e-12)
            for left, right in zip(route[:-1], route[1:], strict=True):
                pheromone[left, right] += deposit
                pheromone[right, left] += deposit
        if best_route is not None:
            elite_deposit = deposit_amount / max(best_cost, 1e-12)
            for left, right in zip(best_route[:-1], best_route[1:], strict=True):
                pheromone[left, right] += elite_deposit
                pheromone[right, left] += elite_deposit
        np.fill_diagonal(pheromone, 0.0)
        history.append(best_cost)

    assert best_route is not None
    return make_result(
        matrix,
        best_route,
        started_at,
        "ant_colony",
        metadata={
            "seed": seed,
            "ants": ant_count,
            "iterations": iterations,
            "alpha": alpha,
            "beta": beta,
            "evaporation_rate": evaporation_rate,
        },
        history=history,
    )
