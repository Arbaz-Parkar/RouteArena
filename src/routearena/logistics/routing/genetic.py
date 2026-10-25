"""A permutation genetic algorithm with order crossover and inversion mutation."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import (
    RouteResult,
    make_result,
    route_cost,
    validate_time_matrix,
)


def _order_crossover(first: np.ndarray, second: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy one parent segment and fill remaining positions in the other parent's order."""
    size = len(first)
    if size < 2:
        return first.copy()
    left, right = sorted(rng.choice(size, size=2, replace=False).tolist())
    child = np.full(size, -1, dtype=int)
    child[left : right + 1] = first[left : right + 1]
    present = set(child[left : right + 1].tolist())
    fill = [int(gene) for gene in second if int(gene) not in present]
    slots = [index for index in range(size) if child[index] == -1]
    for index, gene in zip(slots, fill, strict=True):
        child[index] = gene
    return child


def _mutate_inversion(order: np.ndarray, rng: np.random.Generator) -> None:
    """Reverse a random segment in place."""
    if len(order) >= 2:
        left, right = sorted(rng.choice(len(order), size=2, replace=False).tolist())
        order[left : right + 1] = order[left : right + 1][::-1]


def genetic_algorithm(
    time_matrix: np.ndarray,
    *,
    population_size: int = 80,
    generations: int = 250,
    mutation_rate: float = 0.2,
    tournament_size: int = 4,
    elite_count: int = 2,
    seed: int | None = 0,
) -> RouteResult:
    """Search customer permutations with tournament selection and elitism."""
    started_at = perf_counter()
    matrix = validate_time_matrix(time_matrix)
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if generations < 1:
        raise ValueError("generations must be positive")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not 2 <= tournament_size <= population_size:
        raise ValueError("tournament_size must be between 2 and population_size")
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be between 1 and population_size - 1")

    customers = np.arange(1, matrix.shape[0], dtype=int)
    if len(customers) == 0:
        return make_result(
            matrix, [0, 0], started_at, "genetic_algorithm", metadata={"seed": seed}, history=[0.0]
        )

    rng = np.random.default_rng(seed)
    population = [rng.permutation(customers) for _ in range(population_size)]
    baseline = np.asarray(nearest_neighbor_order(matrix), dtype=int)
    population[0] = baseline

    def evaluate(order: np.ndarray) -> float:
        return route_cost(matrix, [0, *order.tolist(), 0])

    costs = np.asarray([evaluate(order) for order in population], dtype=float)
    best_index = int(np.argmin(costs))
    best_order = population[best_index].copy()
    best_cost = float(costs[best_index])
    history = [best_cost]

    for _ in range(generations):
        ranking = np.argsort(costs, kind="stable")
        next_population = [population[index].copy() for index in ranking[:elite_count]]
        while len(next_population) < population_size:
            contenders_a = rng.choice(population_size, size=tournament_size, replace=False)
            contenders_b = rng.choice(population_size, size=tournament_size, replace=False)
            parent_a = population[min(contenders_a, key=lambda index: (costs[index], index))]
            parent_b = population[min(contenders_b, key=lambda index: (costs[index], index))]
            child = _order_crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                _mutate_inversion(child, rng)
            next_population.append(child)
        population = next_population
        costs = np.asarray([evaluate(order) for order in population], dtype=float)
        generation_best = int(np.argmin(costs))
        if costs[generation_best] < best_cost:
            best_cost = float(costs[generation_best])
            best_order = population[generation_best].copy()
        history.append(best_cost)

    return make_result(
        matrix,
        [0, *best_order.tolist(), 0],
        started_at,
        "genetic_algorithm",
        metadata={
            "seed": seed,
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elite_count": elite_count,
        },
        history=history,
    )


def nearest_neighbor_order(matrix: np.ndarray) -> tuple[int, ...]:
    """Build a greedy customer ordering without starting a nested solver result."""
    unvisited = set(range(1, matrix.shape[0]))
    order = []
    current = 0
    while unvisited:
        following = min(unvisited, key=lambda point: (matrix[current, point], point))
        order.append(following)
        unvisited.remove(following)
        current = following
    return tuple(order)
