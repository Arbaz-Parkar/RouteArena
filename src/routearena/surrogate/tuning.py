"""An end-to-end, seeded ACO tuning demonstration using actual route costs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from routearena.logistics.routing import ant_colony
from routearena.surrogate.bayes import bayesian_optimization, random_search
from routearena.surrogate.models import fit_gaussian_process, select_linear_model
from routearena.surrogate.sampling import (
    full_factorial,
    latin_hypercube,
    random_sampling,
    space_filling_metrics,
    stratified_sampling,
)

ACO_BOUNDS = np.array([[0.5, 2.5], [1.0, 5.0], [0.1, 0.8]], dtype=float)
ACO_PARAMETER_NAMES = ("alpha", "beta", "evaporation rate")


def synthetic_time_matrix(seed: int = 0, customer_count: int = 8) -> np.ndarray:
    """Create a seeded synthetic travel-time matrix with the app's stated assumptions."""
    if customer_count < 2 or customer_count > 12:
        raise ValueError("customer_count must be between 2 and 12")
    rng = np.random.default_rng(seed)
    points_km = rng.uniform(0.0, 10.0, size=(customer_count + 1, 2))
    displacement = points_km[:, None, :] - points_km[None, :, :]
    distance_km = np.sqrt(np.sum(displacement**2, axis=2))
    return distance_km * 1.25 / 25.0 * 60.0


def make_aco_objective(
    time_matrix: np.ndarray,
    *,
    solver_seeds: tuple[int, ...] = (0, 1, 2),
    iterations: int = 45,
    ants: int = 8,
) -> Callable[[np.ndarray], float]:
    """Return mean ACO tour time across fixed seeds for alpha, beta, evaporation."""
    matrix = np.asarray(time_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("time_matrix must be square")
    if not solver_seeds:
        raise ValueError("solver_seeds must not be empty")

    def objective(parameters: np.ndarray) -> float:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("ACO parameters must be a finite (alpha, beta, evaporation) vector")
        alpha, beta, evaporation = values
        results = [
            ant_colony(
                matrix,
                ants=ants,
                iterations=iterations,
                alpha=float(alpha),
                beta=float(beta),
                evaporation_rate=float(evaporation),
                seed=solver_seed,
            ).cost
            for solver_seed in solver_seeds
        ]
        return float(np.mean(results))

    return objective


def _design_points(plan: str, design_size: int, seed: int) -> np.ndarray:
    if plan == "Random":
        return random_sampling(ACO_BOUNDS, design_size, seed=seed)
    if plan == "Latin hypercube":
        return latin_hypercube(ACO_BOUNDS, design_size, seed=seed)
    levels = max(1, round(design_size ** (1.0 / ACO_BOUNDS.shape[0])))
    if plan == "Stratified grid":
        return stratified_sampling(ACO_BOUNDS, levels, seed=seed)
    if plan == "Full factorial":
        return full_factorial(ACO_BOUNDS, max(2, levels))
    raise ValueError("plan must be Random, Latin hypercube, Stratified grid, or Full factorial")


def run_tuning_pipeline(
    time_matrix: np.ndarray,
    *,
    plan: str = "Latin hypercube",
    design_size: int = 18,
    search_budget: int = 12,
    seed: int = 0,
) -> dict[str, object]:
    """Sample ACO settings, compare surrogates, then compare BO with random search."""
    if design_size < 8 or search_budget < 4:
        raise ValueError("design_size must be >= 8 and search_budget must be >= 4")
    design = _design_points(plan, design_size, seed)
    design_unit = (design - ACO_BOUNDS[:, 0]) / (ACO_BOUNDS[:, 1] - ACO_BOUNDS[:, 0])
    design_objective = make_aco_objective(
        time_matrix, solver_seeds=(seed + 10, seed + 11), iterations=40
    )
    design_values = np.asarray([design_objective(point) for point in design], dtype=float)
    selection = select_linear_model(design_unit, design_values, seed=seed)
    train_indices = selection.train_indices
    validation_indices = selection.validation_indices
    gp = fit_gaussian_process(design_unit[train_indices], design_values[train_indices])
    gp_mean, _ = gp.predict(design_unit[validation_indices])
    gp_rmse = float(np.sqrt(np.mean((gp_mean - design_values[validation_indices]) ** 2)))

    search_objective = make_aco_objective(
        time_matrix, solver_seeds=(seed + 20, seed + 21, seed + 22), iterations=45
    )
    initial_samples = max(4, search_budget // 2)
    bayes_result = bayesian_optimization(
        search_objective,
        ACO_BOUNDS,
        initial_samples=initial_samples,
        iterations=search_budget - initial_samples,
        seed=seed + 100,
    )
    random_result = random_search(
        search_objective, ACO_BOUNDS, bayes_result.values.size, seed=seed + 200
    )
    return {
        "plan": plan,
        "design_points": design,
        "design_values": design_values,
        "design_metrics": space_filling_metrics(design_unit),
        "selected_degree": selection.model.degree,
        "linear_validation_rmse": selection.validation_rmse,
        "linear_scores_by_degree": selection.scores_by_degree,
        "gp_validation_rmse": gp_rmse,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "bayesian": bayes_result,
        "random": random_result,
        "budget": int(bayes_result.values.size),
        "objective_description": "Mean ACO tour time over three fixed seeds",
    }
