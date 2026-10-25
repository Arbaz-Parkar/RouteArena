"""Traffic uncertainty models for one closed delivery tour."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from routearena.logistics.routing.common import validate_closed_route, validate_time_matrix


@dataclass(frozen=True)
class TrafficModel:
    """Parameters for the shared day factor and independent per-leg noise."""

    shared_mean: float
    shared_cv: float
    leg_cv: float


TRAFFIC_MODELS = {
    # Mean multipliers and coefficients of variation are explicit assumptions, not traffic data.
    "low": TrafficModel(shared_mean=1.05, shared_cv=0.08, leg_cv=0.06),
    "moderate": TrafficModel(shared_mean=1.20, shared_cv=0.16, leg_cv=0.10),
    "heavy": TrafficModel(shared_mean=1.45, shared_cv=0.28, leg_cv=0.16),
}


@dataclass(frozen=True)
class MonteCarloResult:
    """Summary statistics and simulated total times for a route."""

    route: tuple[int, ...]
    traffic_level: str
    deadline_minutes: float
    sample_count: int
    seed: int | None
    mean_minutes: float
    standard_deviation_minutes: float
    percentile_50_minutes: float
    percentile_90_minutes: float
    percentile_95_minutes: float
    probability_late: float
    samples: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class IntervalResult:
    """Per-leg interval bounds and their route-level totals."""

    route: tuple[int, ...]
    lower_multiplier: float
    upper_multiplier: float
    lower_leg_minutes: np.ndarray = field(repr=False, compare=False)
    upper_leg_minutes: np.ndarray = field(repr=False, compare=False)
    lower_total_minutes: float
    worst_case_minutes: float


def _lognormal_parameters(mean: float, coefficient_of_variation: float) -> tuple[float, float]:
    """Convert an arithmetic mean and CV to NumPy's lognormal parameters."""
    sigma_squared = float(np.log1p(coefficient_of_variation**2))
    sigma = float(np.sqrt(sigma_squared))
    mu = float(np.log(mean) - sigma_squared / 2.0)
    return mu, sigma


def simulate_route(
    time_matrix: np.ndarray,
    route: tuple[int, ...] | list[int],
    *,
    deadline_minutes: float,
    traffic_level: str = "moderate",
    sample_count: int = 10_000,
    seed: int | None = 0,
) -> MonteCarloResult:
    """Simulate a route with shared city traffic and independent leg-level noise.

    Each simulated day draws one lognormal traffic multiplier shared by all legs.
    Every leg also receives an independent mean-one lognormal multiplier. Both
    distributions are parameterized by arithmetic mean and coefficient of
    variation. Their values are teaching assumptions, not measured traffic data.
    """
    matrix = validate_time_matrix(time_matrix)
    closed_route = validate_closed_route(route, matrix.shape[0])
    if traffic_level not in TRAFFIC_MODELS:
        raise ValueError(f"traffic_level must be one of {', '.join(TRAFFIC_MODELS)}")
    if not np.isfinite(deadline_minutes) or deadline_minutes <= 0.0:
        raise ValueError("deadline_minutes must be finite and positive")
    if not isinstance(sample_count, int) or sample_count < 100 or sample_count > 100_000:
        raise ValueError("sample_count must be an integer between 100 and 100000")

    model = TRAFFIC_MODELS[traffic_level]
    shared_mu, shared_sigma = _lognormal_parameters(model.shared_mean, model.shared_cv)
    leg_mu, leg_sigma = _lognormal_parameters(1.0, model.leg_cv)
    leg_times = matrix[np.asarray(closed_route[:-1]), np.asarray(closed_route[1:])]
    rng = np.random.default_rng(seed)
    shared_factor = rng.lognormal(shared_mu, shared_sigma, size=sample_count)
    per_leg_factor = rng.lognormal(leg_mu, leg_sigma, size=(sample_count, len(leg_times)))
    totals = shared_factor * np.sum(per_leg_factor * leg_times, axis=1)

    return MonteCarloResult(
        route=closed_route,
        traffic_level=traffic_level,
        deadline_minutes=float(deadline_minutes),
        sample_count=sample_count,
        seed=seed,
        mean_minutes=float(np.mean(totals)),
        standard_deviation_minutes=float(np.std(totals, ddof=1)),
        percentile_50_minutes=float(np.percentile(totals, 50)),
        percentile_90_minutes=float(np.percentile(totals, 90)),
        percentile_95_minutes=float(np.percentile(totals, 95)),
        probability_late=float(np.mean(totals > deadline_minutes)),
        samples=totals,
    )


def interval_route(
    time_matrix: np.ndarray,
    route: tuple[int, ...] | list[int],
    *,
    lower_multiplier: float = 0.8,
    upper_multiplier: float = 1.8,
) -> IntervalResult:
    """Compute route bounds when each leg lies within chosen multiplicative limits.

    This is set-based analysis: every leg is allowed to independently take any
    value between its lower and upper multiplier. The upper sum is the worst case.
    """
    matrix = validate_time_matrix(time_matrix)
    closed_route = validate_closed_route(route, matrix.shape[0])
    if (
        not np.isfinite(lower_multiplier)
        or not np.isfinite(upper_multiplier)
        or lower_multiplier < 0.0
        or upper_multiplier < lower_multiplier
    ):
        raise ValueError("interval multipliers must be finite with 0 <= lower <= upper")
    legs = matrix[np.asarray(closed_route[:-1]), np.asarray(closed_route[1:])]
    lower_legs = legs * lower_multiplier
    upper_legs = legs * upper_multiplier
    return IntervalResult(
        route=closed_route,
        lower_multiplier=float(lower_multiplier),
        upper_multiplier=float(upper_multiplier),
        lower_leg_minutes=lower_legs,
        upper_leg_minutes=upper_legs,
        lower_total_minutes=float(np.sum(lower_legs)),
        worst_case_minutes=float(np.sum(upper_legs)),
    )


def convergence_curve(
    samples: np.ndarray,
    checkpoints: tuple[int, ...] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample-count, running-mean, and running-90th-percentile checkpoints."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite one-dimensional array")
    if checkpoints is None:
        fractions = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        counts = np.unique(np.maximum(1, np.ceil(values.size * np.asarray(fractions)).astype(int)))
    else:
        counts = np.asarray(sorted(set(checkpoints)), dtype=int)
        if counts.size == 0 or np.any(counts < 1) or np.any(counts > values.size):
            raise ValueError("checkpoints must be non-empty sample counts within the data")
    means = np.asarray([np.mean(values[:count]) for count in counts], dtype=float)
    percentile_90 = np.asarray(
        [np.percentile(values[:count], 90) for count in counts], dtype=float
    )
    return counts, means, percentile_90
