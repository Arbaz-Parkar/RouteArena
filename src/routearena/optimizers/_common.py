"""Internal helpers shared by optimizer implementations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from routearena.optimizers.result import OptimizeResult

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient = Callable[[Array], Array]


def as_point(x: Array | list[float] | tuple[float, ...] | float) -> Array:
    """Convert a scalar or sequence to a finite one-dimensional float array."""
    point = np.atleast_1d(np.asarray(x, dtype=float)).copy()
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("x0 must be a finite one-dimensional point")
    return point


def objective_value(fun: Objective, x: Array) -> float:
    """Evaluate a scalar objective and normalize its return type."""
    value = float(fun(x))
    return value


def gradient_value(grad: Gradient, x: Array) -> Array:
    """Evaluate and validate a gradient vector."""
    value = np.asarray(grad(x), dtype=float)
    if value.shape != x.shape:
        raise ValueError(f"gradient has shape {value.shape}; expected {x.shape}")
    return value


def make_result(
    x: Array,
    fun: float,
    path: list[Array],
    nfev: int,
    njev: int,
    success: bool,
    message: str,
) -> OptimizeResult:
    """Build a result while ensuring the saved path does not alias mutable arrays."""
    return OptimizeResult(
        x=x.copy(),
        fun=float(fun),
        path=np.vstack([point.copy() for point in path]),
        nit=len(path) - 1,
        nfev=nfev,
        njev=njev,
        success=success,
        message=message,
    )


def backtracking_step(
    fun: Objective,
    x: Array,
    value: float,
    gradient: Array,
    direction: Array,
    *,
    initial_step: float = 1.0,
    armijo: float = 1e-4,
    shrink: float = 0.5,
    max_steps: int = 50,
) -> tuple[Array, float, int, bool]:
    """Return the first finite point satisfying the Armijo decrease condition."""
    alpha = initial_step
    directional_derivative = float(gradient @ direction)
    evaluations = 0
    for _ in range(max_steps):
        candidate = x + alpha * direction
        candidate_value = objective_value(fun, candidate)
        evaluations += 1
        if (
            np.isfinite(candidate_value)
            and candidate_value <= value + armijo * alpha * directional_derivative
        ):
            return candidate, candidate_value, evaluations, True
        alpha *= shrink
    return x, value, evaluations, False

