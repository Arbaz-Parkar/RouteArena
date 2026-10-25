"""Second-order and quasi-Newton optimization methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from routearena.optimizers._common import (
    Array,
    Gradient,
    Objective,
    as_point,
    backtracking_step,
    gradient_value,
    make_result,
    objective_value,
)
from routearena.optimizers.result import OptimizeResult

Hessian = Callable[[Array], Array]


def newton(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    hess: Hessian | None = None,
    grad_tol: float = 1e-6,
    max_iter: int = 500,
) -> OptimizeResult:
    """Newton's method with an analytic or finite-difference Hessian and backtracking."""
    _validate_options(grad_tol, max_iter)
    x = as_point(x0)
    value = objective_value(fun, x)
    nfev, njev = 1, 0
    path = [x.copy()]
    if not np.isfinite(value):
        return make_result(x, value, path, nfev, njev, False, "objective is not finite at x0")

    for _ in range(max_iter + 1):
        gradient = gradient_value(grad, x)
        njev += 1
        if not np.all(np.isfinite(gradient)):
            return make_result(x, value, path, nfev, njev, False, "gradient is not finite")
        if np.linalg.norm(gradient) <= grad_tol:
            return make_result(x, value, path, nfev, njev, True, "gradient tolerance reached")
        if len(path) - 1 >= max_iter:
            break
        if hess is None:
            matrix, gradient_calls = _finite_difference_hessian(grad, x)
            njev += gradient_calls
        else:
            matrix = np.asarray(hess(x), dtype=float)
        if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
            return make_result(x, value, path, nfev, njev, False, "Hessian is invalid")
        try:
            direction = -np.linalg.solve(matrix, gradient)
        except np.linalg.LinAlgError:
            direction = -gradient
        if float(gradient @ direction) >= 0.0:
            direction = -gradient
        candidate, candidate_value, used, accepted = backtracking_step(
            fun, x, value, gradient, direction
        )
        nfev += used
        if not accepted:
            return make_result(x, value, path, nfev, njev, False, "line search failed")
        x, value = candidate, candidate_value
        path.append(x.copy())

    return make_result(x, value, path, nfev, njev, False, "maximum iterations reached")


def secant_minimize(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...] | float,
    *,
    x1: float | None = None,
    grad_tol: float = 1e-8,
    max_iter: int = 100,
) -> OptimizeResult:
    """Find a one-dimensional stationary point by applying secant updates to f'."""
    _validate_options(grad_tol, max_iter)
    first_point = as_point(x0)
    if first_point.size != 1:
        raise ValueError("secant_minimize is for one-dimensional objectives")
    first = float(first_point[0])
    second = float(x1 if x1 is not None else first + 0.1 * (abs(first) + 1.0))
    if first == second:
        raise ValueError("x0 and x1 must be distinct")
    path = [np.array([first]), np.array([second])]
    first_value, second_value = objective_value(fun, path[0]), objective_value(fun, path[1])
    nfev, njev = 2, 0
    first_derivative = float(gradient_value(grad, path[0])[0])
    second_derivative = float(gradient_value(grad, path[1])[0])
    njev += 2
    if not all(
        np.isfinite(v)
        for v in (first_value, second_value, first_derivative, second_derivative)
    ):
        return make_result(
            path[-1], second_value, path, nfev, njev, False, "non-finite initial value"
        )

    if abs(first_derivative) <= grad_tol:
        return make_result(
            path[0], first_value, [path[0]], nfev, njev, True, "gradient tolerance reached"
        )

    for _ in range(max_iter):
        denominator = second_derivative - first_derivative
        if abs(denominator) <= np.finfo(float).eps:
            return make_result(
                path[-1], second_value, path, nfev, njev, False, "secant slope is zero"
            )
        candidate_value = second - second_derivative * (second - first) / denominator
        candidate = np.array([candidate_value])
        value = objective_value(fun, candidate)
        derivative = float(gradient_value(grad, candidate)[0])
        nfev += 1
        njev += 1
        if not np.isfinite(value) or not np.isfinite(derivative):
            return make_result(
                path[-1], second_value, path, nfev, njev, False, "iteration diverged"
            )
        path.append(candidate)
        first, second = second, candidate_value
        first_derivative, second_derivative = second_derivative, derivative
        first_value, second_value = second_value, value
        if abs(derivative) <= grad_tol:
            return make_result(
                candidate, value, path, nfev, njev, True, "gradient tolerance reached"
            )
    return make_result(
        path[-1], second_value, path, nfev, njev, False, "maximum iterations reached"
    )


def bfgs(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    grad_tol: float = 1e-6,
    max_iter: int = 2000,
) -> OptimizeResult:
    """BFGS inverse-Hessian updates with an Armijo backtracking line search."""
    _validate_options(grad_tol, max_iter)
    x = as_point(x0)
    value = objective_value(fun, x)
    nfev, njev = 1, 0
    path = [x.copy()]
    if not np.isfinite(value):
        return make_result(x, value, path, nfev, njev, False, "objective is not finite at x0")
    inverse_hessian = np.eye(x.size)

    for _ in range(max_iter + 1):
        gradient = gradient_value(grad, x)
        njev += 1
        if not np.all(np.isfinite(gradient)):
            return make_result(x, value, path, nfev, njev, False, "gradient is not finite")
        if np.linalg.norm(gradient) <= grad_tol:
            return make_result(x, value, path, nfev, njev, True, "gradient tolerance reached")
        if len(path) - 1 >= max_iter:
            break
        direction = -(inverse_hessian @ gradient)
        if float(gradient @ direction) >= 0.0:
            inverse_hessian = np.eye(x.size)
            direction = -gradient
        candidate, candidate_value, used, accepted = backtracking_step(
            fun, x, value, gradient, direction
        )
        nfev += used
        if not accepted:
            return make_result(x, value, path, nfev, njev, False, "line search failed")
        candidate_gradient = gradient_value(grad, candidate)
        njev += 1
        if not np.all(np.isfinite(candidate_gradient)):
            return make_result(x, value, path, nfev, njev, False, "gradient is not finite")
        displacement = candidate - x
        gradient_change = candidate_gradient - gradient
        curvature = float(displacement @ gradient_change)
        if curvature > 1e-12:
            rho = 1.0 / curvature
            identity = np.eye(x.size)
            left = identity - rho * np.outer(displacement, gradient_change)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(displacement, displacement)
            )
        else:
            inverse_hessian = np.eye(x.size)
        x, value = candidate, candidate_value
        path.append(x.copy())

    return make_result(x, value, path, nfev, njev, False, "maximum iterations reached")


def _finite_difference_hessian(grad: Gradient, x: Array) -> tuple[Array, int]:
    """Approximate the Hessian by central differences of the gradient."""
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
    matrix = np.empty((x.size, x.size), dtype=float)
    calls = 0
    for column in range(x.size):
        offset = np.zeros_like(x)
        offset[column] = step[column]
        high = gradient_value(grad, x + offset)
        low = gradient_value(grad, x - offset)
        calls += 2
        matrix[:, column] = (high - low) / (2.0 * step[column])
    return 0.5 * (matrix + matrix.T), calls


def _validate_options(grad_tol: float, max_iter: int) -> None:
    if grad_tol <= 0 or not np.isfinite(grad_tol):
        raise ValueError("grad_tol must be positive and finite")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

