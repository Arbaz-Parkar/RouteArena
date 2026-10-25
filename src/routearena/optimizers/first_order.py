"""First-order optimization methods implemented with NumPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from routearena.optimizers._common import (
    Array,
    Gradient,
    Objective,
    as_point,
    gradient_value,
    make_result,
    objective_value,
)
from routearena.optimizers.line_search import bracketed_line_search
from routearena.optimizers.result import OptimizeResult


def gradient_descent(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    grad_tol: float = 1e-6,
    max_iter: int = 1000,
    line_search_tol: float = 1e-7,
) -> OptimizeResult:
    """Gradient descent with a bracketed golden-section search on each step."""
    _validate_options(grad_tol, max_iter)
    x = as_point(x0)
    value = objective_value(fun, x)
    nfev, njev = 1, 0
    path = [x.copy()]
    if not np.isfinite(value):
        return make_result(x, value, path, nfev, njev, False, "objective is not finite at x0")

    for _ in range(max_iter):
        gradient = gradient_value(grad, x)
        njev += 1
        if not np.all(np.isfinite(gradient)):
            return make_result(x, value, path, nfev, njev, False, "gradient is not finite")
        if np.linalg.norm(gradient) <= grad_tol:
            return make_result(x, value, path, nfev, njev, True, "gradient tolerance reached")
        direction = -gradient
        search = bracketed_line_search(
            lambda alpha: objective_value(fun, x + alpha * direction),
            tol=line_search_tol,
        )
        nfev += search.nfev
        candidate = x + search.x * direction
        candidate_value = objective_value(fun, candidate)
        nfev += 1
        if not np.isfinite(candidate_value):
            return make_result(x, value, path, nfev, njev, False, "line search produced a non-finite value")
        if candidate_value >= value and np.linalg.norm(candidate - x) <= line_search_tol:
            return make_result(x, value, path, nfev, njev, False, "line search could not make progress")
        x, value = candidate, candidate_value
        path.append(x.copy())

    return make_result(x, value, path, nfev, njev, False, "maximum iterations reached")


def _fixed_step_method(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    learning_rate: float,
    grad_tol: float,
    max_iter: int,
    update: Callable[[Array, int], Array],
) -> OptimizeResult:
    """Run a gradient method whose update rule is supplied by a small closure."""
    _validate_options(grad_tol, max_iter)
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    x = as_point(x0)
    value = objective_value(fun, x)
    nfev, njev = 1, 0
    path = [x.copy()]
    if not np.isfinite(value):
        return make_result(x, value, path, nfev, njev, False, "objective is not finite at x0")

    for iteration in range(max_iter):
        gradient = gradient_value(grad, x)
        njev += 1
        if not np.all(np.isfinite(gradient)):
            return make_result(x, value, path, nfev, njev, False, "gradient is not finite")
        if np.linalg.norm(gradient) <= grad_tol:
            return make_result(x, value, path, nfev, njev, True, "gradient tolerance reached")
        candidate = x + update(gradient, iteration)
        if not np.all(np.isfinite(candidate)):
            return make_result(x, value, path, nfev, njev, False, "update produced a non-finite point")
        candidate_value = objective_value(fun, candidate)
        nfev += 1
        if not np.isfinite(candidate_value):
            return make_result(x, value, path, nfev, njev, False, "objective became non-finite")
        x, value = candidate, candidate_value
        path.append(x.copy())

    last_gradient = gradient_value(grad, x)
    njev += 1
    converged = np.all(np.isfinite(last_gradient)) and np.linalg.norm(last_gradient) <= grad_tol
    message = "gradient tolerance reached" if converged else "maximum iterations reached"
    return make_result(x, value, path, nfev, njev, bool(converged), message)


def adagrad(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    learning_rate: float = 0.1,
    epsilon: float = 1e-8,
    grad_tol: float = 1e-6,
    max_iter: int = 2000,
) -> OptimizeResult:
    """Adagrad scales each coordinate by its accumulated squared gradients."""
    accumulator: Array | None = None

    def update(g: Array, _: int) -> Array:
        nonlocal accumulator
        accumulator = g**2 if accumulator is None else accumulator + g**2
        return -learning_rate * g / (np.sqrt(accumulator) + epsilon)

    return _fixed_step_method(
        fun,
        grad,
        x0,
        learning_rate=learning_rate,
        grad_tol=grad_tol,
        max_iter=max_iter,
        update=update,
    )


def rmsprop(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    learning_rate: float = 0.01,
    decay: float = 0.9,
    epsilon: float = 1e-8,
    grad_tol: float = 1e-6,
    max_iter: int = 2000,
) -> OptimizeResult:
    """RMSProp divides each gradient by a moving average of its square."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    average: Array | None = None

    def update(g: Array, _: int) -> Array:
        nonlocal average
        average = g**2 if average is None else decay * average + (1.0 - decay) * g**2
        return -learning_rate * g / (np.sqrt(average) + epsilon)

    return _fixed_step_method(
        fun,
        grad,
        x0,
        learning_rate=learning_rate,
        grad_tol=grad_tol,
        max_iter=max_iter,
        update=update,
    )


def adadelta(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    decay: float = 0.95,
    epsilon: float = 1e-6,
    grad_tol: float = 1e-6,
    max_iter: int = 3000,
) -> OptimizeResult:
    """Adadelta uses running averages of squared gradients and squared updates."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    average_gradient: Array | None = None
    average_update: Array | None = None

    def update(g: Array, _: int) -> Array:
        nonlocal average_gradient, average_update
        if average_gradient is None:
            average_gradient = np.zeros_like(g)
            average_update = np.zeros_like(g)
        average_gradient = decay * average_gradient + (1.0 - decay) * g**2
        delta = -np.sqrt(average_update + epsilon) / np.sqrt(average_gradient + epsilon) * g
        average_update = decay * average_update + (1.0 - decay) * delta**2
        return delta

    return _fixed_step_method(
        fun,
        grad,
        x0,
        learning_rate=1.0,
        grad_tol=grad_tol,
        max_iter=max_iter,
        update=update,
    )


def adam(
    fun: Objective,
    grad: Gradient,
    x0: Array | list[float] | tuple[float, ...],
    *,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    grad_tol: float = 1e-6,
    max_iter: int = 3000,
) -> OptimizeResult:
    """Adam combines bias-corrected moving averages of gradients and squares."""
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta1 and beta2 must be in [0, 1)")
    first: Array | None = None
    second: Array | None = None

    def update(g: Array, iteration: int) -> Array:
        nonlocal first, second
        if first is None:
            first, second = np.zeros_like(g), np.zeros_like(g)
        first = beta1 * first + (1.0 - beta1) * g
        second = beta2 * second + (1.0 - beta2) * g**2
        corrected_first = first / (1.0 - beta1 ** (iteration + 1))
        corrected_second = second / (1.0 - beta2 ** (iteration + 1))
        return -learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

    return _fixed_step_method(
        fun,
        grad,
        x0,
        learning_rate=learning_rate,
        grad_tol=grad_tol,
        max_iter=max_iter,
        update=update,
    )


def _validate_options(grad_tol: float, max_iter: int) -> None:
    if grad_tol <= 0 or not np.isfinite(grad_tol):
        raise ValueError("grad_tol must be positive and finite")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

