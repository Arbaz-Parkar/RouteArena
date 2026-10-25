"""One-dimensional line searches used by the optimization methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarSearchResult:
    """Minimum estimate and work performed by a bounded scalar search."""

    x: float
    fun: float
    nfev: int
    nit: int
    success: bool


def golden_section_search(
    fun: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> ScalarSearchResult:
    """Minimize a unimodal function on a closed interval using golden sections."""
    _validate_interval(lower, upper, tol, max_iter)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = float(fun(left))
    f_right = float(fun(right))
    nfev = 2
    for iteration in range(max_iter):
        if upper - lower <= tol:
            break
        if f_left <= f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = float(fun(left))
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = float(fun(right))
        nfev += 1
    x = (lower + upper) / 2.0
    return ScalarSearchResult(x, float(fun(x)), nfev + 1, iteration + 1, upper - lower <= tol)


def fibonacci_search(
    fun: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> ScalarSearchResult:
    """Minimize a unimodal function with the finite-interval Fibonacci search."""
    _validate_interval(lower, upper, tol, max_iter)
    fib = [0, 1]
    while fib[-1] < (upper - lower) / tol and len(fib) - 1 < max_iter:
        fib.append(fib[-1] + fib[-2])
    order = len(fib) - 1
    if order < 3:
        return golden_section_search(fun, lower, upper, tol=tol, max_iter=max_iter)

    x_left = lower + fib[order - 2] / fib[order] * (upper - lower)
    x_right = lower + fib[order - 1] / fib[order] * (upper - lower)
    f_left, f_right = float(fun(x_left)), float(fun(x_right))
    nfev = 2
    iterations = 0
    for offset in range(1, order - 1):
        iterations += 1
        if upper - lower <= tol:
            break
        if f_left > f_right:
            lower, x_left, f_left = x_left, x_right, f_right
            numerator = fib[order - offset - 1]
            denominator = fib[order - offset]
            x_right = lower + numerator / denominator * (upper - lower)
            f_right = float(fun(x_right))
        else:
            upper, x_right, f_right = x_right, x_left, f_left
            numerator = fib[order - offset - 2]
            denominator = fib[order - offset]
            x_left = lower + numerator / denominator * (upper - lower)
            f_left = float(fun(x_left))
        nfev += 1
    x = (lower + upper) / 2.0
    return ScalarSearchResult(
        x,
        float(fun(x)),
        nfev + 1,
        iterations,
        upper - lower <= max(tol, 1e-12),
    )


def quadratic_fit_search(
    fun: Callable[[float], float],
    x_left: float,
    x_middle: float,
    x_right: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> ScalarSearchResult:
    """Fit a parabola through a bracketing triple and refine its vertex."""
    if not x_left < x_middle < x_right:
        raise ValueError("the three points must be strictly increasing")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")
    f_left, f_middle, f_right = map(float, (fun(x_left), fun(x_middle), fun(x_right)))
    nfev = 3
    if not f_middle <= f_left or not f_middle <= f_right:
        raise ValueError("x_middle must have an objective value no greater than either endpoint")

    success = False
    for iteration in range(max_iter):
        if x_right - x_left <= tol:
            success = True
            break
        numerator = (
            (x_middle - x_left) ** 2 * (f_middle - f_right)
            - (x_middle - x_right) ** 2 * (f_middle - f_left)
        )
        denominator = 2.0 * (
            (x_middle - x_left) * (f_middle - f_right)
            - (x_middle - x_right) * (f_middle - f_left)
        )
        if abs(denominator) <= np.finfo(float).eps:
            candidate = (x_left + x_right) / 2.0
        else:
            candidate = x_middle - numerator / denominator
        margin = min(x_middle - x_left, x_right - x_middle) * 0.1
        candidate = float(np.clip(candidate, x_left + margin, x_right - margin))
        f_candidate = float(fun(candidate))
        nfev += 1
        if abs(candidate - x_middle) <= tol:
            success = True
            if f_candidate < f_middle:
                x_middle, f_middle = candidate, f_candidate
            break
        if candidate < x_middle:
            if f_candidate <= f_middle:
                x_right, f_right = x_middle, f_middle
                x_middle, f_middle = candidate, f_candidate
            else:
                x_left, f_left = candidate, f_candidate
        elif f_candidate <= f_middle:
            x_left, f_left = x_middle, f_middle
            x_middle, f_middle = candidate, f_candidate
        else:
            x_right, f_right = candidate, f_candidate
    x = x_middle
    return ScalarSearchResult(
        x,
        f_middle,
        nfev,
        iteration + 1,
        success or x_right - x_left <= tol,
    )


def bracketed_line_search(
    fun: Callable[[float], float],
    *,
    initial_step: float = 1.0,
    tol: float = 1e-7,
    max_expand: int = 30,
    max_iter: int = 100,
) -> ScalarSearchResult:
    """Expand from zero to bracket a descent direction, then use golden sections."""
    if initial_step <= 0:
        raise ValueError("initial_step must be positive")
    f_zero = float(fun(0.0))
    step = initial_step
    f_step = float(fun(step))
    nfev = 2
    if not np.isfinite(f_step) or f_step >= f_zero:
        bracket = (0.0, step)
    else:
        lower = 0.0
        upper = step
        for _ in range(max_expand):
            next_step = upper * 2.0
            f_next = float(fun(next_step))
            nfev += 1
            if not np.isfinite(f_next) or f_next >= f_step:
                lower, upper = lower, next_step
                break
            lower = upper
            upper, f_step = next_step, f_next
        else:
            lower, upper = 0.0, upper
        bracket = (lower, upper)
    result = golden_section_search(fun, *bracket, tol=tol, max_iter=max_iter)
    return ScalarSearchResult(
        result.x,
        result.fun,
        nfev + result.nfev,
        result.nit,
        result.success,
    )


def _validate_interval(lower: float, upper: float, tol: float, max_iter: int) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("lower and upper must be finite with lower < upper")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")

