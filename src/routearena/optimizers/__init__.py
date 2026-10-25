"""Optimization algorithms and scalar search methods."""

from routearena.optimizers.first_order import adadelta, adagrad, adam, gradient_descent, rmsprop
from routearena.optimizers.line_search import (
    fibonacci_search,
    golden_section_search,
    quadratic_fit_search,
)
from routearena.optimizers.result import OptimizeResult
from routearena.optimizers.second_order import bfgs, newton, secant_minimize

__all__ = [
    "OptimizeResult",
    "adadelta",
    "adagrad",
    "adam",
    "bfgs",
    "fibonacci_search",
    "golden_section_search",
    "gradient_descent",
    "newton",
    "quadratic_fit_search",
    "rmsprop",
    "secant_minimize",
]

