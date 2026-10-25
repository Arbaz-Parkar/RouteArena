"""Sampling, surrogate models, and Bayesian tuning helpers."""

from routearena.surrogate.bayes import (
    SearchResult,
    bayesian_optimization,
    expected_improvement,
    random_search,
)
from routearena.surrogate.models import (
    GaussianProcessSurrogate,
    LinearSurrogate,
    fit_gaussian_process,
    fit_linear_surrogate,
    select_linear_model,
)
from routearena.surrogate.sampling import (
    full_factorial,
    latin_hypercube,
    random_sampling,
    space_filling_metrics,
    stratified_sampling,
)

__all__ = [
    "GaussianProcessSurrogate",
    "LinearSurrogate",
    "SearchResult",
    "bayesian_optimization",
    "expected_improvement",
    "fit_gaussian_process",
    "fit_linear_surrogate",
    "full_factorial",
    "latin_hypercube",
    "random_sampling",
    "random_search",
    "select_linear_model",
    "space_filling_metrics",
    "stratified_sampling",
]
