"""Tests for sampling, from-scratch surrogates, and Bayesian optimization."""

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from routearena.surrogate.bayes import bayesian_optimization, expected_improvement
from routearena.surrogate.models import (
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

_BOUNDS = np.array([[-2.0, 2.0], [0.0, 3.0]])


def test_sampling_plans_are_reproducible_and_respect_bounds() -> None:
    random_a = random_sampling(_BOUNDS, 16, seed=8)
    random_b = random_sampling(_BOUNDS, 16, seed=8)
    lhs = latin_hypercube(_BOUNDS, 16, seed=8)
    stratified = stratified_sampling(_BOUNDS, 4, seed=8)

    assert np.array_equal(random_a, random_b)
    assert random_a.shape == lhs.shape == (16, 2)
    assert stratified.shape == (16, 2)
    assert np.all(random_a >= _BOUNDS[:, 0]) and np.all(random_a <= _BOUNDS[:, 1])
    assert full_factorial(_BOUNDS, 3).shape == (9, 2)
    for column in range(2):
        normalized = (lhs[:, column] - _BOUNDS[column, 0]) / np.diff(_BOUNDS[column])[0]
        assert set(np.floor(normalized * 16).astype(int)) == set(range(16))


def test_centered_discrepancy_matches_scipy_validation_reference() -> None:
    points = latin_hypercube(np.array([[0.0, 1.0]] * 3), 18, seed=17)
    metrics = space_filling_metrics(points)
    assert np.isclose(metrics["centered_discrepancy"], qmc.discrepancy(points, method="CD"), atol=1e-10)
    assert metrics["minimum_pairwise_distance"] > 0.0


def test_linear_basis_and_validation_selection_fit_quadratic_signal() -> None:
    x = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = 2.0 + 3.0 * x[:, 0] - 1.5 * x[:, 0] ** 2
    model = fit_linear_surrogate(x, y, degree=2)
    selected = select_linear_model(x, y, seed=4)

    assert np.allclose(model.predict(x), y, atol=1e-10)
    assert selected.model.degree == 2
    assert selected.validation_rmse < 1e-10


def test_gaussian_process_matches_sklearn_reference() -> None:
    x = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = np.sin(5.0 * x[:, 0]) + 0.2 * x[:, 0]
    query = np.linspace(0.05, 0.95, 9).reshape(-1, 1)
    model = fit_gaussian_process(x, y, length_scale=0.22, noise=1e-6)
    mean, deviation = model.predict(query)
    reference = GaussianProcessRegressor(
        kernel=RBF(length_scale=0.22), alpha=model.noise, optimizer=None, normalize_y=True
    ).fit(x, y)
    reference_mean, reference_deviation = reference.predict(query, return_std=True)

    assert np.allclose(mean, reference_mean, atol=1e-8)
    assert np.allclose(deviation, reference_deviation, atol=1e-8)


def test_expected_improvement_and_bayesian_budget() -> None:
    assert np.allclose(expected_improvement(np.array([0.0]), np.array([0.0]), 1.0), [0.99])
    objective = lambda point: float(np.sum((point - np.array([0.25, 1.5])) ** 2))
    result = bayesian_optimization(
        objective, _BOUNDS, initial_samples=5, iterations=4, candidate_count=64, seed=3
    )

    assert result.points.shape == (9, 2)
    assert result.values.shape == (9,)
    assert np.all(result.points >= _BOUNDS[:, 0])
    assert np.all(result.points <= _BOUNDS[:, 1])
    assert np.all(np.diff(result.best_history) <= 0.0)
