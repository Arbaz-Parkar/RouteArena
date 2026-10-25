"""Known-solution and comparison tests for optimization algorithms."""

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from routearena.optimizers import (
    adadelta,
    adagrad,
    adam,
    bfgs,
    gradient_descent,
    newton,
    rmsprop,
    secant_minimize,
)
from routearena.optimizers.test_functions import (
    parabola,
    parabola_grad,
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)


@pytest.mark.parametrize(
    ("method", "options"),
    [
        (gradient_descent, {}),
        (adagrad, {"learning_rate": 0.2}),
        (rmsprop, {"learning_rate": 0.001}),
        (adadelta, {}),
        (adam, {"learning_rate": 0.001}),
    ],
)
def test_first_order_methods_reach_quadratic_minimum(method, options):
    result = method(
        quadratic_bowl,
        quadratic_bowl_grad,
        [-2.0, 3.0],
        grad_tol=5e-3,
        max_iter=8000,
        **options,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.0, -1.0], atol=3e-3)
    assert result.path.shape[1] == 2
    assert result.nfev >= result.nit + 1


@pytest.mark.parametrize("method", [newton, bfgs])
def test_second_order_methods_match_scipy_on_rosenbrock(method):
    options = {"hess": rosenbrock_hess} if method is newton else {}
    result = method(
        rosenbrock,
        rosenbrock_grad,
        [-1.2, 1.0],
        grad_tol=1e-5,
        max_iter=2000,
        **options,
    )
    reference = scipy_minimize(
        rosenbrock,
        np.array([-1.2, 1.0]),
        jac=rosenbrock_grad,
        method="BFGS",
        tol=1e-9,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.x, reference.x, atol=3e-4)
    assert result.fun == pytest.approx(reference.fun, abs=1e-8)


def test_newton_uses_the_analytic_hessian_on_quadratic():
    result = newton(
        quadratic_bowl,
        quadratic_bowl_grad,
        [-3.0, 6.0],
        hess=quadratic_bowl_hess,
    )

    assert result.success
    np.testing.assert_allclose(result.x, [1.0, -1.0], atol=1e-8)


def test_secant_finds_one_dimensional_stationary_point():
    result = secant_minimize(parabola, parabola_grad, -2.0, x1=0.5)

    assert result.success
    np.testing.assert_allclose(result.x, [3.0], atol=1e-7)
    assert result.fun == pytest.approx(0.0, abs=1e-12)


def test_non_finite_objective_is_reported_without_crashing():
    def objective(x):
        if abs(x[0]) > 1.0:
            return float("inf")
        return -x[0]

    result = gradient_descent(objective, lambda x: np.array([-1.0]), [0.0], max_iter=5)

    assert not result.success
    assert np.all(np.isfinite(result.x))
    assert result.message


@pytest.mark.parametrize(
    ("function", "gradient", "point"),
    [
        (quadratic_bowl, quadratic_bowl_grad, np.array([0.2, -0.7])),
        (rosenbrock, rosenbrock_grad, np.array([-0.8, 1.3])),
    ],
)
def test_analytic_gradients_match_central_differences(function, gradient, point):
    step = 1e-6
    numerical = np.empty_like(point)
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        numerical[index] = (function(point + offset) - function(point - offset)) / (2.0 * step)

    np.testing.assert_allclose(gradient(point), numerical, rtol=1e-6, atol=1e-6)

