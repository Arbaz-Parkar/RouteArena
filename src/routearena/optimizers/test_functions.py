"""Small smooth test surfaces with analytic derivatives."""

import numpy as np


def quadratic_bowl(x: np.ndarray) -> float:
    """A shifted, mildly conditioned quadratic with its minimum at (1, -1)."""
    point = np.asarray(x, dtype=float)
    delta = point - np.array([1.0, -1.0])
    return float(0.5 * (delta[0] ** 2 + 4.0 * delta[1] ** 2))


def quadratic_bowl_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`quadratic_bowl`."""
    point = np.asarray(x, dtype=float)
    return np.array([point[0] - 1.0, 4.0 * (point[1] + 1.0)])


def quadratic_bowl_hess(x: np.ndarray) -> np.ndarray:
    """Analytic Hessian of :func:`quadratic_bowl` (independent of x)."""
    del x
    return np.diag([1.0, 4.0])


def rosenbrock(x: np.ndarray) -> float:
    """The two-dimensional Rosenbrock valley, minimized at (1, 1)."""
    point = np.asarray(x, dtype=float)
    return float(100.0 * (point[1] - point[0] ** 2) ** 2 + (1.0 - point[0]) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the two-dimensional Rosenbrock function."""
    point = np.asarray(x, dtype=float)
    return np.array(
        [
            -400.0 * point[0] * (point[1] - point[0] ** 2) - 2.0 * (1.0 - point[0]),
            200.0 * (point[1] - point[0] ** 2),
        ]
    )


def rosenbrock_hess(x: np.ndarray) -> np.ndarray:
    """Analytic Hessian of the two-dimensional Rosenbrock function."""
    point = np.asarray(x, dtype=float)
    return np.array(
        [
            [1200.0 * point[0] ** 2 - 400.0 * point[1] + 2.0, -400.0 * point[0]],
            [-400.0 * point[0], 200.0],
        ]
    )


def parabola(x: np.ndarray | float) -> float:
    """One-dimensional quadratic with minimum at x=3."""
    value = float(np.asarray(x, dtype=float).reshape(-1)[0])
    return (value - 3.0) ** 2


def parabola_grad(x: np.ndarray | float) -> np.ndarray:
    """Analytic one-dimensional derivative of :func:`parabola`."""
    value = float(np.asarray(x, dtype=float).reshape(-1)[0])
    return np.array([2.0 * (value - 3.0)])
