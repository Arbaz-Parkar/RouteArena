"""Synthetic delivery-time regression trained with RouteArena optimizers."""

from __future__ import annotations

from typing import Any

import numpy as np

from routearena.optimizers import adam, gradient_descent


def run_regression_demo(seed: int = 0) -> dict[str, Any]:
    """Train a small linear model and compare it with closed-form and sklearn fits."""
    rng = np.random.default_rng(seed)
    sample_count = 320
    distance_km = rng.uniform(0.5, 30.0, size=sample_count)
    stop_count = rng.integers(1, 21, size=sample_count).astype(float)
    traffic_level = rng.uniform(0.0, 1.0, size=sample_count)
    features = np.column_stack((distance_km, stop_count, traffic_level))
    target = (
        5.0
        + 2.3 * distance_km
        + 1.15 * stop_count
        + 8.0 * traffic_level
        + rng.normal(0.0, 2.5, size=sample_count)
    )

    order = rng.permutation(sample_count)
    train_indices = order[: int(0.8 * sample_count)]
    test_indices = order[int(0.8 * sample_count) :]
    means = np.mean(features[train_indices], axis=0)
    scales = np.std(features[train_indices], axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    train_x = (features[train_indices] - means) / scales
    test_x = (features[test_indices] - means) / scales
    train_design = np.column_stack((np.ones(train_x.shape[0]), train_x))
    test_design = np.column_stack((np.ones(test_x.shape[0]), test_x))
    train_y = target[train_indices]
    test_y = target[test_indices]

    def objective(weights: np.ndarray) -> float:
        error = train_design @ weights - train_y
        return float(0.5 * np.mean(error**2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return train_design.T @ (train_design @ weights - train_y) / train_y.size

    initial = np.zeros(train_design.shape[1], dtype=float)
    descent = gradient_descent(objective, gradient, initial, grad_tol=1e-7, max_iter=400)
    adaptive = adam(
        objective, gradient, initial, learning_rate=0.05, grad_tol=1e-7, max_iter=2500
    )
    closed_form, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)

    models: dict[str, np.ndarray] = {
        "Gradient descent": descent.x,
        "Adam": adaptive.x,
        "Closed form": closed_form,
    }
    sklearn_error: float | None = None
    try:
        from sklearn.linear_model import LinearRegression

        sklearn_model = LinearRegression(fit_intercept=False).fit(train_design, train_y)
        models["scikit-learn"] = sklearn_model.coef_
        sklearn_error = float(np.mean((sklearn_model.predict(test_design) - test_y) ** 2))
    except ImportError:
        pass

    losses = {
        "Gradient descent": [objective(point) for point in descent.path],
        "Adam": [objective(point) for point in adaptive.path],
    }
    test_mse = {
        name: float(np.mean((test_design @ weights - test_y) ** 2))
        for name, weights in models.items()
    }
    if sklearn_error is not None:
        test_mse["scikit-learn"] = sklearn_error
    return {
        "features": ["distance_km", "stop_count", "traffic_level"],
        "sample_count": sample_count,
        "train_count": int(train_y.size),
        "test_count": int(test_y.size),
        "test_mse": test_mse,
        "loss_history": losses,
        "iterations": {"Gradient descent": descent.nit, "Adam": adaptive.nit},
        "converged": {"Gradient descent": descent.success, "Adam": adaptive.success},
        "sklearn_available": sklearn_error is not None,
        "note": "Synthetic teaching data generated from a known linear delivery-time rule plus noise.",
    }
