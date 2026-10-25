"""Small from-scratch linear-basis and Gaussian-process surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _training_data(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("inputs must be a two-dimensional array with at least two rows")
    if y.shape != (x.shape[0],):
        raise ValueError("targets must contain one value per input row")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training data must be finite")
    return x, y


def polynomial_features(inputs: np.ndarray, degree: int = 1) -> np.ndarray:
    """Build a linear or quadratic polynomial feature matrix with an intercept."""
    x = np.asarray(inputs, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError("inputs must be a finite two-dimensional array")
    if degree not in {1, 2}:
        raise ValueError("degree must be 1 or 2")
    columns = [np.ones(x.shape[0])]
    columns.extend(x[:, index] for index in range(x.shape[1]))
    if degree == 2:
        columns.extend(x[:, index] ** 2 for index in range(x.shape[1]))
        columns.extend(
            x[:, left] * x[:, right]
            for left in range(x.shape[1])
            for right in range(left + 1, x.shape[1])
        )
    return np.column_stack(columns)


@dataclass(frozen=True)
class LinearSurrogate:
    """Least-squares regression over the selected polynomial basis."""

    coefficients: np.ndarray
    degree: int

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict target values at one or more feature vectors."""
        return polynomial_features(inputs, self.degree) @ self.coefficients


def fit_linear_surrogate(
    inputs: np.ndarray, targets: np.ndarray, *, degree: int = 1
) -> LinearSurrogate:
    """Fit a polynomial-basis linear model with a stable least-squares solve."""
    x, y = _training_data(inputs, targets)
    features = polynomial_features(x, degree)
    coefficients, *_ = np.linalg.lstsq(features, y, rcond=None)
    return LinearSurrogate(coefficients=coefficients, degree=degree)


@dataclass(frozen=True)
class LinearModelSelection:
    """Chosen basis and validation scores from one reproducible split."""

    model: LinearSurrogate
    validation_rmse: float
    scores_by_degree: dict[int, float]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def select_linear_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    degrees: tuple[int, ...] = (1, 2),
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> LinearModelSelection:
    """Select a basis by held-out RMSE, with a fixed seeded train/validation split."""
    x, y = _training_data(inputs, targets)
    if not degrees or any(degree not in {1, 2} for degree in degrees):
        raise ValueError("degrees must contain 1 and/or 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(x.shape[0])
    validation_count = max(1, int(round(x.shape[0] * validation_fraction)))
    if validation_count >= x.shape[0]:
        raise ValueError("not enough samples for both training and validation")
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    scores: dict[int, float] = {}
    models: dict[int, LinearSurrogate] = {}
    for degree in degrees:
        model = fit_linear_surrogate(x[train_indices], y[train_indices], degree=degree)
        error = model.predict(x[validation_indices]) - y[validation_indices]
        scores[degree] = float(np.sqrt(np.mean(error**2)))
        models[degree] = model
    chosen_degree = min(scores, key=lambda degree: (scores[degree], degree))
    return LinearModelSelection(
        model=models[chosen_degree],
        validation_rmse=scores[chosen_degree],
        scores_by_degree=scores,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def _rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    distances = (left[:, None, :] - right[None, :, :]) / length_scale
    return np.exp(-0.5 * np.sum(distances**2, axis=2))


@dataclass(frozen=True)
class GaussianProcessSurrogate:
    """RBF Gaussian process with constant observation noise and Cholesky solves."""

    inputs: np.ndarray
    target_mean: float
    target_scale: float
    length_scale: float
    noise: float
    cholesky: np.ndarray
    weights: np.ndarray

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and standard deviation for query points."""
        query = np.asarray(inputs, dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != self.inputs.shape[1]:
            raise ValueError("query inputs must match the training feature count")
        if not np.all(np.isfinite(query)):
            raise ValueError("query inputs must be finite")
        cross = _rbf_kernel(query, self.inputs, self.length_scale)
        normalized_mean = cross @ self.weights
        triangular_solve = np.linalg.solve(self.cholesky, cross.T)
        variance = np.maximum(0.0, 1.0 - np.sum(triangular_solve**2, axis=0))
        return (
            self.target_mean + self.target_scale * normalized_mean,
            self.target_scale * np.sqrt(variance),
        )


def fit_gaussian_process(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    length_scale: float = 0.25,
    noise: float = 1e-6,
) -> GaussianProcessSurrogate:
    """Fit an RBF GP, increasing diagonal jitter only if Cholesky needs it."""
    x, y = _training_data(inputs, targets)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be positive and finite")
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError("noise must be nonnegative and finite")
    mean = float(np.mean(y))
    scale = float(np.std(y))
    if scale <= np.finfo(float).eps:
        scale = 1.0
    normalized_y = (y - mean) / scale
    kernel = _rbf_kernel(x, x, length_scale)
    jitter = 1e-12
    identity = np.eye(x.shape[0])
    for _ in range(8):
        try:
            cholesky = np.linalg.cholesky(kernel + (noise + jitter) * identity)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("could not stabilize the GP covariance matrix")
    weights = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, normalized_y))
    return GaussianProcessSurrogate(
        inputs=x.copy(),
        target_mean=mean,
        target_scale=scale,
        length_scale=float(length_scale),
        noise=float(noise + jitter),
        cholesky=cholesky,
        weights=weights,
    )
