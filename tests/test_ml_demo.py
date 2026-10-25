"""Tests for the seeded How AI Learns demonstration."""

import numpy as np

from routearena.ml.regression_demo import run_regression_demo


def test_ml_demo_is_reproducible_and_reports_held_out_errors() -> None:
    first = run_regression_demo(seed=29)
    second = run_regression_demo(seed=29)

    assert first["test_mse"] == second["test_mse"]
    assert first["train_count"] + first["test_count"] == first["sample_count"]
    assert {"Gradient descent", "Adam", "Closed form"}.issubset(first["test_mse"])
    assert all(np.isfinite(value) and value >= 0.0 for value in first["test_mse"].values())
