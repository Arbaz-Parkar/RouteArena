"""Geography and hub-cost model checks."""

import numpy as np
import pytest

from routearena.logistics.geo import (
    DEFAULT_REFERENCE,
    distance_matrix,
    haversine_distance,
    latlon_to_local_xy,
    local_xy_to_latlon,
    travel_time_matrix,
    validate_coordinates,
)
from routearena.logistics.hub import HubCostModel


def test_haversine_and_distance_matrix_have_expected_values():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    matrix = distance_matrix(points)

    assert haversine_distance(tuple(points[0]), tuple(points[1])) == pytest.approx(
        111_195, rel=1e-3
    )
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)


def test_travel_time_matrix_uses_configured_assumptions():
    points = np.array([[19.03, 73.02], [19.04, 73.02]])
    matrix = travel_time_matrix(points, detour_factor=1.2, average_speed_kmph=24.0)

    assert matrix[0, 1] == pytest.approx(matrix[1, 0])
    assert matrix[0, 1] > 0.0
    assert matrix[0, 0] == 0.0


def test_projection_round_trip_is_accurate_for_a_small_region():
    points = np.array([[19.03, 73.02], [19.04, 73.01], [19.02, 73.03]])
    projected = latlon_to_local_xy(points, reference=DEFAULT_REFERENCE)

    np.testing.assert_allclose(
        local_xy_to_latlon(projected, reference=DEFAULT_REFERENCE),
        points,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "points",
    [
        np.array([[91.0, 0.0]]),
        np.array([[0.0, 181.0]]),
        np.array([[np.nan, 0.0]]),
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        np.zeros((31, 2)),
    ],
)
def test_invalid_coordinate_sets_are_rejected(points):
    with pytest.raises(ValueError):
        validate_coordinates(points)


def test_hub_gradient_matches_central_differences():
    model = HubCostModel(
        np.array([[-1000.0, 300.0], [500.0, -800.0], [1200.0, 900.0]]),
        demands=np.array([1.0, 3.0, 2.0]),
    )
    point = np.array([200.0, -100.0])
    step = 1e-4
    numerical = np.empty(2)
    for index in range(2):
        offset = np.zeros(2)
        offset[index] = step
        numerical[index] = (model.objective(point + offset) - model.objective(point - offset)) / (
            2 * step
        )

    np.testing.assert_allclose(model.gradient(point), numerical, rtol=2e-5, atol=2e-5)


def test_hub_hessian_matches_gradient_differences():
    model = HubCostModel(np.array([[-500.0, 300.0], [700.0, -900.0]]))
    point = np.array([150.0, -100.0])
    step = 1e-3
    numerical = np.column_stack(
        [
            (
                model.gradient(point + np.eye(2)[index] * step)
                - model.gradient(point - np.eye(2)[index] * step)
            )
            / (2 * step)
            for index in range(2)
        ]
    )

    np.testing.assert_allclose(model.hessian(point), numerical, rtol=2e-5, atol=2e-5)
