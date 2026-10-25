"""Small TSPLIB benchmark data and the TSPLIB GEO distance conversion."""

from __future__ import annotations

import numpy as np

BURMA14_OPTIMAL_COST = 3_323
_GEO_RADIUS_KM = 6_378.388

# TSPLIB's burma14.tsp node coordinates, stored as degrees.minutes (not decimal degrees).
_BURMA14_GEO_COORDINATES = np.array(
    [
        [16.47, 96.10],
        [16.47, 94.44],
        [20.09, 92.54],
        [22.39, 93.37],
        [25.23, 97.24],
        [22.00, 96.05],
        [20.47, 97.02],
        [17.20, 96.29],
        [16.30, 97.38],
        [14.05, 98.12],
        [16.53, 97.38],
        [21.52, 95.59],
        [19.41, 97.13],
        [20.09, 94.55],
    ],
    dtype=float,
)


def _tsplib_geo_radians(degrees_minutes: np.ndarray) -> np.ndarray:
    """Convert TSPLIB's ``DD.MM`` encoding to radians."""
    degrees = np.floor(degrees_minutes)
    minutes = degrees_minutes - degrees
    decimal_degrees = degrees + (5.0 * minutes / 3.0)
    return np.pi * decimal_degrees / 180.0


def burma14_distance_matrix() -> np.ndarray:
    """Build the 14-city TSPLIB ``burma14`` integer GEO distance matrix in km."""
    latitude = _tsplib_geo_radians(_BURMA14_GEO_COORDINATES[:, 0])
    longitude = _tsplib_geo_radians(_BURMA14_GEO_COORDINATES[:, 1])
    delta_longitude = longitude[:, None] - longitude[None, :]
    q1 = np.cos(delta_longitude)
    q2 = np.cos(latitude[:, None] - latitude[None, :])
    q3 = np.cos(latitude[:, None] + latitude[None, :])
    cosine = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    distances = np.floor(_GEO_RADIUS_KM * np.arccos(cosine) + 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances
