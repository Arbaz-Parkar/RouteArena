"""Geographic distance and simple travel-time assumptions."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
MAX_STOPS = 30
DEFAULT_REFERENCE = (19.03, 73.02)


def validate_coordinates(points: np.ndarray, *, max_stops: int = MAX_STOPS) -> np.ndarray:
    """Validate a non-empty array of unique (latitude, longitude) pairs."""
    if max_stops < 1:
        raise ValueError("max_stops must be positive")
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("coordinates must be a non-empty array of (latitude, longitude) pairs")
    if values.shape[0] > max_stops:
        raise ValueError(f"at most {max_stops} stops are supported")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinates must be finite")
    latitudes, longitudes = values[:, 0], values[:, 1]
    if np.any((latitudes < -90.0) | (latitudes > 90.0)):
        raise ValueError("latitude must be between -90 and 90 degrees")
    if np.any((longitudes < -180.0) | (longitudes > 180.0)):
        raise ValueError("longitude must be between -180 and 180 degrees")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise ValueError("duplicate coordinates are not allowed")
    return values


def haversine_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Return the great-circle distance in metres between two latitude/longitude pairs."""
    a = _validate_coordinate(point_a)
    b = _validate_coordinate(point_b)
    lat_a, lon_a = np.radians(a)
    lat_b, lon_b = np.radians(b)
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    central_angle = 2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(max(0.0, 1.0 - haversine)))
    return float(EARTH_RADIUS_M * central_angle)


def distance_matrix(points: np.ndarray) -> np.ndarray:
    """Build a symmetric great-circle distance matrix in metres."""
    coordinates = validate_coordinates(points)
    radians = np.radians(coordinates)
    latitudes = radians[:, 0]
    longitudes = radians[:, 1]
    delta_lat = latitudes[:, None] - latitudes[None, :]
    delta_lon = longitudes[:, None] - longitudes[None, :]
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(latitudes[:, None])
        * np.cos(latitudes[None, :])
        * np.sin(delta_lon / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    angles = 2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(1.0 - haversine))
    result = EARTH_RADIUS_M * angles
    np.fill_diagonal(result, 0.0)
    return result


def travel_time_matrix(
    points: np.ndarray,
    *,
    detour_factor: float = 1.25,
    average_speed_kmph: float = 20.0,
) -> np.ndarray:
    """Estimate symmetric travel times in minutes from distance and explicit assumptions."""
    if not np.isfinite(detour_factor) or detour_factor < 1.0:
        raise ValueError("detour_factor must be finite and at least 1")
    if not np.isfinite(average_speed_kmph) or average_speed_kmph <= 0.0:
        raise ValueError("average_speed_kmph must be positive and finite")
    distance_km = distance_matrix(points) / 1000.0
    return distance_km * detour_factor / average_speed_kmph * 60.0


def latlon_to_local_xy(
    points: np.ndarray,
    *,
    reference: tuple[float, float] = DEFAULT_REFERENCE,
) -> np.ndarray:
    """Project latitude/longitude to local equirectangular x/y coordinates in metres.

    The local projection uses the reference latitude for the longitude scale. It is suitable
    for small areas around the reference point, not for large regions or navigation.
    """
    coordinates = validate_coordinates(points)
    ref_lat, ref_lon = _validate_coordinate(reference)
    latitudes = np.radians(coordinates[:, 0])
    longitudes = np.radians(coordinates[:, 1])
    reference_latitude = np.radians(ref_lat)
    reference_longitude = np.radians(ref_lon)
    x = EARTH_RADIUS_M * np.cos(reference_latitude) * (longitudes - reference_longitude)
    y = EARTH_RADIUS_M * (latitudes - reference_latitude)
    return np.column_stack((x, y))


def local_xy_to_latlon(
    points_xy: np.ndarray,
    *,
    reference: tuple[float, float] = DEFAULT_REFERENCE,
) -> np.ndarray:
    """Convert local equirectangular x/y coordinates in metres to latitude/longitude."""
    values = np.asarray(points_xy, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("local coordinates must be a finite array of (x, y) pairs")
    ref_lat, ref_lon = _validate_coordinate(reference)
    reference_latitude = np.radians(ref_lat)
    latitudes = ref_lat + np.degrees(values[:, 1] / EARTH_RADIUS_M)
    longitude_scale = EARTH_RADIUS_M * np.cos(reference_latitude)
    if abs(longitude_scale) < 1e-12:
        raise ValueError("the reference latitude is too close to a pole for this projection")
    longitudes = ref_lon + np.degrees(values[:, 0] / longitude_scale)
    result = np.column_stack((latitudes, longitudes))
    if np.any((latitudes < -90.0) | (latitudes > 90.0)):
        raise ValueError("projected point falls outside valid latitude bounds")
    if np.any((longitudes < -180.0) | (longitudes > 180.0)):
        raise ValueError("projected point falls outside valid longitude bounds")
    return result


def _validate_coordinate(point: tuple[float, float] | np.ndarray) -> np.ndarray:
    """Validate one latitude/longitude pair without applying duplicate-point rules."""
    value = np.asarray(point, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("coordinate must be a finite (latitude, longitude) pair")
    latitude, longitude = value
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be between -180 and 180 degrees")
    return value

