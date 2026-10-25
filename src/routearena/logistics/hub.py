"""Smoothed, demand-weighted hub placement in a local planar projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from routearena.logistics.geo import DEFAULT_REFERENCE, latlon_to_local_xy, validate_coordinates


@dataclass
class HubCostModel:
    """Weighted sum of smoothed Euclidean distances from customers to a candidate hub."""

    customers_xy: np.ndarray
    demands: np.ndarray | None = None
    epsilon_m: float = 25.0

    def __post_init__(self) -> None:
        customers = np.asarray(self.customers_xy, dtype=float)
        if customers.ndim != 2 or customers.shape[1] != 2 or customers.shape[0] == 0:
            raise ValueError("customers_xy must be a non-empty array of (x, y) pairs")
        if not np.all(np.isfinite(customers)):
            raise ValueError("customer coordinates must be finite")
        if np.unique(customers, axis=0).shape[0] != customers.shape[0]:
            raise ValueError("duplicate customer coordinates are not allowed")
        if self.demands is None:
            weights = np.ones(customers.shape[0], dtype=float)
        else:
            weights = np.asarray(self.demands, dtype=float)
        if weights.shape != (customers.shape[0],):
            raise ValueError("demands must contain one weight per customer")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("demand weights must be positive and finite")
        if not np.isfinite(self.epsilon_m) or self.epsilon_m <= 0.0:
            raise ValueError("epsilon_m must be positive and finite")
        self.customers_xy = customers
        self.demands = weights

    @classmethod
    def from_latlon(
        cls,
        customer_latlon: np.ndarray,
        demands: np.ndarray | None = None,
        *,
        epsilon_m: float = 25.0,
        reference: tuple[float, float] = DEFAULT_REFERENCE,
    ) -> HubCostModel:
        """Create a hub objective after projecting customer coordinates to local metres."""
        coordinates = validate_coordinates(customer_latlon)
        return cls(latlon_to_local_xy(coordinates, reference=reference), demands, epsilon_m)

    def objective(self, hub_xy: np.ndarray) -> float:
        """Evaluate demand-weighted smoothed distance cost in demand-metres."""
        hub = _validate_hub(hub_xy)
        displacement = hub[None, :] - self.customers_xy
        distances = np.sqrt(np.sum(displacement**2, axis=1) + self.epsilon_m**2)
        return float(np.dot(self.demands, distances))

    def gradient(self, hub_xy: np.ndarray) -> np.ndarray:
        """Return the analytic gradient of the smoothed hub cost."""
        hub = _validate_hub(hub_xy)
        displacement = hub[None, :] - self.customers_xy
        distances = np.sqrt(np.sum(displacement**2, axis=1) + self.epsilon_m**2)
        return np.sum(self.demands[:, None] * displacement / distances[:, None], axis=0)

    def hessian(self, hub_xy: np.ndarray) -> np.ndarray:
        """Return the analytic Hessian of the smoothed hub cost."""
        hub = _validate_hub(hub_xy)
        displacement = hub[None, :] - self.customers_xy
        squared_distances = np.sum(displacement**2, axis=1) + self.epsilon_m**2
        distances = np.sqrt(squared_distances)
        identity = np.eye(2)
        matrices = (
            identity[None, :, :] / distances[:, None, None]
            - displacement[:, :, None] * displacement[:, None, :] / distances[:, None, None] ** 3
        )
        return np.sum(self.demands[:, None, None] * matrices, axis=0)


def _validate_hub(hub_xy: np.ndarray) -> np.ndarray:
    """Convert a candidate hub point to a finite two-dimensional vector."""
    point = np.asarray(hub_xy, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("hub position must be a finite (x, y) pair")
    return point
