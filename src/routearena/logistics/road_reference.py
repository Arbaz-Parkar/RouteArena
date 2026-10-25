"""Optional OpenStreetMap road-distance reference using OSMnx."""

from __future__ import annotations

import numpy as np

from routearena.logistics.geo import DEFAULT_REFERENCE, validate_coordinates


def osmnx_road_distance_matrix(
    points: np.ndarray,
    *,
    graph=None,
    center: tuple[float, float] = DEFAULT_REFERENCE,
    radius_m: float = 5_000.0,
    network_type: str = "drive",
) -> np.ndarray:
    """Return directed shortest road distances in metres between coordinates.

    If no graph is provided, this downloads a drivable OSM graph around ``center``.
    Coordinates snap to their nearest graph node. The matrix preserves one-way-road
    direction and can therefore be asymmetric; RouteArena's from-scratch solvers
    currently require symmetric matrices, so this function is a reference only.
    OSM speeds and traffic are not inferred by this distance-only helper.
    """
    coordinates = validate_coordinates(points)
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite")
    if not network_type:
        raise ValueError("network_type must be a non-empty string")
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "OSMnx is optional. Install it with: python -m pip install -e '.[maps]'"
        ) from exc

    if graph is None:
        graph = ox.graph.graph_from_point(
            center,
            dist=radius_m,
            network_type=network_type,
            simplify=True,
        )
    nodes = ox.distance.nearest_nodes(
        graph,
        X=coordinates[:, 1],
        Y=coordinates[:, 0],
    )
    node_ids = np.asarray(nodes, dtype=np.int64).reshape(-1)
    distances = np.full((len(node_ids), len(node_ids)), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for row, source in enumerate(node_ids):
        lengths = nx.single_source_dijkstra_path_length(graph, int(source), weight="length")
        for column, target in enumerate(node_ids):
            if int(target) in lengths:
                distances[row, column] = float(lengths[int(target)])
    if not np.all(np.isfinite(distances)):
        raise ValueError(
            "the downloaded road graph does not connect every snapped stop "
            "in the requested direction"
        )
    return distances
