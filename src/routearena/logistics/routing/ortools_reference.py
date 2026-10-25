"""Optional OR-Tools reference solver for comparing small symmetric routes."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from routearena.logistics.routing.common import RouteResult, make_result, validate_time_matrix


def ortools_reference(
    time_matrix: np.ndarray,
    *,
    time_limit_seconds: int = 2,
    cost_scale: int = 1_000,
) -> RouteResult:
    """Find a feasible route with OR-Tools; its result is a heuristic reference.

    OR-Tools' routing engine works with integer arc costs, so input minutes are
    scaled and rounded for the search. ``RouteResult.cost`` is recalculated from
    the original matrix. The reference is not guaranteed to be globally optimal.
    """
    matrix = validate_time_matrix(time_matrix)
    if not isinstance(time_limit_seconds, int) or time_limit_seconds < 1:
        raise ValueError("time_limit_seconds must be a positive integer")
    if not isinstance(cost_scale, int) or cost_scale < 1:
        raise ValueError("cost_scale must be a positive integer")
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise ImportError(
            "OR-Tools is optional. Install it with: python -m pip install -e '.[benchmarks]'"
        ) from exc

    node_count = matrix.shape[0]
    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cost(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(matrix[from_node, to_node] * cost_scale))

    callback_index = routing.RegisterTransitCallback(transit_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = time_limit_seconds

    started_at = perf_counter()
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools did not find a feasible route within the time limit")

    route = [0]
    index = routing.Start(0)
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
    return make_result(
        matrix,
        route,
        started_at,
        "ortools_reference",
        metadata={
            "time_limit_seconds": time_limit_seconds,
            "cost_scale": cost_scale,
            "scaled_objective": int(solution.ObjectiveValue()),
            "optimality_proven": False,
        },
    )
