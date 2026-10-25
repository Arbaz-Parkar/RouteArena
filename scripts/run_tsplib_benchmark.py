"""Compare RouteArena and optional OR-Tools solvers on TSPLIB burma14."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from routearena.logistics.routing import ant_colony, genetic_algorithm, held_karp
from routearena.logistics.routing.ortools_reference import ortools_reference
from routearena.logistics.routing.tsplib import (
    BURMA14_OPTIMAL_COST,
    burma14_distance_matrix,
)
from routearena.records import write_run_record

SEED = 2026
ROOT = Path(__file__).resolve().parents[1]


def _result_summary(result, optimum: float) -> dict[str, Any]:
    return {
        "route": list(result.route),
        "cost_km": result.cost,
        "known_optimum_km": optimum,
        "relative_gap": result.cost / optimum - 1.0,
        "runtime_seconds": result.runtime_seconds,
        "algorithm": result.algorithm,
        "metadata": result.metadata,
    }


def run_benchmark() -> dict[str, Any]:
    """Run the exact benchmark and seeded heuristic comparisons."""
    matrix = burma14_distance_matrix()
    exact = held_karp(matrix, max_customers=13)
    if exact.cost != BURMA14_OPTIMAL_COST:
        raise RuntimeError(
            f"TSPLIB reference check failed: expected {BURMA14_OPTIMAL_COST}, "
            f"got {exact.cost:g}"
        )

    summaries = {
        "held_karp": _result_summary(exact, BURMA14_OPTIMAL_COST),
        "genetic_algorithm": _result_summary(
            genetic_algorithm(matrix, population_size=60, generations=120, seed=SEED),
            BURMA14_OPTIMAL_COST,
        ),
        "ant_colony": _result_summary(
            ant_colony(matrix, ants=20, iterations=100, seed=SEED),
            BURMA14_OPTIMAL_COST,
        ),
    }
    try:
        ortools_result = ortools_reference(matrix, time_limit_seconds=2)
    except ImportError:
        summaries["ortools_reference"] = {
            "available": False,
            "install": "python -m pip install -e '.[benchmarks]'",
        }
    else:
        summaries["ortools_reference"] = {
            "available": True,
            **_result_summary(ortools_result, BURMA14_OPTIMAL_COST),
        }

    report = {
        "instance": "TSPLIB burma14",
        "cities": int(matrix.shape[0]),
        "known_optimum_km": BURMA14_OPTIMAL_COST,
        "seed": SEED,
        "results": summaries,
        "sources": {
            "coordinates": "https://raw.githubusercontent.com/mastqe/tsplib/master/burma14.tsp",
            "known_optimum": "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html",
        },
    }
    write_run_record(
        "tsplib-burma14",
        config={"instance": "burma14", "seed": SEED},
        seed=SEED,
        results=report,
        runs_dir=ROOT / "runs",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
