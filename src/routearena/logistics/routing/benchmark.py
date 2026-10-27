"""Repeatable multi-seed benchmark for the TSPLIB burma14 instance."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from routearena.logistics.routing import (
    ant_colony,
    genetic_algorithm,
    held_karp,
    nearest_neighbor,
    ortools_reference,
    two_opt,
)
from routearena.logistics.routing.tsplib import (
    BURMA14_OPTIMAL_COST,
    burma14_distance_matrix,
)
from routearena.records import write_run_record

DEFAULT_OUTPUT_DIR = Path("outputs") / "routing-study"
SEED_COUNT = 30
SEED_START = 2026
_T_95_DF_29 = 2.045229642
_GENETIC_SETTINGS = {"population_size": 60, "generations": 120}
_ANT_COLONY_SETTINGS = {"ants": 20, "iterations": 100}


def _environment() -> dict[str, Any]:
    """Capture software versions and machine details for reproducibility."""
    packages = ("numpy", "plotly", "streamlit", "ortools")
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "logical_cpu_count": os.cpu_count(),
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write flat study rows, serializing list and mapping values as JSON."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _statistics(values: list[float]) -> dict[str, float | None]:
    """Return sample statistics and a two-sided 95% Student-t interval for 30 seeds."""
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return {
            "mean": mean,
            "standard_deviation": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    standard_deviation = float(np.std(data, ddof=1))
    margin = _T_95_DF_29 * standard_deviation / sqrt(len(data))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def _summary_row(
    method: str,
    method_type: str,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize cost, optimum gap, and runtime for one method."""
    if not trials:
        return {
            "method": method,
            "method_type": method_type,
            "runs": 0,
            "status": "unavailable",
        }
    costs = [float(row["cost_km"]) for row in trials]
    gaps = [float(row["optimality_gap_percent"]) for row in trials]
    runtimes = [float(row["runtime_seconds"]) for row in trials]
    cost_stats = _statistics(costs)
    gap_stats = _statistics(gaps)
    runtime_stats = _statistics(runtimes)
    return {
        "method": method,
        "method_type": method_type,
        "runs": len(trials),
        "status": "available",
        "mean_cost_km": cost_stats["mean"],
        "standard_deviation_cost_km": cost_stats["standard_deviation"],
        "ci95_cost_lower_km": cost_stats["ci95_lower"],
        "ci95_cost_upper_km": cost_stats["ci95_upper"],
        "median_cost_km": float(np.median(costs)),
        "minimum_cost_km": float(np.min(costs)),
        "maximum_cost_km": float(np.max(costs)),
        "mean_optimality_gap_percent": gap_stats["mean"],
        "standard_deviation_gap_percent": gap_stats["standard_deviation"],
        "ci95_gap_lower_percent": gap_stats["ci95_lower"],
        "ci95_gap_upper_percent": gap_stats["ci95_upper"],
        "mean_runtime_seconds": runtime_stats["mean"],
        "standard_deviation_runtime_seconds": runtime_stats["standard_deviation"],
        "ci95_runtime_lower_seconds": runtime_stats["ci95_lower"],
        "ci95_runtime_upper_seconds": runtime_stats["ci95_upper"],
    }


def run_burma14_study(*, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run fixed-seed GA and ACO trials against the exact TSPLIB burma14 optimum.

    The confidence intervals describe variation across solver seeds on this one
    benchmark instance. They do not estimate performance across all TSP instances.
    """
    matrix = burma14_distance_matrix()
    exact = held_karp(matrix, max_customers=13)
    if exact.cost != BURMA14_OPTIMAL_COST:
        raise RuntimeError(
            f"Held-Karp returned {exact.cost:g} km; expected "
            f"the TSPLIB optimum of {BURMA14_OPTIMAL_COST} km"
        )

    trials: list[dict[str, Any]] = []
    seeds = list(range(SEED_START, SEED_START + SEED_COUNT))
    for seed in seeds:
        results = (
            (
                "Genetic algorithm",
                genetic_algorithm(
                    matrix,
                    **_GENETIC_SETTINGS,
                    seed=seed,
                ),
            ),
            (
                "Ant colony",
                ant_colony(
                    matrix,
                    **_ANT_COLONY_SETTINGS,
                    seed=seed,
                ),
            ),
        )
        for method, result in results:
            trials.append(
                {
                    "instance": "TSPLIB burma14",
                    "method": method,
                    "seed": seed,
                    "cost_km": float(result.cost),
                    "known_optimum_km": float(BURMA14_OPTIMAL_COST),
                    "optimality_gap_percent": float(
                        100.0 * (result.cost - BURMA14_OPTIMAL_COST) / BURMA14_OPTIMAL_COST
                    ),
                    "runtime_seconds": float(result.runtime_seconds),
                    "route": list(result.route),
                    "algorithm_settings": dict(
                        _GENETIC_SETTINGS if method == "Genetic algorithm" else _ANT_COLONY_SETTINGS
                    ),
                }
            )

    greedy = nearest_neighbor(matrix)
    deterministic_results = [
        ("Nearest neighbor", greedy),
        ("Nearest neighbor + 2-opt", two_opt(matrix, greedy.route)),
        ("Held-Karp exact", exact),
    ]
    reference_trials: list[dict[str, Any]] = []
    for method, result in deterministic_results:
        reference_trials.append(
            {
                "instance": "TSPLIB burma14",
                "method": method,
                "seed": None,
                "cost_km": float(result.cost),
                "known_optimum_km": float(BURMA14_OPTIMAL_COST),
                "optimality_gap_percent": float(
                    100.0 * (result.cost - BURMA14_OPTIMAL_COST) / BURMA14_OPTIMAL_COST
                ),
                "runtime_seconds": float(result.runtime_seconds),
                "route": list(result.route),
            }
        )

    ortools_available = True
    try:
        ortools_result = ortools_reference(matrix, time_limit_seconds=2)
    except ImportError:
        ortools_available = False
    else:
        reference_trials.append(
            {
                "instance": "TSPLIB burma14",
                "method": "OR-Tools reference",
                "seed": None,
                "cost_km": float(ortools_result.cost),
                "known_optimum_km": float(BURMA14_OPTIMAL_COST),
                "optimality_gap_percent": float(
                    100.0 * (ortools_result.cost - BURMA14_OPTIMAL_COST) / BURMA14_OPTIMAL_COST
                ),
                "runtime_seconds": float(ortools_result.runtime_seconds),
                "route": list(ortools_result.route),
            }
        )

    combined_trials = [*trials, *reference_trials]
    summary = [
        _summary_row(
            "Genetic algorithm",
            "stochastic; fixed-seed trials",
            [row for row in trials if row["method"] == "Genetic algorithm"],
        ),
        _summary_row(
            "Ant colony",
            "stochastic; fixed-seed trials",
            [row for row in trials if row["method"] == "Ant colony"],
        ),
        *[
            _summary_row(method, "single-run reference", [row])
            for method, row in ((row["method"], row) for row in reference_trials)
        ],
    ]
    if not ortools_available:
        summary.append(_summary_row("OR-Tools reference", "optional single-run reference", []))

    settings = {
        "instance": "TSPLIB burma14",
        "cities": int(matrix.shape[0]),
        "known_optimum_km": int(BURMA14_OPTIMAL_COST),
        "solver_seed_start": SEED_START,
        "solver_seed_count": SEED_COUNT,
        "solver_seeds": seeds,
        "genetic_algorithm": dict(_GENETIC_SETTINGS),
        "ant_colony": dict(_ANT_COLONY_SETTINGS),
        "ortools_time_limit_seconds": 2,
        "confidence_interval": (
            "Two-sided 95% Student-t interval for the mean across 30 seeds; "
            "df=29; mean +/- 2.045229642 * sample SD / sqrt(30)."
        ),
    }
    environment = _environment()
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "environment": environment,
        "ortools_available": ortools_available,
        "interpretation": (
            "Seed intervals measure stochastic solver variation on burma14 only; "
            "they do not establish general performance across TSP instances. "
            "Runtime depends on the recorded machine and software environment."
        ),
        "sources": {
            "instance": "https://raw.githubusercontent.com/mastqe/tsplib/master/burma14.tsp",
            "known_optimum": "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html",
        },
    }

    record_path = write_run_record(
        "routing-study-burma14",
        config=settings,
        seed=SEED_START,
        results={
            "summary": summary,
            "trials": combined_trials,
            "environment": environment,
        },
        runs_dir=Path("runs"),
    )
    metadata["run_record"] = str(record_path)
    report = {"metadata": metadata, "summary": summary, "trials": combined_trials}

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "routing_study_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / "routing_study_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(combined_trials, destination / "routing_study_trials.csv")
    _write_csv(summary, destination / "routing_study_summary.csv")
    return report
