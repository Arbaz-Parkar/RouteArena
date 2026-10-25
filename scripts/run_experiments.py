"""Run repeatable optimizer, hub, routing, uncertainty, and RAG experiments.

The default run is modest and CPU-only. Pass ``--quick`` for a short wiring
check; quick output is still real output for the smaller configured sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.backends.backend_agg
import matplotlib.figure
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from routearena.genai.rag.retrieval import build_knowledge_chunks, retrieve
from routearena.logistics.hub import HubCostModel
from routearena.logistics.routing import (
    ant_colony,
    genetic_algorithm,
    held_karp,
    nearest_neighbor,
    two_opt,
)
from routearena.logistics.uncertainty import convergence_curve, interval_route, simulate_route
from routearena.optimizers import adam, bfgs, gradient_descent, newton
from routearena.optimizers.test_functions import (
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)
from routearena.records import write_run_record

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def _optimizer_experiments(quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare from-scratch optimizers and compare hub solutions with SciPy."""
    rng = np.random.default_rng(SEED)
    repeats = 2 if quick else 6
    starts = [rng.uniform(-2.5, 2.5, size=2) for _ in range(repeats)]
    functions = {
        "Quadratic bowl": (quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess),
        "Rosenbrock": (rosenbrock, rosenbrock_grad, rosenbrock_hess),
    }
    optimizers = {
        "Gradient descent": lambda fun, grad, hess, x: gradient_descent(
            fun, grad, x, max_iter=500
        ),
        "Adam": lambda fun, grad, hess, x: adam(
            fun, grad, x, learning_rate=0.03, max_iter=1200
        ),
        "BFGS": lambda fun, grad, hess, x: bfgs(fun, grad, x, max_iter=500),
        "Newton": lambda fun, grad, hess, x: newton(
            fun, grad, x, hess=hess, max_iter=300
        ),
    }
    optimizer_rows: list[dict[str, Any]] = []
    for function_name, (fun, grad, hess) in functions.items():
        for start_number, start in enumerate(starts):
            for method_name, method in optimizers.items():
                began = perf_counter()
                result = method(fun, grad, hess, start.copy())
                optimizer_rows.append(
                    {
                        "group": "optimizer",
                        "problem": function_name,
                        "method": method_name,
                        "repeat": start_number,
                        "objective": float(result.fun),
                        "objective_gap": float(result.fun),
                        "iterations": int(result.nit),
                        "function_evaluations": int(result.nfev),
                        "runtime_seconds": perf_counter() - began,
                        "converged": bool(result.success),
                    }
                )

    hub_rng = np.random.default_rng(SEED + 1)
    customers = hub_rng.normal(0.0, 600.0, size=(9, 2))
    demands = hub_rng.integers(1, 6, size=len(customers)).astype(float)
    model = HubCostModel(customers, demands)
    hub_optimizers = {
        "Gradient descent": lambda x: gradient_descent(
            model.objective, model.gradient, x, max_iter=800
        ),
        "Adam": lambda x: adam(
            model.objective, model.gradient, x, learning_rate=10.0, max_iter=1500
        ),
        "BFGS": lambda x: bfgs(model.objective, model.gradient, x, max_iter=500),
        "Newton": lambda x: newton(
            model.objective, model.gradient, x, hess=model.hessian, max_iter=300
        ),
    }
    hub_rows: list[dict[str, Any]] = []
    for start_number, start in enumerate(starts):
        reference = scipy_minimize(
            model.objective,
            start * 300.0,
            jac=model.gradient,
            method="BFGS",
            options={"gtol": 1e-7, "maxiter": 1000},
        )
        for method_name, method in hub_optimizers.items():
            began = perf_counter()
            result = method(start * 300.0)
            hub_rows.append(
                {
                    "group": "hub_vs_scipy",
                    "problem": "Demand-weighted hub placement",
                    "method": method_name,
                    "repeat": start_number,
                    "objective": float(result.fun),
                    "objective_gap_to_scipy": float(result.fun - reference.fun),
                    "distance_to_scipy_m": float(np.linalg.norm(result.x - reference.x)),
                    "reference_objective": float(reference.fun),
                    "reference_success": bool(reference.success),
                    "reference_iterations": int(reference.nit),
                    "reference_function_evaluations": int(reference.nfev),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "runtime_seconds": perf_counter() - began,
                    "converged": bool(result.success),
                }
            )
    return optimizer_rows, hub_rows


def _random_time_matrix(seed: int, customer_count: int) -> np.ndarray:
    """Make travel times from seeded points in a synthetic ten-kilometre square."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 10.0, size=(customer_count + 1, 2))
    delta = points[:, None, :] - points[None, :, :]
    distance_km = np.sqrt(np.sum(delta**2, axis=2))
    # Apply the same explicit 1.25 detour and 25 km/h speed assumptions used in the app.
    return distance_km * 1.25 / 25.0 * 60.0


def _routing_experiments(quick: bool) -> list[dict[str, Any]]:
    """Compare route heuristics with Held-Karp on several small seeded instances."""
    customer_counts = [5] if quick else [5, 7, 9]
    repeats = 2 if quick else 4
    seeds = [0, 1] if quick else [0, 1, 2, 3]
    rows: list[dict[str, Any]] = []
    for customer_count in customer_counts:
        for repeat in range(repeats):
            matrix = _random_time_matrix(SEED + 100 * customer_count + repeat, customer_count)
            exact = held_karp(matrix)
            for solver_seed in seeds:
                greedy = nearest_neighbor(matrix)
                results = {
                    "Nearest neighbor": greedy,
                    "Nearest neighbor + 2-opt": two_opt(matrix, greedy.route),
                    "Genetic algorithm": genetic_algorithm(
                        matrix,
                        population_size=36 if quick else 60,
                        generations=50 if quick else 120,
                        seed=solver_seed,
                    ),
                    "Ant colony": ant_colony(
                        matrix,
                        ants=max(4, customer_count),
                        iterations=40 if quick else 100,
                        seed=solver_seed,
                    ),
                    "Held-Karp exact": exact,
                }
                for method_name, result in results.items():
                    rows.append(
                        {
                            "group": "routing",
                            "problem": f"{customer_count} customers",
                            "method": method_name,
                            "repeat": repeat,
                            "seed": solver_seed,
                            "cost": float(result.cost),
                            "optimality_gap_percent": float(
                                100.0 * max(0.0, result.cost - exact.cost) / max(exact.cost, 1e-12)
                            ),
                            "runtime_seconds": float(result.runtime_seconds),
                        }
                    )
    return rows


def _uncertainty_experiments(quick: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare shortest and simulated robust candidates, then show MC convergence."""
    matrix = _random_time_matrix(SEED + 300, 6)
    exact = held_karp(matrix)
    greedy = nearest_neighbor(matrix)
    candidates = {
        "Nearest neighbor": greedy,
        "Nearest neighbor + 2-opt": two_opt(matrix, greedy.route),
        "Genetic algorithm": genetic_algorithm(matrix, generations=60, seed=SEED),
        "Ant colony": ant_colony(matrix, iterations=60, seed=SEED),
        "Held-Karp exact": exact,
    }
    sample_count = 1_000 if quick else 10_000
    risk = {
        name: simulate_route(
            matrix,
            result.route,
            deadline_minutes=max(1.0, exact.cost * 1.4),
            traffic_level="moderate",
            sample_count=sample_count,
            seed=SEED + index,
        )
        for index, (name, result) in enumerate(candidates.items())
    }
    shortest_name = min(candidates, key=lambda name: (candidates[name].cost, name))
    robust_name = min(risk, key=lambda name: (risk[name].percentile_90_minutes, name))
    count, mean, percentile = convergence_curve(
        risk[shortest_name].samples,
        checkpoints=[
            int(sample_count * fraction)
            for fraction in ([0.1, 0.25, 0.5, 1.0] if quick else [0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
        ],
    )
    convergence_rows = [
        {
            "group": "uncertainty_convergence",
            "problem": shortest_name,
            "method": "Monte Carlo running estimate",
            "sample_count": int(n),
            "running_mean_minutes": float(m),
            "running_p90_minutes": float(p),
        }
        for n, m, p in zip(count, mean, percentile, strict=True)
    ]
    candidate_rows = [
        {
            "group": "uncertainty_candidates",
            "problem": "Shortest vs simulated robust",
            "method": name,
            "deterministic_minutes": float(candidates[name].cost),
            "mean_minutes": float(risk[name].mean_minutes),
            "p90_minutes": float(risk[name].percentile_90_minutes),
            "probability_late": float(risk[name].probability_late),
            "interval_worst_case_minutes": float(interval_route(matrix, candidates[name].route).worst_case_minutes),
            "selected_as_shortest": name == shortest_name,
            "selected_as_robust": name == robust_name,
        }
        for name in candidates
    ]
    metadata = {
        "sample_count_per_candidate": sample_count,
        "deadline_minutes": float(risk[shortest_name].deadline_minutes),
        "shortest_candidate": shortest_name,
        "robust_candidate_lowest_p90": robust_name,
        "traffic_level": "moderate",
        "note": "Monte Carlo and interval values use documented teaching assumptions, not measured traffic.",
    }
    return [*convergence_rows, *candidate_rows], metadata


def _rag_experiments() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure source hit rate at k=4 and refusal on the hand-written OOS set."""
    questions = json.loads((ROOT / "knowledge" / "eval_questions.json").read_text(encoding="utf-8"))
    chunks = build_knowledge_chunks(ROOT / "knowledge")
    answerable = [item for item in questions if item["expected_source"] is not None]
    out_of_scope = [item for item in questions if item["expected_source"] is None]
    rows: list[dict[str, Any]] = []
    hits = 0
    refusals = 0
    for item in answerable:
        matches = retrieve(item["question"], chunks=chunks, top_k=4, min_score=0.06)
        hit = any(match.chunk.source == item["expected_source"] for match in matches)
        hits += hit
        rows.append(
            {
                "group": "rag_eval",
                "question": item["question"],
                "expected_source": item["expected_source"],
                "top_sources": [match.chunk.source for match in matches],
                "hit_at_4": hit,
                "refused": False,
            }
        )
    for item in out_of_scope:
        matches = retrieve(item["question"], chunks=chunks, top_k=4, min_score=0.06)
        refused = not matches
        refusals += refused
        rows.append(
            {
                "group": "rag_eval",
                "question": item["question"],
                "expected_source": None,
                "top_sources": [match.chunk.source for match in matches],
                "hit_at_4": None,
                "refused": refused,
            }
        )
    summary = {
        "hit_at_4": hits / len(answerable) if answerable else 0.0,
        "hit_count": hits,
        "answerable_count": len(answerable),
        "out_of_scope_refusal_rate": refusals / len(out_of_scope) if out_of_scope else 0.0,
        "refusal_count": refusals,
        "out_of_scope_count": len(out_of_scope),
        "top_k": 4,
        "minimum_score": 0.06,
    }
    return rows, summary


def _library_versions() -> dict[str, str]:
    """Collect versions for reproducibility metadata."""
    packages = ("numpy", "scipy", "matplotlib", "openai", "plotly", "streamlit", "pydantic")
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write heterogeneous result rows with JSON encoding for nested values."""
    columns = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report mean and sample standard deviation across starts and solver seeds."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("group")
        if group in {"optimizer", "hub_vs_scipy", "routing"}:
            groups[(str(group), str(row["problem"]), str(row["method"]))].append(row)

    metrics = (
        "objective",
        "objective_gap",
        "objective_gap_to_scipy",
        "distance_to_scipy_m",
        "iterations",
        "function_evaluations",
        "cost",
        "optimality_gap_percent",
        "runtime_seconds",
    )
    aggregate: list[dict[str, Any]] = []
    for (experiment_group, problem, method), samples in sorted(groups.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in samples if row.get(metric) is not None]
            if not values:
                continue
            aggregate.append(
                {
                    "group": "aggregate",
                    "experiment_group": experiment_group,
                    "problem": problem,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "runs": len(values),
                }
            )
    return aggregate


def _write_plot(rows: list[dict[str, Any]], destination: Path) -> None:
    """Save a compact PNG with optimizer, routing, uncertainty, and retrieval panels."""
    figure = matplotlib.figure.Figure(figsize=(13, 9))
    matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    optimizer_data = [row for row in rows if row.get("group") in {"optimizer", "hub_vs_scipy"}]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in optimizer_data:
        if row.get("objective") is not None:
            grouped[f"{row['problem']}\n{row['method']}"].append(float(row["objective"]))
    if grouped:
        labels = list(grouped)
        axes[0, 0].bar(range(len(labels)), [np.mean(grouped[label]) for label in labels])
        axes[0, 0].set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        axes[0, 0].set_title("Final objective by method and problem")
        axes[0, 0].set_ylabel("Objective (lower is better)")

    routing_data = [row for row in rows if row.get("group") == "routing"]
    route_gaps: dict[str, list[float]] = defaultdict(list)
    for row in routing_data:
        route_gaps[str(row["method"])].append(float(row["optimality_gap_percent"]))
    if route_gaps:
        labels = list(route_gaps)
        axes[0, 1].bar(labels, [np.mean(route_gaps[label]) for label in labels])
        axes[0, 1].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0, 1].set_title("Routing heuristic gap from Held-Karp")
        axes[0, 1].set_ylabel("Mean optimality gap (%)")

    convergence = [row for row in rows if row.get("group") == "uncertainty_convergence"]
    if convergence:
        axes[1, 0].plot(
            [row["sample_count"] for row in convergence],
            [row["running_mean_minutes"] for row in convergence],
            marker="o",
            label="Running mean",
        )
        axes[1, 0].plot(
            [row["sample_count"] for row in convergence],
            [row["running_p90_minutes"] for row in convergence],
            marker="s",
            label="Running 90th percentile",
        )
        axes[1, 0].set_title("Monte Carlo convergence")
        axes[1, 0].set_xlabel("Simulated days")
        axes[1, 0].set_ylabel("Estimated minutes")
        axes[1, 0].legend()

    rag = next((row for row in rows if row.get("group") == "rag_eval_summary"), None)
    if rag:
        axes[1, 1].bar(
            ["Hit at k=4", "OOS refusal"],
            [100.0 * rag["hit_at_4"], 100.0 * rag["out_of_scope_refusal_rate"]],
        )
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_ylabel("Rate (%)")
        axes[1, 1].set_title("Local RAG evaluation")
    figure.suptitle("RouteArena reproducible experiments")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")


def run(quick: bool, output_dir: Path) -> dict[str, Any]:
    """Run all P0 experiment groups and save raw rows plus metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_rows, hub_rows = _optimizer_experiments(quick)
    routing_rows = _routing_experiments(quick)
    uncertainty_rows, uncertainty_summary = _uncertainty_experiments(quick)
    rag_rows, rag_summary = _rag_experiments()
    summary_rows = [{"group": "rag_eval_summary", **rag_summary}]
    raw_rows = [*optimizer_rows, *hub_rows, *routing_rows, *uncertainty_rows, *rag_rows, *summary_rows]
    aggregate_rows = _aggregate_rows(raw_rows)
    all_rows = [*raw_rows, *aggregate_rows]
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "quick_mode": quick,
        "versions_and_machine": _library_versions(),
        "row_counts": {
            "optimizers": len(optimizer_rows),
            "hub_vs_scipy": len(hub_rows),
            "routing": len(routing_rows),
            "uncertainty": len(uncertainty_rows),
            "rag_questions": len(rag_rows),
            "aggregate_metrics": len(aggregate_rows),
        },
        "uncertainty": uncertainty_summary,
        "rag": rag_summary,
        "notes": [
            "All random generators use recorded deterministic seeds.",
            "Hub SciPy results are a comparison reference, not the application solver.",
            "Synthetic routing points occupy a 10 km square; time uses a 1.25 detour factor and 25 km/h assumed speed.",
            "Traffic model values are assumptions and do not represent measured travel times.",
            "Quick mode evaluates smaller real samples; do not present it as the full sweep.",
        ],
    }
    run_record = write_run_record(
        "experiment-sweep",
        config={"quick_mode": quick, "output_dir": str(output_dir), "seed": SEED},
        seed=SEED,
        results={"row_counts": metadata["row_counts"], "rag": rag_summary, "uncertainty": uncertainty_summary},
        runs_dir=ROOT / "runs",
    )
    metadata["run_record"] = str(run_record)
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_results.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(all_rows, output_dir / "experiment_results.csv")
    _write_csv(aggregate_rows, output_dir / "experiment_summary.csv")
    _write_plot(all_rows, output_dir / "summary.png")
    return metadata


def main() -> None:
    """Parse CLI flags and report output paths and actual RAG metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a small real sample for a short run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="directory for ignored experiment artifacts (default: outputs/)",
    )
    args = parser.parse_args()
    try:
        metadata = run(args.quick, args.output_dir)
    except ImportError as exc:
        raise SystemExit("Experiment dependency missing. Install with: python -m pip install -e '.[dev]'") from exc
    print(f"Saved experiment artifacts under {args.output_dir.resolve()}")
    print(
        "RAG hit@4: "
        f"{metadata['rag']['hit_count']}/{metadata['rag']['answerable_count']} "
        f"= {metadata['rag']['hit_at_4']:.1%}; OOS refusal: "
        f"{metadata['rag']['refusal_count']}/{metadata['rag']['out_of_scope_count']} "
        f"= {metadata['rag']['out_of_scope_refusal_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
