"""Compare Bayesian optimization and random search over repeated ACO tuning runs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.backends.backend_agg
import matplotlib.figure
import numpy as np

from routearena.records import write_run_record
from routearena.surrogate.tuning import run_tuning_pipeline, synthetic_time_matrix

ROOT = Path(__file__).resolve().parents[1]


def run(quick: bool, output_dir: Path) -> dict:
    """Run multiple fixed-budget comparisons and persist raw and aggregate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repetition_count = 2 if quick else 5
    design_size = 12 if quick else 18
    budget = 8 if quick else 12
    matrix = synthetic_time_matrix(seed=527, customer_count=8)
    rows = []
    histories = {"Bayesian optimization": [], "Random search": []}

    for repetition in range(repetition_count):
        result = run_tuning_pipeline(
            matrix,
            plan="Latin hypercube",
            design_size=design_size,
            search_budget=budget,
            seed=100 + repetition,
        )
        bayesian = result["bayesian"]
        random = result["random"]
        histories["Bayesian optimization"].append(bayesian.best_history)
        histories["Random search"].append(random.best_history)
        rows.append(
            {
                "repetition": repetition,
                "seed": 100 + repetition,
                "design_points": len(result["design_values"]),
                "selected_degree": result["selected_degree"],
                "linear_validation_rmse": result["linear_validation_rmse"],
                "gp_validation_rmse": result["gp_validation_rmse"],
                "bayesian_best_minutes": bayesian.best_value,
                "random_best_minutes": random.best_value,
                "bayesian_parameters": bayesian.best_point.tolist(),
                "random_parameters": random.best_point.tolist(),
                "evaluation_budget": result["budget"],
                "design_metrics": result["design_metrics"],
            }
        )

    final_bayesian = np.asarray([row["bayesian_best_minutes"] for row in rows])
    final_random = np.asarray([row["random_best_minutes"] for row in rows])
    summary = {
        "bayesian_mean_minutes": float(np.mean(final_bayesian)),
        "bayesian_std_minutes": float(np.std(final_bayesian, ddof=1)) if len(rows) > 1 else 0.0,
        "random_mean_minutes": float(np.mean(final_random)),
        "random_std_minutes": float(np.std(final_random, ddof=1)) if len(rows) > 1 else 0.0,
    }
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "quick_mode": quick,
        "seed_rule": "one fixed routing instance; repetitions use solver/design seeds 100 onward",
        "repetitions": repetition_count,
        "design_size": design_size,
        "evaluation_budget_per_method": budget,
        "objective": "Mean ACO travel time across three fixed seeds",
        "summary": summary,
        "versions_and_machine": _versions(),
        "note": (
            "A lower objective is better; report the observed values even if random search wins."
        ),
    }
    record = write_run_record(
        "tuning-comparison",
        config={"quick_mode": quick, "repetitions": repetition_count, "seed": 527},
        seed=527,
        results={"summary": summary, "repetitions": rows},
        runs_dir=ROOT / "runs",
    )
    metadata["run_record"] = str(record)
    (output_dir / "tuning_results.json").write_text(
        json.dumps({"metadata": metadata, "runs": rows}, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "tuning_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    with (output_dir / "tuning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "mean_minutes", "std_minutes", "runs"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "method": "Bayesian optimization",
                    "mean_minutes": summary["bayesian_mean_minutes"],
                    "std_minutes": summary["bayesian_std_minutes"],
                    "runs": repetition_count,
                },
                {
                    "method": "Random search",
                    "mean_minutes": summary["random_mean_minutes"],
                    "std_minutes": summary["random_std_minutes"],
                    "runs": repetition_count,
                },
            ]
        )
    _save_plot(histories, output_dir / "tuning_comparison.png")
    return metadata


def _versions() -> dict[str, str]:
    packages = ("numpy", "matplotlib", "scipy", "scikit-learn", "streamlit")
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _save_plot(histories: dict[str, list[np.ndarray]], destination: Path) -> None:
    figure = matplotlib.figure.Figure(figsize=(8, 5))
    matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
    axis = figure.subplots()
    for name, runs in histories.items():
        values = np.vstack(runs)
        axis.plot(
            np.arange(1, values.shape[1] + 1),
            np.mean(values, axis=0),
            marker="o",
            label=name,
        )
    axis.set_xlabel("Objective evaluation")
    axis.set_ylabel("Mean ACO route time (minutes)")
    axis.set_title("Mean best-so-far objective across repeated seeds")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run two short repetitions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tuning")
    args = parser.parse_args()
    metadata = run(args.quick, args.output_dir)
    summary = metadata["summary"]
    print(f"Saved repeated tuning results under {args.output_dir.resolve()}")
    print(
        f"Bayesian optimization: {summary['bayesian_mean_minutes']:.3f} ± "
        f"{summary['bayesian_std_minutes']:.3f} min; random search: "
        f"{summary['random_mean_minutes']:.3f} ± {summary['random_std_minutes']:.3f} min"
    )


if __name__ == "__main__":
    main()
