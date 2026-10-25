# RouteArena

[![CI](https://github.com/Arbaz-Parkar/RouteArena/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Arbaz-Parkar/RouteArena/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**An interactive operations research workbench for last-mile delivery planning.**

RouteArena demonstrates how optimization, simulation, and lightweight machine learning can support delivery planning. It runs locally as a Streamlit application and includes implementations of its core optimization methods, reproducible experiment scripts, and optional reference integrations.

## Capabilities

| Area | Included functionality |
| --- | --- |
| Optimization | First- and second-order methods, line searches, and visual objective surfaces |
| Facility location | Demand-weighted hub placement with gradient-based solvers |
| Route planning | Nearest neighbor, 2-opt, Held–Karp, genetic algorithm, and ant colony optimization |
| Risk analysis | Seeded Monte Carlo simulation and interval-based worst-case estimates |
| Smart tuning | Sampling designs, surrogate models, and budget-matched Bayesian and random search |
| Learning demonstration | Synthetic regression trained with gradient descent and Adam |
| Research assistant | Offline retrieval with citations; optional, explicitly enabled model responses |
| Reference tooling | TSPLIB benchmark, optional OR-Tools solver, OSMnx road-distance matrix, and Docker image |

## Contents

- [Quick start](#quick-start)
- [Development](#development)
- [Optional model configuration](#optional-model-configuration)
- [Experiments and benchmarks](#experiments-and-benchmarks)
- [Application areas](#application-areas)
- [Optional integrations](#optional-integrations)
- [Docker](#docker)
- [Continuous integration](#continuous-integration)
- [Data, assumptions, and scope](#data-assumptions-and-scope)
- [Development](#development)
- [Data, assumptions, and scope](#data-assumptions-and-scope)

## Quick start

Python 3.11 or newer is required.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
```

### macOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
```

## Development

Run the automated checks from the activated virtual environment:

```powershell
python -m pytest --cov=routearena
ruff check .
ruff format --check .
```

## Optional model configuration

The application works offline by default. Model answers are opt-in and require explicit confirmation in the Assistant before a question or compact result summary is sent to the configured endpoint. Avoid sending customer names, uploaded stop data, or coordinates to an external provider.

### Local model with Ollama on Windows

Install Ollama from its [Windows download page](https://ollama.com/download/windows), then open PowerShell and pull a small model. Ollama's model registry currently lists `qwen2.5:1.5b` at about 986 MB; actual memory use and CPU speed depend on the computer.

```powershell
ollama pull qwen2.5:1.5b
ollama list
Copy-Item .env.example .env
notepad .env
```

Set these entries in `.env`, using the exact model name shown by `ollama list`:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:1.5b
```

Keep Ollama running while RouteArena is open. Ollama provides an OpenAI-compatible local endpoint; `ollama` is a placeholder key for this local connection. A hosted provider can use the same SDK path with its own base URL, model, and real API key. Never commit `.env`.

## Experiments and benchmarks

Rebuild the local Markdown chunk index with:

```powershell
python scripts/build_index.py
python scripts/eval_rag.py
```

Run the reproducible CPU experiment sweep with:

```powershell
python scripts/run_experiments.py
```

This compares optimizers on test surfaces and hub placement, checks the hub results against SciPy, compares routing heuristics with Held-Karp, examines Monte Carlo convergence and robust candidates, and evaluates local RAG retrieval. It saves raw JSON and CSV rows, mean and standard-deviation aggregates in `experiment_summary.csv`, `experiment_metadata.json` (timestamp, seed, package versions, and machine information), and `summary.png` under the ignored `outputs/` folder. A matching run record is also written under `runs/`. Use `--quick` for smaller real samples. Report the metrics produced by the script; do not substitute unrun or illustrative results.

## Application areas

### Optimizer Arena and hub placement

The Optimizer Arena visualizes first- and second-order optimization methods on test functions. The hub-placement demonstration minimizes demand-weighted distance from candidate facilities to customer locations.

### Delivery planning and uncertainty

The Delivery Planner compares route-construction and improvement methods with exact Held–Karp solutions on supported instance sizes. Risk and Uncertainty reports seeded Monte Carlo estimates and configurable interval-based worst cases. The traffic model includes a shared day-wide lognormal factor and independent per-leg factors; its parameters are exposed in the interface.

### Smart Tuning

The **Smart Tuning** page compares random, Latin-hypercube, stratified-grid, and factorial designs for ACO settings. It fits a polynomial-basis linear model and an RBF Gaussian process, reports held-out error, then compares expected-improvement Bayesian optimization with random search at the same evaluation budget. The objective is the mean route time over fixed ACO seeds on one synthetic instance; the page records the seed and settings.

Run repeated fixed-budget comparisons with:

```powershell
python scripts/run_tuning_experiment.py
```

Use `--quick` for two shorter repetitions. The script writes per-run values, a mean/std CSV summary, package and machine metadata, and a convergence PNG under `outputs/tuning/`. Results are empirical; Bayesian optimization may lose to random search on a given run.

### Learning demonstration

The Optimizer Arena includes a **How AI Learns** section. It trains a small linear delivery-time model on synthetic examples and compares gradient descent and Adam with least squares and scikit-learn. These predictions demonstrate the training process and are not traffic estimates.

### Assistant scenarios

In the Assistant, an opted-in model can turn a sentence into a strict Pydantic scenario object. Required fields and allowed ranges are validated before display; parsing does not automatically run a solver. Scenario parsing requires a configured model, while RAG and template explanations remain available offline.

## Optional integrations

### TSPLIB and OR-Tools benchmark

The P2 references are optional and do not replace RouteArena's from-scratch solvers.
Install the OR-Tools reference and run the 14-city TSPLIB `burma14` benchmark with:

```powershell
python -m pip install -e ".[benchmarks]"
python scripts/run_tsplib_benchmark.py
```

The benchmark checks Held-Karp against the published optimum of 3,323 km, then reports
seeded genetic and ant-colony results and an OR-Tools heuristic reference when installed.
OR-Tools does not guarantee an optimal route. The coordinates come from the
[TSPLIB burma14 instance](https://raw.githubusercontent.com/mastqe/tsplib/master/burma14.tsp);
the known optimum is listed by [TSPLIB's symmetric-instance reference](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html).

### OpenStreetMap road distances

Install OSMnx and NetworkX to calculate an optional OpenStreetMap road-distance matrix:

```powershell
python -m pip install -e ".[maps]"
```

Call `routearena.logistics.road_reference.osmnx_road_distance_matrix` with an array of
`(latitude, longitude)` points. If no graph is passed, it downloads the drivable graph
around Nerul; this needs an internet connection. The function returns directed road
distances in metres. It does not estimate traffic or travel time, and the current
from-scratch solvers require symmetric matrices, so this is a separate reference.
OSMnx speed and routing behavior is described in its
[installation guide](https://osmnx.readthedocs.io/en/stable/installation.html) and
[routing reference](https://osmnx.readthedocs.io/en/stable/user-reference.html).

## Docker

Build and run the local web application with Docker Desktop:

```powershell
docker build -t routearena .
docker run --rm -p 8501:8501 routearena
```

Then open `http://localhost:8501`. The image runs as a non-root user and starts in
offline mode. Optional LLM settings can be supplied through Docker environment variables;
do not bake `.env` or API keys into the image.

## Continuous integration

GitHub Actions runs the test suite with coverage, Ruff lint, and Ruff formatting checks
on Python 3.11 and 3.12 for pushes to `main`, pull requests, and manual runs.

## Data, assumptions, and scope

- The default demonstration area is Nerul, Navi Mumbai. Synthetic stops are seeded and placed near mapped roadside anchors. An uploaded CSV must include `name`, `lat`, `lon`, and `demand`; it is validated in memory and is not modified.
- The built-in route cost uses Haversine distance, a 1.25 detour factor, and an assumed average speed of 25 km/h. These are teaching assumptions, not measured road travel times. Map lines connect stops directly and do not provide turn-by-turn directions.
- Risk estimates use the traffic model and parameters shown in the application. They are simulated assumptions, not live or historical traffic data.
- Run summaries are written under the git-ignored `runs/` directory. They omit customer names and uploaded coordinates.
- RouteArena is a local decision-support and teaching application. It is not a live navigation service or a production fleet-routing system.
