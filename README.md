# RouteArena

RouteArena is a local teaching app for last-mile delivery planning. It makes optimization methods visible: first the methods race across mathematical landscapes, then they compete on hub placement and a one-rider delivery route.

The current app includes an Optimizer Arena, a hub-placement demonstration, a Delivery Planner with nearest-neighbor, 2-opt, Held-Karp, genetic, and ant-colony solvers, a Risk and Uncertainty page, an Assistant, and an Under the Hood syllabus map. The risk page uses a seeded Monte Carlo model with a shared lognormal day-wide traffic factor and independent per-leg factors, plus a configurable interval worst case. Model parameters are stated in the UI and are assumptions, not traffic measurements. Optimizer, hub, route, and risk runs save JSON summaries under the ignored `runs/` folder without customer names or uploaded coordinates. The Assistant searches local Markdown notes and those summaries, cites retrieved sources, and works without an API key. An optional OpenAI-compatible model can rewrite retrieved answers and compact result summaries; a numeric grounding check falls back to the offline answer when it sees an unsupported number. The route stops are synthetic unless provided in a CSV. Route costs use Haversine distance with an assumed 1.25 detour factor and 25 km/h average speed; these are not measured road travel times. Map lines connect points directly and are not turn-by-turn directions. RouteArena is a decision-support demonstration, not a production routing engine.

The default map region is Nerul, Navi Mumbai. Synthetic customer locations are seeded and jittered near mapped Nerul roadside anchors to keep the demonstration on the urban land area. CSV uploads must contain `name`, `lat`, `lon`, and `demand`; they are validated in memory and are not modified. Map tiles use OpenStreetMap's token-free style and need an internet connection in the browser.

## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
streamlit run app.py
```

Run the tests with:

```powershell
python -m pytest --cov=routearena
ruff check .
ruff format --check .
```

The app works offline by default. Model answers are opt-in in the Assistant and require a visible checkbox before the question or compact result summary is sent to the configured endpoint. Do not send uploaded customer names or coordinates to an external provider.

### Optional local model on Windows

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

## Unit 3 and learning demonstrations

The **Smart Tuning** page compares random, Latin-hypercube, stratified-grid, and factorial designs for ACO settings. It fits a polynomial-basis linear model and an RBF Gaussian process, reports held-out error, then compares expected-improvement Bayesian optimization with random search at the same evaluation budget. The objective is the mean route time over fixed ACO seeds on one synthetic instance; the page records the seed and settings.

Run repeated fixed-budget comparisons with:

```powershell
python scripts/run_tuning_experiment.py
```

Use `--quick` for two shorter repetitions. The script writes per-run values, a mean/std CSV summary, package and machine metadata, and a convergence PNG under `outputs/tuning/`. Results are empirical; Bayesian optimization may lose to random search on a given run.

The Optimizer Arena also contains a **How AI Learns** section. It trains a small linear delivery-time model on synthetic examples and compares gradient descent and Adam with least squares and scikit-learn. These predictions demonstrate the training process and are not traffic estimates.

In the Assistant, an opted-in model can turn a sentence into a strict Pydantic scenario object. Required fields and allowed ranges are validated before display; parsing does not automatically run a solver. Scenario parsing requires a configured model, while RAG and template explanations remain available offline.

