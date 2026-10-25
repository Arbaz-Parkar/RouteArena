# RouteArena: Implementation Brief for Codex

Main project for Algorithm for Optimization (course code M23CS1MJ02), MSc Computer Science Semester 1, SIES Nerul. It must be built today, so scope is prioritized (section 4). 

Fill in before sending:
- Deadline: today, preferrable before 7PM
- LLM provider: Any basic LLM that is free 
- Hardware: CPU only for Now

## 1. Role and working style

Act as a senior Python engineer and technical mentor. The student must explain every module in a viva and will show this repository to recruiters for AI/ML internships. Code must be readable, tested where it matters, and honest about results.

How to work:

- Build a thin vertical slice first (a page that plans a route with two algorithms), then widen it. Keep the repository runnable after every milestone.
- Prefer boring, well-known libraries and simple designs. No unnecessary abstraction.
- After each milestone report: files created or changed, what each module does in plain language (3 to 5 lines), tests actually run with their real output, commands the student must run, and a suggested commit message with the exact `git add` command. Do not commit unless asked.
- Give exact Windows PowerShell commands.
- Investigate root causes of errors. Do not apply random fixes.
- Tell the student clearly when a manual step is needed (API key, model download).
- Ask a question only when this brief cannot answer it.

Academic note: the course practicals mention Julia, but the student chose Python for this project because it is the industry standard. Implement the course algorithms from scratch in NumPy so they are transparent, and use SciPy or scikit-learn only to validate them in tests.

## 2. What RouteArena is

RouteArena is a local web application where optimization algorithms compete to solve a real last-mile delivery problem, and an AI assistant explains the results in plain English.

Tagline: "Where optimization algorithms compete to plan smarter deliveries."

Three audiences:

- The professor: every unit of the syllabus is implemented and visibly demonstrated.
- Non-CS viewers: a story-first interface with plain-English captions, analogies, and visual races between algorithms.
- Recruiters: engineering quality, reproducible experiments, and a small but properly evaluated GenAI and RAG component.

## 3. The real-world problem

A small delivery business (a cloud kitchen, pharmacy, or grocery store) in an Indian city must decide:

1. Where to place its hub or dark store, given customer locations and demand. This is a continuous optimization problem.
2. In what order one rider should visit N stops and return to the hub. This is a discrete routing problem (a travelling-salesman-type problem).
3. Whether the plan will meet a delivery deadline when travel times are uncertain. This is optimization under uncertainty.
4. How to tune the solvers cheaply, because evaluating a plan under uncertainty is expensive. This is surrogate-based optimization.

Why it matters (state qualitatively): fuel and rider time, late deliveries, and rider workload. Do not quote numeric industry statistics unless you can cite a genuine, verifiable source.

Data honesty rules (these must appear in the UI and README):

- The map region is real. Default center is approximately 19.03 N, 73.02 E (Nerul, Navi Mumbai). Verify on a map that generated points fall on land, not in the creek or sea.
- Stops are synthetic (randomly generated inside a bounding box, seeded) unless the user uploads a CSV. Label them as synthetic.
- Travel time is distance times a configurable detour factor divided by a configurable average speed. The detour factor and speed are assumptions, not measurements.
- Traffic uncertainty comes from a documented stochastic model. It is not measured traffic.
- RouteArena is a decision-support demonstration, not a production routing engine.

## 4. Scope and priorities

P0, must work today (milestones 0 to 6):

- From-scratch optimizers (Units 1 and 2) and the Optimizer Arena page.
- Hub placement using those optimizers.
- Route planning: nearest-neighbor baseline, exact dynamic programming for small N, genetic algorithm, ant colony optimization, and a 2-opt improvement step.
- Monte Carlo uncertainty analysis and interval (worst-case) analysis.
- Streamlit app with the core pages.
- Assistant with grounded explanations and a basic RAG knowledge base, working offline without any LLM as well.

P1, if time remains (milestone 7):

- Sampling plans, surrogate models (linear with basis functions, Gaussian process), and Bayesian optimization of solver settings (Unit 3). If this slips, the syllabus mapping must say so honestly.
- Natural-language scenario parsing with structured output.
- Retrieval evaluation set.

P2, stretch (milestone 8):

- GitHub Actions CI, Dockerfile, OR-Tools as a reference solver, OSMnx road distances, a standard TSP benchmark instance with a known optimum.

Deferred (do not build): multi-vehicle routing, per-stop time windows, live traffic APIs, polynomial chaos, genetic programming and grammars, LLM fine-tuning, agent frameworks, LangChain, user accounts, databases, cloud deployment.

## 5. Tech stack and engineering practices

Language: Python. Check the installed version first and confirm compatible wheels exist for every dependency. Use 3.11 or 3.12 in a fresh venv if needed (for example `py -3.11 -m venv .venv`).

Libraries (keep the list small, justify any addition):

- numpy, scipy, pandas, scikit-learn (validation and metrics only for from-scratch code), matplotlib, plotly.
- streamlit for the UI. Use Plotly map traces with an open street-map style that needs no token.
- pydantic v2 and pydantic-settings for data models and configuration.
- openai Python SDK as the single LLM client, configured with a base URL, API key, and model name. This one code path covers hosted OpenAI-compatible providers and local Ollama (which exposes an OpenAI-compatible endpoint).
- chromadb for the vector store. Embeddings: use Chroma's default embedding function, or sentence-transformers (all-MiniLM-L6-v2) if it installs cleanly. If embeddings cannot be loaded, fall back to TF-IDF retrieval with scikit-learn and say so in the UI.
- pytest, pytest-cov, ruff.

Do not use LangChain. Plain SDK calls and Chroma are easier to understand, debug, and explain, and avoid API churn.

Practices to apply:

- `src/` layout, a package with type hints and docstrings, `pyproject.toml` or a pinned `requirements.txt` (pin only after imports work).
- Configuration through environment variables and a validated settings class. Provide `.env.example`. Never commit `.env`, keys, the vector store, or generated data.
- Reproducibility: every stochastic function takes a seed or a `numpy.random.Generator`. Each run writes a JSON record (config, seed, results, library versions, timestamp) under `runs/`.
- Logging with the standard `logging` module. No print-debugging left in library code.
- Lint and format with ruff. Run pytest with coverage.
- Small, focused functions. Clear names. No magic numbers without a named constant or config entry.
- Conventional, descriptive commit messages (suggested, not executed).

## 6. Syllabus coverage

Terminology follows the syllabus (which appears to follow Kochenderfer and Wheeler, Algorithms for Optimization). Implement each item from scratch in NumPy unless marked otherwise.

Unit 1, Introduction to Optimization Process:
- Problem formulation: objective, constraints, decision variables (documented for each task in section 7).
- Golden section search, Fibonacci search, quadratic fit search (one-dimensional). Use golden section search as the exact line search inside gradient descent.
- Contour plots, critical points, local minima conditions on the hub-cost surface and test functions.

Unit 2, Order Methods:
- First-order: gradient descent, Adagrad, RMSProp, Adadelta, Adam. Conjugate gradient and hypergradient descent are optional.
- Second-order: Newton's method, secant method, and one quasi-Newton method (BFGS).
- Test functions: a quadratic bowl, Rosenbrock, and the smoothed hub-cost surface. Provide analytic gradients and check them numerically in tests.

Unit 3, Sampling and Surrogate Models (P1):
- Sampling plans: full factorial, random, uniform projection (permutation-based, Latin-hypercube style), stratified.
- Space-filling metrics: minimum pairwise distance and a discrepancy measure (validate against `scipy.stats.qmc`).
- Surrogates: linear model with basis functions, fit to noisy objective samples, model selection on a validation split, and a Gaussian process (RBF kernel, from scratch, validated against scikit-learn).
- Bayesian optimization with Expected Improvement using the Gaussian process.

Unit 4, Optimization and Uncertainty:
- Uncertainty propagation by Monte Carlo sampling; set-based (interval) uncertainty with worst-case analysis; probabilistic uncertainty with percentiles and probability of lateness.
- Dynamic programming (Held-Karp exact routing for small N).
- Ant colony optimization and a genetic algorithm for routing.
- Taylor approximation (first-order propagation of demand uncertainty on hub cost) is optional.
- Not required: polynomial chaos, expression optimization, genetic programming, grammars, probabilistic prototype trees.

The app must include an "Under the Hood" page mapping each feature to its syllabus topic, listing any item not implemented.

## 7. Module specifications

### 7.1 Optimizers (`optimizers/`)
A common interface: `minimize(f, grad, x0, **options)` returns a result object with the final point, value, the full path, iteration count, function and gradient evaluation counts, and a convergence flag. Stopping rules: gradient norm tolerance, maximum iterations. Handle divergence (NaN or infinity) without crashing. Tests: known minima within tolerance, agreement with `scipy.optimize` on smooth problems, gradient checks.

### 7.2 Hub placement (`logistics/hub.py`)
Given customer coordinates and demand weights, minimize the weighted sum of distances to a hub position. Use a smoothed distance sqrt(d squared plus epsilon squared) so the objective is differentiable. Work in a local planar projection (metres or kilometres from a reference point) so gradients are well scaled, and convert back to latitude and longitude for display. Document the projection. Compare the optimizers on this surface and draw their paths over a contour plot.

### 7.3 Distance and time model (`logistics/geo.py`)
Haversine distance between points, a detour (circuity) factor, and an average speed. All three are configurable and documented as assumptions. Build a symmetric travel-time matrix. Validate coordinates (latitude and longitude bounds, a maximum stop count of 30, no duplicate or NaN points).

### 7.4 Routing (`logistics/routing/`)
All solvers take a travel-time matrix and return a closed tour starting and ending at the hub, its cost, runtime, and algorithm metadata.
- Nearest neighbor baseline and a 2-opt improvement step.
- Exact Held-Karp dynamic programming, limited to a documented small N (about 12 or fewer). Refuse larger inputs with a clear message.
- Genetic algorithm: permutation encoding, order crossover, swap or inversion mutation, tournament selection, elitism, configurable parameters.
- Ant colony optimization: pheromone matrix, alpha, beta, evaporation rate, number of ants and iterations, with a convergence history.
Tests: valid permutations, tours start and end at the hub, optimality of DP on tiny instances (brute-force check for N up to 7), heuristics never worse than a random tour on average.

### 7.5 Uncertainty (`logistics/uncertainty.py`)
Monte Carlo travel-time simulator. A suggested model: each sampled day has a shared city-wide traffic factor (lognormal, with mean depending on a traffic level) multiplied by independent per-leg noise (lognormal). The shared factor matters because it correlates delays across legs. Document the model and every parameter.
Outputs for a tour: mean total time, standard deviation, 50th, 90th and 95th percentiles, probability of exceeding a deadline, and a histogram. Also provide interval (set-based) analysis: lower and upper travel-time bounds per leg and the worst-case total time. Compare the deterministic shortest tour with a robust tour (best 90th percentile among candidates from the solvers). Check Monte Carlo convergence as the number of samples grows. Cache nothing that depends on the seed without including the seed in the key.

### 7.6 Sampling, surrogate and Bayesian tuning (`surrogate/`, P1)
The expensive objective is "mean tour cost over several seeds" of ACO or the GA as a function of its hyperparameters (for ACO: alpha, beta, evaporation rate). Steps: generate a design with each sampling plan, evaluate the objective, fit the surrogates, report held-out error, then run Bayesian optimization with a small evaluation budget and compare it with random search at the same budget, averaged over several repetitions. Report whatever happens, including if Bayesian optimization does not win.

### 7.7 ML demonstration (`ml/`)
A "How AI learns" demo: fit a small linear model (with optional polynomial features) predicting delivery time from distance, number of stops, and traffic level using the from-scratch optimizers (gradient descent versus Adam), compare against the closed-form solution and scikit-learn, and show loss curves. State clearly that this uses synthetic data generated from the documented travel-time model and shows the learning mechanism only.

## 8. GenAI component

The LLM never computes optimization results. Solvers produce all numbers. The LLM only (a) turns a user's sentence into a validated scenario and (b) explains computed results in plain English. The app must work fully without any LLM configured ("offline mode" uses template-based explanations and TF-IDF or plain keyword retrieval).

### 8.1 LLM client (`genai/client.py`)
- A thin wrapper over the OpenAI SDK using `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` from settings. Low temperature, token limits, timeouts, a few retries with backoff, and clear error messages. On failure, fall back to offline mode and tell the user.
- For local use, document installing Ollama on Windows and pulling a small model that fits a laptop. Verify the model exists with `ollama list` before assuming a name.
- Never log keys. Never send uploaded stop data to an external provider without a visible notice in the UI that hosted providers receive the text sent to them.

### 8.2 Scenario parsing
Natural language such as "10 deliveries around Nerul, rider must finish in 90 minutes, heavy traffic" is converted to a pydantic `Scenario` (number of stops, region preset, deadline minutes, traffic level enum, average speed override, optional objective). Use JSON output or function calling, then validate with pydantic with strict bounds. If validation fails, show what was missing and ask the user to rephrase. Never execute model output as code, and never let it change anything outside the schema.

### 8.3 Grounded explanation
Pass the structured results (costs, runtimes, optimality gap, percentiles, probability of lateness, assumptions) as context. The prompt must instruct the model to use only supplied numbers and to say when something is not provided. Add a cheap grounding check: extract numbers from the explanation and verify they appear in the context (within rounding). If the check fails, log it, show a warning, and fall back to the template explanation.

### 8.4 RAG assistant (`genai/rag/`)
- Knowledge base: markdown files in `knowledge/`, written by you in plain English for non-experts, one per syllabus topic (what it is, an everyday analogy, when it works, when it fails, and where it appears in RouteArena), plus a glossary, an FAQ about the project and its assumptions, and automatically generated text summaries of saved run records. Examples of analogies: gradient descent as walking downhill in fog, ant colony as ants leaving scent trails, genetic algorithm as breeding better routes, Monte Carlo as replaying the same day thousands of times with different traffic.
- Indexing: chunk by markdown heading with a size cap and small overlap. Store source file, section, and syllabus unit as metadata. Persist Chroma in a gitignored directory. Provide `scripts/build_index.py` to rebuild.
- Retrieval: top-k (default 4) with a similarity threshold. If nothing passes the threshold, answer "I could not find that in my knowledge base" rather than guessing.
- Generation: answer only from retrieved passages, cite sources as `[file: section]`, and keep answers short and in plain language.
- Conversation: keep only the last few turns. No long-term memory.
- Safety: user input is untrusted. Knowledge files are local and trusted. Render model output as plain markdown, never as raw HTML. The assistant cannot call tools, write files, or run solvers except through the validated `Scenario` path.
- Evaluation: write a small hand-made set (10 to 15 questions) with the expected source document for each, plus 3 out-of-scope questions that should be refused. Report retrieval hit rate at k and the refusal rate. Real numbers only.

## 9. Application (Streamlit)

Style: clean, high contrast, friendly and plain. No emojis, no heavy animation. Every page begins with two plain-English sentences on what the viewer is about to see. Any technical term gets a short help tooltip. Use Plotly charts with clear labels and units.

Pages:

1. Overview: the story ("one rider, N stops, uncertain traffic"), the four decisions from section 3, a short results summary pulled from real saved experiment files, and the data-honesty notice.
2. Optimizer Arena: choose a landscape (bowl, valley, hub-cost surface) and optimizers, then watch paths race over a contour plot with a step-by-step slider. Show iterations to converge for each. Include the "How AI learns" demo.
3. Delivery Planner: generate synthetic stops (seed, count) or upload a CSV (columns: name, lat, lon, demand; size limit and validation), place the hub (optimized or manual), run the chosen solvers, and display the routes on the map with total time, runtime, and optimality gap when the exact solution is available.
4. Risk and Uncertainty: pick a plan, set deadline and traffic level, run Monte Carlo, show the histogram, percentiles, probability of lateness, the interval worst case, and the shortest-versus-robust comparison.
5. Smart Tuning (P1): sampling plans, surrogate fit, and Bayesian optimization progress compared with random search.
6. Assistant: a chat box with scenario parsing and RAG, showing sources and a clear "offline mode" or "model: name" indicator. Include buttons such as "Explain this result" that use the latest run.
7. Under the Hood: the syllabus mapping, assumptions, limitations, and the reproducibility instructions.

Engineering rules for the UI: cache loaded resources with `st.cache_resource` and pure computations with `st.cache_data` including the seed in the cache key; do not rerun solvers when an unrelated control changes; validate all uploads; never overwrite uploaded files; keep long computations behind an explicit run button with a spinner.

## 10. Target repository structure

Adapt to what already exists. Create files only when they hold real functionality.

```
RouteArena/
├── app.py
├── pyproject.toml / requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── knowledge/                 plain-English markdown for the RAG index
├── scripts/
│   ├── run_experiments.py
│   ├── build_index.py
│   └── eval_rag.py
├── src/routearena/
│   ├── config.py              settings, seeds
│   ├── optimizers/            line_search.py, first_order.py, second_order.py
│   ├── logistics/             geo.py, hub.py, uncertainty.py, routing/
│   ├── surrogate/             sampling.py, models.py, bayesopt.py
│   ├── ml/                    linear_demo.py
│   ├── genai/                 client.py, scenario.py, explain.py, rag/
│   ├── records.py             run records (JSON)
│   └── ui/                    one module per page
├── tests/
├── docs/
│   ├── syllabus_mapping.md
│   ├── methodology.md
│   ├── limitations.md
│   └── code_walkthrough.md
├── outputs/                   generated metrics and plots
└── runs/                      generated run records (gitignored)
```

## 11. Evaluation and experiments

`scripts/run_experiments.py` runs everything below with fixed seeds and saves JSON, CSV and PNG files under `outputs/` plus an `experiment_metadata.json` (seeds, config, versions, machine info). Provide a quick smoke mode clearly labelled as not a result.

1. Optimizer comparison: iterations, function evaluations, and final error for each optimizer on each test function, over several starting points. Report mean and spread.
2. Hub placement: how each optimizer reaches the same optimum, and agreement with a SciPy reference.
3. Routing quality: for many random instances at small N, the optimality gap of nearest neighbor, nearest neighbor with 2-opt, GA, and ACO versus the exact DP solution. For larger N, compare against the best known among the solvers. Report mean, spread, and runtime over several seeds, never a single run.
4. Uncertainty: Monte Carlo convergence of the mean and 90th percentile, and whether the robust tour lowers probability of lateness relative to the shortest tour, with the cost in extra expected time.
5. Tuning (P1): surrogate held-out error for each sampling plan, and Bayesian optimization versus random search at the same budget over repeated runs.
6. RAG: retrieval hit rate at k and refusal rate on the evaluation set.
7. Grounding: share of explanations that pass the number check when an LLM is available.

Present results in the README as short labelled lists and plots, not comparison tables. If an experiment cannot be run, say so and why.

## 12. Testing and quality

Tests must be fast and must not need network access, an LLM, or downloaded models (mock the LLM client and use a tiny fake embedding function).

- Optimizers: known minima, gradient checks, divergence handling, agreement with SciPy.
- Geo and hub: distance sanity values, symmetric matrices, validation errors, projection round trip.
- Routing: tour validity, DP versus brute force on tiny cases, solver determinism for a fixed seed.
- Uncertainty: percentiles are ordered, probability of lateness is between 0 and 1, interval worst case is at least the mean.
- Surrogate (P1): GP matches scikit-learn on a small problem, sampling plans have the right shape and projection properties.
- GenAI: scenario schema accepts valid input and rejects bad bounds, the grounding check catches an invented number, retrieval returns the expected source for a toy corpus, offline mode works with no settings.

Run `python -m pytest --cov`, `ruff check`, `ruff format --check`, an import check, and a Streamlit startup check. Report real results only. Add a GitHub Actions workflow (pytest and ruff) in P2.

## 13. Documentation and portfolio deliverables

README (accurate commands only): one-paragraph problem statement, architecture diagram (a Mermaid diagram renders on GitHub), what each page shows, setup on Windows, LLM setup (optional), how to run experiments and tests, measured results from real runs, limitations, future work, and references. Include a screenshot of each page captured from a real run.

Docs: `syllabus_mapping.md` (each syllabus topic to code location and page), `methodology.md` (algorithms and equations that match the code), `limitations.md`, and `code_walkthrough.md` (plain-language module guide in the order a professor would ask about it).

At the end, in the chat only (not in the repository), give the student: three resume bullets using only measured numbers, ten likely viva questions with short answers, and a two-minute demo script for a non-technical viewer.

## 14. Milestones

0. Audit the workspace, Python version, Git state, and dependency compatibility. Report and wait for confirmation before deleting anything. Then scaffold the package, settings, logging, ruff, and pytest.
1. Unit 1 and Unit 2 optimizers with tests, the test functions, and a minimal Optimizer Arena page.
2. Geo model, hub placement, and the hub contour visualization.
3. Routing solvers (nearest neighbor, 2-opt, DP, GA, ACO) with tests and the Delivery Planner page.
4. Uncertainty module and the Risk and Uncertainty page.
5. Run records, knowledge base, offline assistant, RAG indexing, and the Assistant page (offline mode first).
6. LLM client, grounded explanations, and RAG answers with citations. Experiments script for items 1 to 4 and 6. Syllabus mapping page. This completes P0.
7. P1: sampling plans, surrogates, Bayesian tuning, scenario parsing, and the Smart Tuning page.
8. P2 items as time allows, final audit, README polish.

If a milestone cannot finish, leave the repository working and give a precise continuation checklist. The project must always be demonstrable, so cut P1 and P2 before compromising P0.

## 15. Honesty rules

- Never fabricate results, screenshots, citations, statistics, or test output.
- Never claim a feature works without running it.
- Never present smoke-test numbers as results.
- Label all synthetic data and every modelling assumption.
- If an algorithm loses to a simpler baseline, report that.
- Keep README and docs consistent with the code that exists.
- Describe the LLM features accurately: the model explains and parses, it does not optimize.

## 16. Definition of done

- The app launches locally and every P0 page works.
- All Unit 1 and Unit 2 methods run from scratch with tests, and the Arena shows them racing.
- Hub placement, four routing methods with exact DP for small N, and Monte Carlo plus interval analysis work end to end.
- The assistant works offline, and with an LLM when configured, with cited retrieval and a grounding check.
- `run_experiments.py` produces real metrics for the P0 experiments.
- Tests and lint pass apart from documented optional integration tests.
- README, docs, and the syllabus mapping describe what actually exists, including anything left out.

## 17. First action

Run Milestone 0 and report the Python and dependency situation and your keep, simplify, or remove list if any files already exist. Then start Milestone 1.