# RouteArena

RouteArena is a local teaching app for last-mile delivery planning. It makes optimization methods visible: first the methods race across mathematical landscapes, then they compete on hub placement and a one-rider delivery route.

The current app includes an Optimizer Arena, a hub-placement demonstration, and a Delivery Planner with nearest-neighbor, 2-opt, Held-Karp, genetic, and ant-colony solvers. The route stops are synthetic unless provided in a CSV. Route costs use Haversine distance with an assumed 1.25 detour factor and 25 km/h average speed; these are not measured road travel times. Map lines connect points directly and are not turn-by-turn directions. RouteArena is a decision-support demonstration, not a production routing engine.

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

The app works without an LLM. Optional provider settings are documented in `.env.example`; no API key is needed for the optimizer arena.

