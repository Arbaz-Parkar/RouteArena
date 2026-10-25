# RouteArena

RouteArena is a local teaching app for last-mile delivery planning. It makes optimization methods visible: first they race across simple mathematical landscapes, then they will be applied to hub placement, routing, and uncertain travel times.

The current milestone contains the optimizer foundations and the first interactive arena. Later milestones will add delivery geography and routing. Results shown by the current page are mathematical demonstrations, not real delivery estimates.

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

