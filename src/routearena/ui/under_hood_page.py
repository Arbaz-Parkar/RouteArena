"""A plain-language map from RouteArena features to course topics."""

import streamlit as st


_COVERAGE = [
    {
        "Unit": "Unit 1 · Optimization process",
        "RouteArena feature": "Golden section, Fibonacci, quadratic fit, contour views, and problem setup",
        "Status": "Implemented",
        "Where to look": "Optimizer Arena · optimizers/line_search.py · logistics/hub.py",
    },
    {
        "Unit": "Unit 2 · Order methods",
        "RouteArena feature": "Gradient descent, Adagrad, RMSProp, Adadelta, Adam, Newton, secant, and BFGS",
        "Status": "Implemented; conjugate gradient and hypergradient are optional gaps",
        "Where to look": "Optimizer Arena · optimizers/first_order.py · optimizers/second_order.py",
    },
    {
        "Unit": "Unit 3 · Sampling and surrogate models",
        "RouteArena feature": "Random, factorial, stratified, and Latin-hypercube designs; discrepancy metrics; polynomial and RBF GP surrogates; expected-improvement tuning",
        "Status": "Implemented in Smart Tuning; experiment reports compare Bayesian and random search",
        "Where to look": "Smart Tuning · surrogate/sampling.py · surrogate/models.py · surrogate/bayes.py",
    },
    {
        "Unit": "Unit 4 · Uncertainty and combinatorial optimization",
        "RouteArena feature": "Monte Carlo and interval risk, Held-Karp, nearest neighbor, 2-opt, genetic algorithm, and ant colony",
        "Status": "Implemented",
        "Where to look": "Delivery Planner · Risk and Uncertainty · logistics/routing/ · logistics/uncertainty.py",
    },
    {
        "Unit": "GenAI demonstration · project extension",
        "RouteArena feature": "Optional grounded explanations, cited local RAG, validated scenario parsing, and run summaries",
        "Status": "Implemented when a model is configured; all answers still work offline",
        "Where to look": "Assistant · genai/client.py · genai/grounding.py · genai/scenario.py · genai/rag/",
    },
    {
        "Unit": "ML demonstration · project extension",
        "RouteArena feature": "Synthetic delivery-time prediction with gradient descent and Adam",
        "Status": "Implemented with closed-form and scikit-learn references",
        "Where to look": "Optimizer Arena · ml/regression_demo.py · ui/ml_demo.py",
    },
]


def render_under_hood_page() -> None:
    """Show coverage, assumptions, limits, and reproducibility commands."""
    st.title("Under the Hood")
    st.write(
        "This page connects the app to the optimization syllabus. It also lists the current gaps "
        "so the project’s coverage is easy to explain and verify."
    )
    st.dataframe(_COVERAGE, hide_index=True, use_container_width=True)

    st.subheader("Model assumptions and limits")
    st.markdown(
        """
        - Customer stops are synthetic unless supplied through a CSV upload.
        - Route time is estimated from straight-line distance, a detour factor, and an assumed average speed.
        - Traffic is sampled from the documented lognormal teaching model; it is not measured traffic.
        - The hub uses a local planar projection around Nerul and is intended for a small area.
        - RouteArena plans one rider’s closed tour. It is a teaching and decision-support demo, not a live navigation service.
        - An optional language model rewrites supplied results. The solvers produce all numeric results.
        """
    )
    st.subheader("Reproducibility")
    st.code(
        "python -m pytest --cov=routearena\n"
        "python scripts/build_index.py\n"
        "python scripts/eval_rag.py\n"
        "python scripts/run_experiments.py\n"
        "python scripts/run_tuning_experiment.py",
        language="powershell",
    )
    st.caption(
        "Stochastic solver runs accept explicit seeds and save local JSON summaries under runs/. "
        "Experiment tables, plots, and metadata are written under outputs/."
    )
