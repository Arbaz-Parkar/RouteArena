"""Interactive Unit 3 demonstration for sampling, surrogates, and Bayesian tuning."""

import plotly.graph_objects as go
import streamlit as st

from routearena.records import write_run_record
from routearena.surrogate.tuning import (
    ACO_PARAMETER_NAMES,
    run_tuning_pipeline,
    synthetic_time_matrix,
)


@st.cache_data(show_spinner=False)
def _cached_tuning(seed: int, plan: str, design_size: int, search_budget: int) -> dict:
    """Cache the complete tuning run by all computation inputs."""
    matrix = synthetic_time_matrix(seed=seed, customer_count=8)
    return run_tuning_pipeline(
        matrix,
        plan=plan,
        design_size=design_size,
        search_budget=search_budget,
        seed=seed,
    )


def _summary_record(result: dict) -> dict:
    """Keep the persisted record compact while retaining the core measurements."""
    bayesian = result["bayesian"]
    random = result["random"]
    return {
        "design_point_count": int(len(result["design_values"])),
        "design_metrics": result["design_metrics"],
        "selected_polynomial_degree": int(result["selected_degree"]),
        "linear_validation_rmse": float(result["linear_validation_rmse"]),
        "linear_scores_by_degree": result["linear_scores_by_degree"],
        "gp_validation_rmse": float(result["gp_validation_rmse"]),
        "search_budget": int(result["budget"]),
        "bayesian_best": {"value": bayesian.best_value, "parameters": bayesian.best_point},
        "random_best": {"value": random.best_value, "parameters": random.best_point},
        "bayesian_best_history": bayesian.best_history,
        "random_best_history": random.best_history,
    }


def render_tuning_page() -> None:
    """Render one seeded ACO tuning study with a fair random-search budget."""
    st.title("Smart Tuning")
    st.write(
        "This page explores how sampling and surrogate models can reduce expensive solver trials. "
        "Bayesian optimization and random search receive the same evaluation budget."
    )
    st.caption(
        "The objective is mean ACO route time over fixed solver seeds on a synthetic eight-stop "
        "instance. Travel times use a 1.25 detour factor and assumed 25 km/h speed."
    )

    left, right = st.columns(2)
    with left:
        plan = st.selectbox(
            "Initial sampling plan",
            ["Latin hypercube", "Random", "Stratified grid", "Full factorial"],
            help="Each design point is a set of ACO alpha, beta, and evaporation settings.",
        )
        design_size = st.slider("Requested design points", min_value=8, max_value=32, value=18)
    with right:
        seed = st.number_input("Experiment seed", min_value=0, max_value=100_000, value=23, step=1)
        budget = st.slider("Solver evaluations per search", min_value=8, max_value=20, value=12)

    run = st.button("Run sampling, surrogate, and tuning study", type="primary")
    if run:
        with st.spinner("Evaluating seeded ACO settings and fitting surrogates…"):
            result = _cached_tuning(int(seed), plan, int(design_size), int(budget))
        st.session_state["tuning_result"] = result
        st.session_state["tuning_key"] = (int(seed), plan, int(design_size), int(budget))
        try:
            record = write_run_record(
                "smart-tuning",
                config={
                    "seed": int(seed),
                    "sampling_plan": plan,
                    "requested_design_size": int(design_size),
                    "search_budget": int(budget),
                    "objective": "mean ACO tour time over fixed seeds",
                },
                seed=int(seed),
                results=_summary_record(result),
            )
            st.session_state["tuning_record"] = record.name
        except OSError as exc:
            st.warning(f"Tuning finished, but its local run record could not be saved: {exc}")

    result = st.session_state.get("tuning_result")
    current_key = (int(seed), plan, int(design_size), int(budget))
    if result is None or st.session_state.get("tuning_key") != current_key:
        st.info("Choose settings and run the study to see measured results.")
        return

    metrics = result["design_metrics"]
    first, second, third = st.columns(3)
    first.metric("Evaluated design points", str(len(result["design_values"])))
    second.metric("Minimum pair distance", f"{metrics['minimum_pairwise_distance']:.3f}")
    third.metric("Centered discrepancy", f"{metrics['centered_discrepancy']:.3f}")
    st.caption(
        "A larger minimum pair distance and a smaller centered discrepancy generally indicate "
        "a more space-filling design."
    )

    design = result["design_points"]
    design_figure = go.Figure(
        go.Scatter3d(
            x=design[:, 0],
            y=design[:, 1],
            z=design[:, 2],
            mode="markers",
            marker={"size": 5, "color": result["design_values"], "colorscale": "Viridis", "showscale": True},
            text=[f"Mean route time: {value:.2f} minutes" for value in result["design_values"]],
            hovertemplate="alpha=%{x:.2f}<br>beta=%{y:.2f}<br>evaporation=%{z:.2f}<br>%{text}<extra></extra>",
        )
    )
    design_figure.update_layout(
        title=f"{result['plan']} design of ACO settings",
        scene={
            "xaxis_title": ACO_PARAMETER_NAMES[0],
            "yaxis_title": ACO_PARAMETER_NAMES[1],
            "zaxis_title": ACO_PARAMETER_NAMES[2],
        },
        height=500,
    )
    st.plotly_chart(design_figure, use_container_width=True)

    linear = float(result["linear_validation_rmse"])
    gp_error = float(result["gp_validation_rmse"])
    left, right = st.columns(2)
    left.metric(
        f"Selected polynomial degree {result['selected_degree']} validation RMSE",
        f"{linear:.3f} minutes",
    )
    right.metric("RBF Gaussian process validation RMSE", f"{gp_error:.3f} minutes")
    st.caption(f"Linear-model scores by degree: {result['linear_scores_by_degree']}")

    bayesian = result["bayesian"]
    random = result["random"]
    comparison = go.Figure()
    comparison.add_trace(
        go.Scatter(
            x=list(range(1, len(bayesian.best_history) + 1)),
            y=bayesian.best_history,
            mode="lines+markers",
            name="Bayesian optimization",
        )
    )
    comparison.add_trace(
        go.Scatter(
            x=list(range(1, len(random.best_history) + 1)),
            y=random.best_history,
            mode="lines+markers",
            name="Random search",
        )
    )
    comparison.update_layout(
        title=f"Best-so-far mean route time · same budget of {result['budget']} evaluations",
        xaxis_title="Objective evaluation",
        yaxis_title="Mean route time (minutes)",
        height=400,
    )
    st.plotly_chart(comparison, use_container_width=True)
    st.dataframe(
        [
            {
                "Search": name,
                "Best mean route time (minutes)": search.best_value,
                "alpha": float(search.best_point[0]),
                "beta": float(search.best_point[1]),
                "evaporation rate": float(search.best_point[2]),
            }
            for name, search in (("Bayesian optimization", bayesian), ("Random search", random))
        ],
        hide_index=True,
        use_container_width=True,
    )
    record_name = st.session_state.get("tuning_record")
    if record_name:
        st.caption(f"This run was saved locally as `{record_name}`.")
