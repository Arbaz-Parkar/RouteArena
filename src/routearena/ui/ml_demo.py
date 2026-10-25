"""The small "How AI Learns" regression demonstration in the Optimizer Arena."""

import plotly.graph_objects as go
import streamlit as st

from routearena.ml.regression_demo import run_regression_demo
from routearena.records import write_run_record


@st.cache_data(show_spinner=False)
def _cached_regression_demo(seed: int) -> dict:
    """Cache the deterministic synthetic training run by its seed."""
    return run_regression_demo(seed)


def render_ml_demo() -> None:
    """Show gradient-based learning, a closed-form reference, and test error."""
    st.subheader("How AI learns")
    st.write(
        "A small model learns delivery time from synthetic distance, stop-count, "
        "and traffic examples. "
        "Gradient descent and Adam update its weights; the closed-form solution gives a reference."
    )
    st.caption(
        "The data is generated from a known linear rule with added noise. "
        "It demonstrates training; "
        "it is not a trained traffic predictor."
    )
    seed = st.number_input("Training seed", min_value=0, max_value=100_000, value=7, step=1)
    run = st.button("Train the example model", key="run_ml_demo")
    if not run and "ml_demo_result" not in st.session_state:
        st.info("Train the example to compare the learning curves and held-out errors.")
        return
    if run:
        with st.spinner("Training on synthetic examples…"):
            result = _cached_regression_demo(int(seed))
        st.session_state["ml_demo_result"] = result
        st.session_state["ml_demo_seed"] = int(seed)
        try:
            record = write_run_record(
                "ml-learning-demo",
                config={"training_seed": int(seed), "sample_count": result["sample_count"]},
                seed=int(seed),
                results={
                    "test_mse": result["test_mse"],
                    "iterations": result["iterations"],
                    "converged": result["converged"],
                },
            )
            st.session_state["ml_demo_record"] = record.name
        except OSError as exc:
            st.warning(f"Training finished, but its local run record could not be saved: {exc}")

    result = st.session_state["ml_demo_result"]
    if st.session_state.get("ml_demo_seed") != int(seed):
        st.info("The seed changed. Train again to calculate the new run.")
        return

    figure = go.Figure()
    for name, values in result["loss_history"].items():
        figure.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="lines", name=name))
    figure.update_layout(
        xaxis_title="Weight update",
        yaxis_title="Training mean squared error / 2",
        yaxis_type="log",
        height=350,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        [{"Method": name, "Held-out MSE": value} for name, value in result["test_mse"].items()],
        hide_index=True,
        use_container_width=True,
    )
    record_name = st.session_state.get("ml_demo_record")
    if record_name:
        st.caption(f"This seeded training run was saved locally as `{record_name}`.")
    if not result["sklearn_available"]:
        st.caption("Install the dev dependencies to include the scikit-learn comparison.")
