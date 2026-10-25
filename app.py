"""Streamlit entry point for RouteArena."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from routearena.optimizers import (
    adadelta,
    adagrad,
    adam,
    bfgs,
    gradient_descent,
    newton,
    rmsprop,
    secant_minimize,
)
from routearena.optimizers.line_search import (
    fibonacci_search,
    golden_section_search,
    quadratic_fit_search,
)
from routearena.optimizers.test_functions import (
    parabola,
    parabola_grad,
    quadratic_bowl,
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)

st.set_page_config(page_title="RouteArena | Optimizer Arena", layout="wide")

_ALGORITHMS = {
    "Gradient descent": gradient_descent,
    "Adagrad": adagrad,
    "RMSProp": rmsprop,
    "Adadelta": adadelta,
    "Adam": adam,
    "Newton": newton,
    "BFGS": bfgs,
}
_FUNCTIONS = {
    "Quadratic bowl": (quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess),
    "Rosenbrock valley": (rosenbrock, rosenbrock_grad, rosenbrock_hess),
}


@st.cache_data(show_spinner=False)
def _run_selected(
    function_name: str,
    algorithm_names: tuple[str, ...],
    start: tuple[float, float],
):
    """Run the chosen deterministic optimizers; cache by all computation inputs."""
    fun, grad, hess = _FUNCTIONS[function_name]
    results = {}
    for name in algorithm_names:
        algorithm = _ALGORITHMS[name]
        options = {"max_iter": 2500, "grad_tol": 1e-6}
        if name == "Newton":
            options["hess"] = hess
        if name in {"Adam", "RMSProp", "Adagrad"}:
            options["learning_rate"] = 0.1 if function_name == "Quadratic bowl" else 0.01
        results[name] = algorithm(fun, grad, np.asarray(start, dtype=float), **options)
    return results


@st.cache_data(show_spinner=False)
def _surface(function_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache a contour grid once for each mathematical surface."""
    fun, _, _ = _FUNCTIONS[function_name]
    bounds = (-2.0, 3.0) if function_name == "Quadratic bowl" else (-2.0, 2.0)
    axis = np.linspace(*bounds, 100)
    xx, yy = np.meshgrid(axis, axis)
    zz = np.empty_like(xx)
    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            zz[row, col] = fun(np.array([xx[row, col], yy[row, col]]))
    return axis, zz


@st.cache_data(show_spinner=False)
def _run_scalar_searches():
    """Compare the syllabus's one-dimensional search methods on one parabola."""
    results = {
        "Golden section": golden_section_search(parabola, -2.0, 8.0),
        "Fibonacci": fibonacci_search(parabola, -2.0, 8.0),
        "Quadratic fit": quadratic_fit_search(parabola, 0.0, 3.0, 6.0),
        "Secant": secant_minimize(parabola, parabola_grad, -2.0, x1=0.0),
    }
    return results


def _render_scalar_searches() -> None:
    """Show the scalar line searches and secant method on a shared example."""
    st.subheader("One-dimensional search methods")
    st.write(
        "These methods use a one-variable parabola with its minimum at x = 3. "
        "The markers show where each method finished."
    )
    results = _run_scalar_searches()
    axis = np.linspace(-2.0, 8.0, 300)
    figure = go.Figure(go.Scatter(x=axis, y=[parabola(x) for x in axis], name="Objective"))
    for name, result in results.items():
        point = float(np.asarray(result.x).reshape(-1)[0])
        figure.add_trace(
            go.Scatter(
                x=[point],
                y=[parabola(point)],
                mode="markers",
                name=name,
                marker={"size": 10},
            )
        )
    figure.update_layout(xaxis_title="x", yaxis_title="Objective", height=350)
    st.plotly_chart(figure, use_container_width=True)
    metrics = st.columns(len(results))
    for column, (name, result) in zip(metrics, results.items(), strict=False):
        with column:
            point = float(np.asarray(result.x).reshape(-1)[0])
            column.metric(name, f"x = {point:.6g}")
            column.caption(f"{result.nit} steps · {result.nfev} objective calls")


def _contour_figure(function_name: str, results, step: int) -> go.Figure:
    """Draw a function's contour map and the selected optimizer paths."""
    axis, zz = _surface(function_name)

    fig = go.Figure(
        go.Contour(
            x=axis,
            y=axis,
            z=zz,
            colorscale="Viridis",
            contours={"showlines": True},
            colorbar={"title": "Objective"},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>cost=%{z:.3f}<extra></extra>",
        )
    )
    colors = ["#ff7f0e", "#e45756", "#72b7b2", "#b279a2", "#f2cf5b", "#4c78a8", "#54a24b"]
    for index, (name, result) in enumerate(results.items()):
        path = result.path[: step + 1]
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                name=name,
                line={"color": colors[index % len(colors)], "width": 2},
                marker={"size": 5},
            )
        )
    fig.update_layout(
        height=620,
        xaxis_title="Decision variable x₁",
        yaxis_title="Decision variable x₂",
        legend_title="Optimizer",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def main() -> None:
    """Render the first RouteArena page."""
    st.title("Optimizer Arena")
    st.write(
        "Watch several optimization methods search the same mathematical landscape. "
        "Each line shows the sequence of guesses the method makes before it settles near a minimum."
    )

    left, right = st.columns([1, 2])
    with left:
        function_name = st.selectbox("Landscape", list(_FUNCTIONS))
        algorithms = st.multiselect(
            "Methods to compare",
            list(_ALGORITHMS),
            default=["Gradient descent", "Adam", "BFGS"],
            help="Each method follows a different rule for choosing its next point.",
        )
        start_x = st.number_input("Starting x₁", value=-1.5, min_value=-4.0, max_value=4.0)
        start_y = st.number_input("Starting x₂", value=2.0, min_value=-4.0, max_value=4.0)
        run = st.button("Run the race", type="primary", disabled=not algorithms)
        st.caption(
            "These surfaces are mathematical examples. Delivery locations and route planning "
            "will be added in later milestones."
        )

    if run:
        with st.spinner("Running the selected optimizers…"):
            names = tuple(algorithms)
            results = _run_selected(function_name, names, (float(start_x), float(start_y)))
        st.session_state["arena_results"] = results
        st.session_state["arena_function"] = function_name
        st.session_state["arena_config"] = (tuple(algorithms), float(start_x), float(start_y))

    results = st.session_state.get("arena_results")
    selected_config = (tuple(algorithms), float(start_x), float(start_y))
    if (
        not results
        or st.session_state.get("arena_function") != function_name
        or st.session_state.get("arena_config") != selected_config
    ):
        with right:
            st.info("Choose a landscape and methods, then run the race to see their paths.")
        _render_scalar_searches()
        return

    max_step = max(len(result.path) - 1 for result in results.values())
    step = st.slider("Show path through step", min_value=0, max_value=max_step, value=max_step)
    with right:
        st.plotly_chart(_contour_figure(function_name, results, step), use_container_width=True)

    st.subheader("Results")
    columns = st.columns(len(results))
    for column, (name, result) in zip(columns, results.items(), strict=False):
        with column:
            column.metric("Method", name)
            column.metric("Final objective", f"{result.fun:.6g}")
            column.caption(
                f"{result.nit} steps · {result.nfev} objective calls · "
                f"{'converged' if result.success else 'stopped: ' + result.message}"
            )
    _render_scalar_searches()


if __name__ == "__main__":
    main()

