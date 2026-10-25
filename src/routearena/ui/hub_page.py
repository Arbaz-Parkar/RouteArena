"""Streamlit demonstration of demand-weighted hub placement."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from routearena.logistics.hub import HubCostModel
from routearena.optimizers import adam, bfgs, gradient_descent, newton
from routearena.records import write_run_record


@st.cache_data(show_spinner=False)
def _make_customers(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible illustrative customer set in a local plane."""
    generator = np.random.default_rng(seed)
    locations = generator.uniform([-3500.0, -2500.0], [3500.0, 2500.0], size=(count, 2))
    demands = generator.integers(1, 6, size=count).astype(float)
    return locations, demands


@st.cache_data(show_spinner=False)
def _solve_hub(seed: int, count: int):
    """Fit four optimizers to the same synthetic hub-cost surface."""
    customers, demands = _make_customers(seed, count)
    model = HubCostModel(customers, demands)
    start = np.array([-4500.0, 3500.0])
    results = {
        "Gradient descent": gradient_descent(model.objective, model.gradient, start),
        "Adam": adam(
            model.objective,
            model.gradient,
            start,
            learning_rate=50.0,
            grad_tol=1e-2,
            max_iter=2000,
        ),
        "Newton": newton(model.objective, model.gradient, start, hess=model.hessian),
        "BFGS": bfgs(model.objective, model.gradient, start),
    }
    return customers, demands, model, results


def _figure(customers: np.ndarray, demands: np.ndarray, model: HubCostModel, results) -> go.Figure:
    """Draw customer points, the hub-cost contours, and optimizer paths."""
    lower = np.min(customers, axis=0) - 1200.0
    upper = np.max(customers, axis=0) + 1200.0
    x_axis = np.linspace(lower[0], upper[0], 75)
    y_axis = np.linspace(lower[1], upper[1], 75)
    xx, yy = np.meshgrid(x_axis, y_axis)
    costs = np.empty_like(xx)
    for row in range(xx.shape[0]):
        for column in range(xx.shape[1]):
            costs[row, column] = model.objective(np.array([xx[row, column], yy[row, column]]))

    figure = go.Figure(
        go.Contour(
            x=x_axis / 1000.0,
            y=y_axis / 1000.0,
            z=costs,
            colorscale="Viridis",
            contours={"showlines": True},
            colorbar={"title": "Weighted distance<br>(demand·m)"},
            hovertemplate="x=%{x:.2f} km<br>y=%{y:.2f} km<br>cost=%{z:.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=customers[:, 0] / 1000.0,
            y=customers[:, 1] / 1000.0,
            mode="markers",
            name="Synthetic customers",
            marker={"size": 8 + 2 * demands, "color": "white", "line": {"color": "black", "width": 1}},
            text=[f"Demand weight: {weight:g}" for weight in demands],
            hovertemplate="x=%{x:.2f} km<br>y=%{y:.2f} km<br>%{text}<extra></extra>",
        )
    )
    colors = ["#ff7f0e", "#e45756", "#72b7b2", "#b279a2"]
    for index, (name, result) in enumerate(results.items()):
        path = result.path
        figure.add_trace(
            go.Scatter(
                x=path[:, 0] / 1000.0,
                y=path[:, 1] / 1000.0,
                mode="lines+markers",
                name=name,
                line={"color": colors[index], "width": 2},
                marker={"size": 4},
            )
        )
    figure.update_layout(
        xaxis_title="Local east-west distance (km)",
        yaxis_title="Local north-south distance (km)",
        legend_title="Points and optimizer paths",
        height=620,
    )
    return figure


def render_hub_page() -> None:
    """Render the hub placement page and its optimizer comparison."""
    st.title("Hub Placement")
    st.write(
        "A delivery hub should sit where it keeps the weighted customer distance small. "
        "This page compares optimizers on a smooth cost map using reproducible synthetic points."
    )
    st.caption(
        "The points are synthetic and live in a local flat coordinate plane; this is not a street map. "
        "The objective sums demand-weighted straight-line distances, not road travel time."
    )
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=17, step=1)
    count = st.slider("Synthetic customer count", min_value=3, max_value=30, value=10)
    if not st.button("Compare hub optimizers", type="primary"):
        st.info("Set the seed and customer count, then compare the optimizers.")
        return

    with st.spinner("Finding hub locations…"):
        customers, demands, model, results = _solve_hub(int(seed), int(count))
    record_results = {
        name: {
            "point_km": (result.x / 1000.0).tolist(),
            "objective": result.fun,
            "iterations": result.nit,
            "converged": result.success,
        }
        for name, result in results.items()
    }
    try:
        record_path = write_run_record(
            "hub-placement",
            config={"customer_count": int(count), "surface": "smoothed weighted distance"},
            seed=int(seed),
            results=record_results,
        )
        st.caption(f"Hub comparison saved locally as `{record_path.name}`.")
    except OSError as exc:
        st.warning(f"Hub comparison finished, but its local run record could not be saved: {exc}")
    st.plotly_chart(_figure(customers, demands, model, results), use_container_width=True)

    columns = st.columns(len(results))
    for column, (name, result) in zip(columns, results.items(), strict=False):
        with column:
            column.metric(name, f"Cost {result.fun:,.1f}")
            column.caption(
                f"Hub: ({result.x[0] / 1000:.2f}, {result.x[1] / 1000:.2f}) km · "
                f"{result.nit} steps · {'converged' if result.success else result.message}"
            )

