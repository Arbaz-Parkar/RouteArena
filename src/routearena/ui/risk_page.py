"""Monte Carlo and interval analysis for the latest planned delivery routes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from routearena.logistics.uncertainty import (
    TRAFFIC_MODELS,
    convergence_curve,
    interval_route,
    simulate_route,
)
from routearena.records import write_run_record


def render_risk_page() -> None:
    """Render risk summaries for the candidate routes from Delivery Planner."""
    st.title("Risk and Uncertainty")
    st.write(
        "Explore how uncertain traffic can change the time a route takes. "
        "The traffic model is simulated and does not use live or measured traffic."
    )

    results = st.session_state.get("delivery_results")
    delivery_key = st.session_state.get("delivery_key")
    current_key = st.session_state.get("delivery_current_key")
    matrix = st.session_state.get("delivery_time_matrix")
    if not results or matrix is None:
        st.info("Run at least one route on the Delivery Planner page before analyzing risk.")
        return
    if current_key != delivery_key:
        st.info("The planner inputs have changed. Return to Delivery Planner and run the updated plan.")
        return

    method_names = list(results)
    controls = st.columns(3)
    with controls[0]:
        selected_method = st.selectbox("Route to inspect", method_names)
    with controls[1]:
        traffic_level = st.selectbox("Traffic assumption", list(TRAFFIC_MODELS), index=1)
    with controls[2]:
        deadline = st.number_input(
            "Delivery deadline (minutes)", min_value=1.0, max_value=600.0, value=90.0
        )

    settings = st.columns(4)
    with settings[0]:
        sample_count = st.select_slider(
            "Simulation days",
            options=[500, 1_000, 2_500, 5_000, 10_000, 20_000],
            value=5_000,
        )
    with settings[1]:
        seed = st.number_input("Simulation seed", min_value=0, max_value=1_000_000, value=29)
    with settings[2]:
        lower_multiplier = st.slider("Interval lower factor", 0.5, 1.0, 0.8, 0.05)
    with settings[3]:
        upper_multiplier = st.slider("Interval upper factor", 1.0, 3.0, 1.8, 0.1)

    model = TRAFFIC_MODELS[traffic_level]
    st.caption(
        f"Model assumptions: shared city-wide factor mean {model.shared_mean:.2f}, "
        f"shared variation CV {model.shared_cv:.2f}, independent per-leg variation CV "
        f"{model.leg_cv:.2f}. Each is a documented teaching parameter, not measured traffic."
    )
    analysis_key = (
        delivery_key,
        selected_method,
        traffic_level,
        float(deadline),
        int(sample_count),
        int(seed),
        float(lower_multiplier),
        float(upper_multiplier),
    )
    run = st.button("Run risk analysis", type="primary")
    if run:
        with st.spinner("Simulating traffic uncertainty for each candidate route…"):
            candidate_risks = {
                name: simulate_route(
                    matrix,
                    result.route,
                    deadline_minutes=float(deadline),
                    traffic_level=traffic_level,
                    sample_count=int(sample_count),
                    seed=int(seed),
                )
                for name, result in results.items()
            }
            selected_interval = interval_route(
                matrix,
                results[selected_method].route,
                lower_multiplier=float(lower_multiplier),
                upper_multiplier=float(upper_multiplier),
            )
        st.session_state["risk_analysis_key"] = analysis_key
        st.session_state["risk_candidate_results"] = candidate_risks
        st.session_state["risk_selected_interval"] = selected_interval
        record_results = {
            name: {
                "mean_minutes": result.mean_minutes,
                "percentile_90_minutes": result.percentile_90_minutes,
                "percentile_95_minutes": result.percentile_95_minutes,
                "probability_late": result.probability_late,
            }
            for name, result in candidate_risks.items()
        }
        selected_result = candidate_risks[selected_method]
        record_results["selected_method"] = {
            "name": selected_method,
            "mean_minutes": selected_result.mean_minutes,
            "probability_late": selected_result.probability_late,
        }
        record_results["selected_probability_late"] = selected_result.probability_late
        record_results["interval"] = {
            "lower_total_minutes": selected_interval.lower_total_minutes,
            "worst_case_minutes": selected_interval.worst_case_minutes,
            "lower_multiplier": selected_interval.lower_multiplier,
            "upper_multiplier": selected_interval.upper_multiplier,
        }
        try:
            record_path = write_run_record(
                "risk",
                config={
                    "customer_count": matrix.shape[0] - 1,
                    "candidate_methods": list(results),
                    "selected_method": selected_method,
                    "traffic_level": traffic_level,
                    "deadline_minutes": float(deadline),
                    "sample_count": int(sample_count),
                    "lower_multiplier": float(lower_multiplier),
                    "upper_multiplier": float(upper_multiplier),
                },
                seed=int(seed),
                results=record_results,
            )
            st.session_state["risk_record_name"] = record_path.name
        except OSError as exc:
            st.warning(f"Risk analysis finished, but its local record could not be saved: {exc}")

    if st.session_state.get("risk_analysis_key") != analysis_key:
        st.info("Choose the assumptions, then select Run risk analysis.")
        return

    candidate_risks = st.session_state["risk_candidate_results"]
    selected = candidate_risks[selected_method]
    interval = st.session_state["risk_selected_interval"]
    shortest_name = min(results, key=lambda name: (results[name].cost, name))
    robust_name = min(
        candidate_risks,
        key=lambda name: (candidate_risks[name].percentile_90_minutes, name),
    )

    st.subheader(f"Risk summary: {selected_method}")
    metrics = st.columns(4)
    metrics[0].metric("Mean time", f"{selected.mean_minutes:.1f} min")
    metrics[1].metric("90th percentile", f"{selected.percentile_90_minutes:.1f} min")
    metrics[2].metric("95th percentile", f"{selected.percentile_95_minutes:.1f} min")
    metrics[3].metric("Chance of missing deadline", f"{selected.probability_late:.1%}")
    st.caption(
        f"Median {selected.percentile_50_minutes:.1f} min · "
        f"standard deviation {selected.standard_deviation_minutes:.1f} min · "
        f"{selected.sample_count:,} simulated days"
    )

    histogram = go.Figure(
        go.Histogram(
            x=selected.samples,
            nbinsx=40,
            marker_color="#3977c3",
            hovertemplate="Total time: %{x:.1f} min<br>Simulated days: %{y}<extra></extra>",
        )
    )
    histogram.add_vline(
        x=selected.deadline_minutes,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Deadline",
    )
    histogram.update_layout(
        xaxis_title="Simulated route time (minutes)",
        yaxis_title="Number of simulated days",
        height=380,
    )
    st.plotly_chart(histogram, use_container_width=True)

    counts, means, percentiles_90 = convergence_curve(selected.samples)
    convergence = go.Figure()
    convergence.add_trace(
        go.Scatter(x=counts, y=means, mode="lines+markers", name="Running mean")
    )
    convergence.add_trace(
        go.Scatter(
            x=counts,
            y=percentiles_90,
            mode="lines+markers",
            name="Running 90th percentile",
        )
    )
    convergence.update_layout(
        xaxis_title="Simulated days included",
        yaxis_title="Route time (minutes)",
        height=340,
    )
    st.subheader("Monte Carlo convergence")
    st.plotly_chart(convergence, use_container_width=True)

    st.subheader("Shortest route and robust candidate")
    shortest_risk = candidate_risks[shortest_name]
    robust_risk = candidate_risks[robust_name]
    comparison = st.columns(2)
    with comparison[0]:
        st.markdown(f"**Shortest deterministic plan: {shortest_name}**")
        st.write(
            f"Estimated tour: {results[shortest_name].cost:.1f} min. "
            f"Simulated 90th percentile: {shortest_risk.percentile_90_minutes:.1f} min; "
            f"deadline risk: {shortest_risk.probability_late:.1%}."
        )
    with comparison[1]:
        st.markdown(f"**Robust candidate: {robust_name}**")
        st.write(
            f"Best 90th percentile among the selected solver routes: "
            f"{robust_risk.percentile_90_minutes:.1f} min; "
            f"deadline risk: {robust_risk.probability_late:.1%}."
        )

    st.subheader("Interval bounds for the selected route")
    st.write(
        f"With every leg between ×{interval.lower_multiplier:.2f} and "
        f"×{interval.upper_multiplier:.2f} of its base estimate, the route total ranges from "
        f"{interval.lower_total_minutes:.1f} to {interval.worst_case_minutes:.1f} minutes. "
        "The upper total is the set-based worst case under these bounds."
    )
    labels = ["Hub", *st.session_state.get("delivery_names", [])]
    leg_names = [
        f"{labels[left]} → {labels[right]}"
        for left, right in zip(selected.route[:-1], selected.route[1:], strict=True)
    ]
    bounds = go.Figure()
    bounds.add_trace(
        go.Bar(
            x=leg_names,
            y=interval.lower_leg_minutes,
            name="Lower bound",
            marker_color="#60a5fa",
        )
    )
    bounds.add_trace(
        go.Bar(
            x=leg_names,
            y=interval.upper_leg_minutes,
            name="Upper bound",
            marker_color="#f97316",
        )
    )
    bounds.update_layout(
        barmode="group",
        xaxis_title="Route leg",
        yaxis_title="Travel time bound (minutes)",
        height=380,
    )
    st.plotly_chart(bounds, use_container_width=True)
    record_name = st.session_state.get("risk_record_name")
    if record_name:
        st.caption(f"Latest risk run saved locally as `{record_name}`.")
