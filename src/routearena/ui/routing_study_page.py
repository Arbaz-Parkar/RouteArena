"""Streamlit page for a reproducible multi-seed routing benchmark."""

from __future__ import annotations

import csv
import io
import json

import plotly.graph_objects as go
import streamlit as st

from routearena.logistics.routing.benchmark import run_burma14_study
from routearena.logistics.routing.tsplib import BURMA14_OPTIMAL_COST


def _csv_bytes(rows: list[dict]) -> bytes:
    """Convert study rows to downloadable UTF-8 CSV bytes."""
    output = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8")


def _cost_figure(summary: list[dict]) -> go.Figure:
    """Plot stochastic-method mean costs and their 95% confidence intervals."""
    rows = [row for row in summary if row.get("ci95_cost_lower_km") is not None]
    means = [float(row["mean_cost_km"]) for row in rows]
    figure = go.Figure(
        go.Bar(
            x=[row["method"] for row in rows],
            y=means,
            marker_color=["#0F766E", "#2563EB"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [
                    float(row["ci95_cost_upper_km"]) - float(row["mean_cost_km"]) for row in rows
                ],
                "arrayminus": [
                    float(row["mean_cost_km"]) - float(row["ci95_cost_lower_km"]) for row in rows
                ],
                "thickness": 1.5,
                "width": 5,
            },
            hovertemplate=(
                "%{x}<br>Mean cost: %{y:.1f} km<br>95% confidence interval shown<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=BURMA14_OPTIMAL_COST,
        line_dash="dash",
        line_color="#B45309",
        annotation_text=f"Published optimum: {BURMA14_OPTIMAL_COST:,} km",
    )
    figure.update_layout(
        title="Stochastic solver performance across 30 fixed seeds",
        xaxis_title="Algorithm",
        yaxis_title="Tour cost (km; lower is better)",
        height=390,
        showlegend=False,
    )
    return figure


def _display_summary(summary: list[dict]) -> list[dict]:
    """Build a compact, readable view while retaining full precision in downloads."""
    display_rows = []
    for row in summary:
        lower = row.get("ci95_cost_lower_km")
        upper = row.get("ci95_cost_upper_km")
        interval = "—" if lower is None else f"{lower:.1f}–{upper:.1f} km"
        mean_cost = row.get("mean_cost_km")
        mean_gap = row.get("mean_optimality_gap_percent")
        mean_runtime = row.get("mean_runtime_seconds")
        display_rows.append(
            {
                "Method": row["method"],
                "Runs": row["runs"],
                "Mean cost (km)": None if mean_cost is None else round(mean_cost, 2),
                "Std. dev. (km)": (
                    None
                    if row.get("standard_deviation_cost_km") is None
                    else round(row["standard_deviation_cost_km"], 2)
                ),
                "95% CI for mean": interval,
                "Mean gap to optimum (%)": None if mean_gap is None else round(mean_gap, 3),
                "Mean runtime (s)": (None if mean_runtime is None else round(mean_runtime, 5)),
                "Status": row["status"],
            }
        )
    return display_rows


def render_routing_study_page() -> None:
    """Run, inspect, and download the fixed-seed TSPLIB benchmark study."""
    st.title("Routing Study")
    st.write(
        "Compare RouteArena’s stochastic routing methods across 30 fixed seeds on the "
        "standard TSPLIB burma14 problem. Held–Karp verifies the published optimum; "
        "nearest neighbor, 2-opt, and optional OR-Tools provide reference results."
    )
    st.caption(
        "Confidence intervals describe seed-to-seed variation on this one benchmark instance. "
        "They do not establish performance across all routing problems."
    )
    st.caption(
        "Benchmark references: [TSPLIB burma14 coordinates]"
        "(https://raw.githubusercontent.com/mastqe/tsplib/master/burma14.tsp) · "
        "[published optimum](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html)."
    )

    st.subheader("Study design")
    st.markdown(
        "- **Instance:** TSPLIB `burma14`, 14 cities, published optimum 3,323 km.\n"
        "- **Stochastic trials:** genetic algorithm and ant colony, each run on the same 30 seeds.\n"
        "- **Statistics:** mean, sample standard deviation, median, range, and two-sided "
        "95% Student-t confidence intervals for mean cost, gap, and runtime.\n"
        "- **References:** exact Held–Karp, deterministic nearest neighbor and 2-opt, and "
        "OR-Tools when its optional dependency is installed."
    )

    if st.button("Run 30-seed routing study", type="primary"):
        with st.spinner("Running 60 seeded heuristic trials and reference solvers…"):
            try:
                report = run_burma14_study()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"The routing study could not be completed: {exc}")
            else:
                st.session_state["routing_study_report"] = report

    report = st.session_state.get("routing_study_report")
    if report is None:
        st.info("Run the study to calculate results and enable the downloads.")
        return

    metadata = report["metadata"]
    summary = report["summary"]
    trials = report["trials"]
    stochastic_rows = [row for row in summary if row.get("ci95_cost_lower_km") is not None]
    st.subheader("Results")
    first, second, third = st.columns(3)
    first.metric("Exact optimum", f"{metadata['settings']['known_optimum_km']:,} km")
    second.metric("Seeds per stochastic method", str(metadata["settings"]["solver_seed_count"]))
    third.metric(
        "Heuristic trials", str(len(stochastic_rows) * metadata["settings"]["solver_seed_count"])
    )

    st.plotly_chart(_cost_figure(summary), use_container_width=True)
    st.dataframe(_display_summary(summary), hide_index=True, use_container_width=True)

    if not metadata["ortools_available"]:
        st.info(
            "OR-Tools was not installed, so its reference row is unavailable. "
            'Install it with `python -m pip install -e ".[benchmarks]"` and rerun the study.'
        )

    st.subheader("Download results")
    downloads = st.columns(3)
    downloads[0].download_button(
        "Summary CSV",
        data=_csv_bytes(summary),
        file_name="routing_study_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    downloads[1].download_button(
        "All trials CSV",
        data=_csv_bytes(trials),
        file_name="routing_study_trials.csv",
        mime="text/csv",
        use_container_width=True,
    )
    downloads[2].download_button(
        "Full report JSON",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="routing_study_report.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Settings and environment details"):
        st.json(metadata)
        st.caption(f"Local run record: `{metadata['run_record']}`")
