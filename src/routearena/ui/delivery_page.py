"""Interactive single-rider route comparison page."""

from __future__ import annotations

import csv
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from routearena.logistics.geo import distance_matrix
from routearena.logistics.routing import (
    ant_colony,
    genetic_algorithm,
    held_karp,
    nearest_neighbor,
    two_opt,
)
from routearena.records import write_run_record

_MAX_UPLOAD_BYTES = 2 * 1024 * 1024
_MAX_CUSTOMERS = 29
_EXACT_CUSTOMERS = 12
_HUB = (19.03358, 73.01816)  # Nerul railway station, approximate map center.

# Roadside reference points listed in a Navi Mumbai Municipal Corporation tender.
# The lake-side point is excluded; these five anchors are described near service roads.
_NERUL_ANCHORS = np.array(
    [
        [19.041527103, 73.008456186],
        [19.043881670, 73.009110756],
        [19.044544368, 73.026097636],
        [19.039013559, 73.029196570],
        [19.040428514, 73.023681311],
    ],
    dtype=float,
)
_METHODS = (
    "Nearest neighbor",
    "Nearest neighbor + 2-opt",
    "Held-Karp exact",
    "Genetic algorithm",
    "Ant colony",
)


def _synthetic_stops(count: int, seed: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Generate reproducible synthetic stops around documented Nerul road anchors."""
    rng = np.random.default_rng(seed)
    anchors = _NERUL_ANCHORS[rng.integers(0, len(_NERUL_ANCHORS), size=count)]
    # About 20 m in latitude and longitude at Nerul; these remain demonstration points.
    jitter = rng.uniform(-0.00018, 0.00018, size=(count, 2))
    coordinates = anchors + jitter
    names = [f"Synthetic stop {index + 1}" for index in range(count)]
    demand = rng.integers(1, 6, size=count).astype(float)
    return coordinates, names, demand


def _uploaded_stops(upload, hub: tuple[float, float]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Parse and validate a small CSV without changing the uploaded file."""
    contents = upload.getvalue()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise ValueError("CSV must be no larger than 2 MB")
    try:
        text = contents.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        required = {"name", "lat", "lon", "demand"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV columns must include name, lat, lon, and demand")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded") from exc
    if not 1 <= len(rows) <= _MAX_CUSTOMERS:
        raise ValueError("CSV must contain between 1 and 29 customer rows (plus the hub)")

    coordinates: list[tuple[float, float]] = []
    names: list[str] = []
    demand: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            name = str(row["name"]).strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            weight = float(row["demand"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid number or missing value on CSV row {line_number}") from exc
        if not name:
            raise ValueError(f"Name is empty on CSV row {line_number}")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Demand must be finite and nonnegative on row {line_number}")
        coordinates.append((lat, lon))
        names.append(name)
        demand.append(weight)

    coordinate_array = np.asarray(coordinates, dtype=float)
    # Reuse the geo module's coordinate, uniqueness, and stop-count checks.
    distance_matrix(np.vstack((hub, coordinate_array)))
    return coordinate_array, names, np.asarray(demand, dtype=float)


def _travel_time_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Apply the documented straight-line, detour, and average-speed assumptions."""
    distances_km = distance_matrix(coordinates)
    detour_factor = 1.25
    average_speed_kmph = 25.0
    return distances_km * detour_factor / average_speed_kmph * 60.0


@st.cache_data(show_spinner=False)
def _solve_routes(
    matrix: np.ndarray,
    methods: tuple[str, ...],
    seed: int,
) -> dict[str, object]:
    """Run selected solvers once for this exact matrix and seed."""
    results = {}
    if "Nearest neighbor" in methods:
        results["Nearest neighbor"] = nearest_neighbor(matrix)
    if "Nearest neighbor + 2-opt" in methods:
        baseline = nearest_neighbor(matrix)
        results["Nearest neighbor + 2-opt"] = two_opt(matrix, baseline.route)
    if "Held-Karp exact" in methods:
        results["Held-Karp exact"] = held_karp(matrix)
    if "Genetic algorithm" in methods:
        results["Genetic algorithm"] = genetic_algorithm(
            matrix, population_size=60, generations=160, seed=seed
        )
    if "Ant colony" in methods:
        results["Ant colony"] = ant_colony(matrix, iterations=100, seed=seed)
    return results


def _route_map(
    coordinates: np.ndarray,
    names: list[str],
    results: dict[str, object],
    hub: tuple[float, float],
) -> go.Figure:
    """Show stop markers and straight-line tour segments over token-free OSM tiles."""
    figure = go.Figure()
    colors = ["#1677ff", "#f97316", "#16a34a", "#9333ea", "#dc2626"]
    for index, (name, result) in enumerate(results.items()):
        ordered = coordinates[list(result.route)]
        figure.add_trace(
            go.Scattermapbox(
                lat=ordered[:, 0],
                lon=ordered[:, 1],
                mode="lines",
                name=name,
                line={"width": 3, "color": colors[index % len(colors)]},
                hovertemplate=f"{name}<extra></extra>",
            )
        )
    point_names = ["Hub", *names]
    figure.add_trace(
        go.Scattermapbox(
            lat=coordinates[:, 0],
            lon=coordinates[:, 1],
            mode="markers+text",
            name="Stops",
            text=["Hub", *(str(index + 1) for index in range(len(names)))],
            textposition="top right",
            customdata=np.asarray(point_names, dtype=object),
            hovertemplate="%{customdata}<br>%{lat:.5f}, %{lon:.5f}<extra></extra>",
            marker={"size": 12, "color": ["#111827", *["#64748b"] * len(names)]},
        )
    )
    figure.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": hub[0], "lon": hub[1]},
            "zoom": 12,
        },
        height=610,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure


def render_delivery_page() -> None:
    """Render the Delivery Planner and compare route algorithms."""
    st.title("Delivery Planner")
    st.write(
        "Choose a hub and plan one rider’s loop with several route algorithms. "
        "The map uses real geography, while customer stops and travel times are a demonstration."
    )
    hub_columns = st.columns(2)
    with hub_columns[0]:
        hub_lat = st.number_input(
            "Hub latitude", min_value=-90.0, max_value=90.0, value=_HUB[0], format="%.5f"
        )
    with hub_columns[1]:
        hub_lon = st.number_input(
            "Hub longitude", min_value=-180.0, max_value=180.0, value=_HUB[1], format="%.5f"
        )
    hub = (float(hub_lat), float(hub_lon))
    st.caption(
        "Synthetic stops use a fixed seed and are jittered around mapped Nerul roadside anchors. "
        "CSV columns: name, lat, lon, demand. Uploads are limited to 2 MB and 29 customers "
        "plus the hub."
    )

    source = st.radio("Stop data", ["Synthetic stops", "Upload CSV"], horizontal=True)
    uploaded = None
    if source == "Synthetic stops":
        left, right = st.columns(2)
        with left:
            count = st.slider("Customer stops", min_value=3, max_value=_MAX_CUSTOMERS, value=8)
        with right:
            seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=17)
        coordinates_customers, names, demand = _synthetic_stops(int(count), int(seed))
    else:
        uploaded = st.file_uploader("Upload customer locations", type=["csv"])
        seed = st.number_input("Solver seed", min_value=0, max_value=1_000_000, value=17)
        if uploaded is None:
            st.info("Upload a CSV with name, lat, lon, and demand columns to continue.")
            return
        try:
            coordinates_customers, names, demand = _uploaded_stops(uploaded, hub)
        except ValueError as exc:
            st.error(str(exc))
            return

    coordinates = np.vstack((hub, coordinates_customers))
    try:
        matrix = _travel_time_matrix(coordinates)
    except ValueError as exc:
        st.error(f"Invalid hub or stop locations: {exc}")
        return
    customer_count = len(names)
    default_methods = [
        "Nearest neighbor",
        "Nearest neighbor + 2-opt",
        "Genetic algorithm",
        "Ant colony",
    ]
    if customer_count <= _EXACT_CUSTOMERS:
        default_methods.insert(2, "Held-Karp exact")
    methods = st.multiselect(
        "Algorithms to compare",
        _METHODS,
        default=default_methods,
        help="Held-Karp proves the best tour for up to 12 customers and is hidden above that limit.",
    )
    if customer_count > _EXACT_CUSTOMERS:
        st.caption("Held-Karp is limited to 12 customer stops because its exact search grows exponentially.")

    st.caption(
        f"Travel-time assumption: straight-line Haversine distance × 1.25 detour factor ÷ "
        f"25 km/h average speed. These are assumptions, not measured road times. "
        f"Total listed demand: {demand.sum():g} units."
    )
    data_key = (source, tuple(coordinates.ravel()), tuple(names), tuple(methods), int(seed))
    st.session_state["delivery_current_key"] = data_key
    run = st.button("Plan routes", type="primary", disabled=not methods)
    if run:
        with st.spinner("Comparing route plans…"):
            results = _solve_routes(matrix, tuple(methods), int(seed))
        st.session_state["delivery_results"] = results
        st.session_state["delivery_key"] = data_key
        st.session_state["delivery_time_matrix"] = matrix
        st.session_state["delivery_coords"] = coordinates
        st.session_state["delivery_names"] = names
        record_results = {
            name: {
                "route": list(result.route),
                "cost_minutes": result.cost,
                "runtime_seconds": result.runtime_seconds,
                "metadata": result.metadata,
            }
            for name, result in results.items()
        }
        try:
            record_path = write_run_record(
                "routing",
                config={
                    "customer_count": customer_count,
                    "stop_source": source,
                    "algorithms": list(methods),
                    "detour_factor": 1.25,
                    "average_speed_kmph": 25.0,
                },
                seed=int(seed),
                results=record_results,
            )
            st.session_state["delivery_record_name"] = record_path.name
        except OSError as exc:
            st.warning(f"Routes were planned, but the local run record could not be saved: {exc}")

    if st.session_state.get("delivery_key") != data_key:
        st.info("Choose the locations and algorithms, then select Plan routes.")
        return
    results = st.session_state["delivery_results"]
    exact_cost = results.get("Held-Karp exact")
    exact_cost = None if exact_cost is None else exact_cost.cost
    st.plotly_chart(_route_map(coordinates, names, results, hub), use_container_width=True)

    st.subheader("Route comparison")
    columns = st.columns(min(3, len(results)))
    for index, (name, result) in enumerate(results.items()):
        column = columns[index % len(columns)]
        with column:
            column.metric("Algorithm", name)
            column.metric("Estimated tour time", f"{result.cost:.1f} min")
            column.caption(f"Runtime: {result.runtime_seconds * 1000:.2f} ms")
            if exact_cost is not None and name != "Held-Karp exact":
                gap = max(0.0, (result.cost - exact_cost) / max(exact_cost, 1e-12) * 100.0)
                column.caption(f"Gap from exact best: {gap:.2f}%")

    st.caption(
        "Lines connect stop coordinates directly; they are not road-following directions. "
        "Route costs are based on synthetic travel times."
    )
    record_name = st.session_state.get("delivery_record_name")
    if record_name:
        st.caption(f"Latest route run saved locally as `{record_name}`.")
    st.caption(
        "Map references: [Nerul railway station](https://mapcarta.com/W572630813) and "
        "[Navi Mumbai Municipal Corporation roadside locations](https://nmmc.etenders.in/tender_document/tender_7805/tender_document/tender_notice_document/1678080412Chargingstationfinaltendercompressed.pdf)."
    )
