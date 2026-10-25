"""Offline question answering over local notes and saved experiment summaries."""

from __future__ import annotations

import streamlit as st

from routearena.genai.rag.retrieval import answer_from_knowledge, retrieve

_MAX_MESSAGES = 12


def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only a few recent chat turns in the current browser session."""
    return messages[-_MAX_MESSAGES:]


def _explain_latest_results() -> str | None:
    """Explain current solver results using only values produced by RouteArena."""
    routes = st.session_state.get("delivery_results")
    delivery_key = st.session_state.get("delivery_key")
    if not routes or delivery_key != st.session_state.get("delivery_current_key"):
        return None
    fastest_name = min(routes, key=lambda name: (routes[name].cost, name))
    fastest = routes[fastest_name]
    explanation = (
        f"The shortest candidate in the latest planner run was {fastest_name}, "
        f"with an estimated tour time of {fastest.cost:.1f} minutes. "
    )
    exact = routes.get("Held-Karp exact")
    if exact is not None:
        explanation += (
            f"Held-Karp found the exact best tour for this small instance at "
            f"{exact.cost:.1f} minutes. "
        )
    analysis_key = st.session_state.get("risk_analysis_key")
    risk_results = st.session_state.get("risk_candidate_results")
    if not analysis_key or analysis_key[0] != delivery_key:
        risk_results = None
    if risk_results:
        robust_name = min(
            risk_results,
            key=lambda name: (risk_results[name].percentile_90_minutes, name),
        )
        robust = risk_results[robust_name]
        explanation += (
            f"Under the last simulated traffic assumptions, {robust_name} had the lowest "
            f"90th-percentile time at {robust.percentile_90_minutes:.1f} minutes. "
            f"Its simulated chance of missing the selected deadline was "
            f"{robust.probability_late:.1%}. "
        )
    explanation += "These are model outputs, not measured road times. [current RouteArena run]"
    return explanation


def render_assistant_page() -> None:
    """Render the no-key, offline RouteArena assistant."""
    st.title("Assistant")
    st.write(
        "Ask about an optimization method, route, or project assumption. "
        "The offline assistant searches local notes and saved run summaries, then cites its sources."
    )
    st.caption("Offline mode · local TF-IDF retrieval · no language model or API key is used")

    explain = st.button(
        "Explain the latest route and risk results",
        disabled=(
            not st.session_state.get("delivery_results")
            or st.session_state.get("delivery_key")
            != st.session_state.get("delivery_current_key")
        ),
    )
    if explain:
        explanation = _explain_latest_results()
        if explanation is not None:
            messages = st.session_state.get("assistant_messages", [])
            messages.append({"role": "assistant", "content": explanation})
            st.session_state["assistant_messages"] = _trim_history(messages)

    messages = st.session_state.get("assistant_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about RouteArena")
    if question:
        messages.append({"role": "user", "content": question})
        matches = retrieve(question)
        answer = answer_from_knowledge(question, matches)
        messages.append({"role": "assistant", "content": answer})
        messages = _trim_history(messages)
        st.session_state["assistant_messages"] = messages
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
