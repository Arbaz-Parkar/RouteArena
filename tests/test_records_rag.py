"""Checks for local run records and offline retrieval."""

from __future__ import annotations

import json

import numpy as np

from routearena.genai.rag.retrieval import (
    answer_from_knowledge,
    build_index,
    build_knowledge_chunks,
    load_chunks,
    retrieve,
)
from routearena.records import read_run_records, write_run_record


def test_run_record_round_trip_keeps_config_seed_and_results(tmp_path) -> None:
    path = write_run_record(
        "routing",
        config={"customer_count": 4, "algorithms": ["nearest_neighbor"]},
        seed=12,
        results={
            "nearest_neighbor": {
                "cost_minutes": np.float64(43.2),
                "route": np.array([0, 2, 1, 0]),
            }
        },
        runs_dir=tmp_path,
    )
    record = read_run_records(runs_dir=tmp_path, limit=1)[0]
    assert path.exists()
    assert record["run_type"] == "routing"
    assert record["config"]["customer_count"] == 4
    assert record["seed"] == 12
    assert record["results"]["nearest_neighbor"]["cost_minutes"] == 43.2
    assert record["results"]["nearest_neighbor"]["route"] == [0, 2, 1, 0]
    assert "numpy" in record["library_versions"]


def test_heading_chunking_caps_chunk_size_and_adds_unit_metadata(tmp_path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    body = "A route visits stops and returns to its hub. " * 30
    (knowledge / "routing.md").write_text(f"# Routing\n\n## Tour\n\n{body}", encoding="utf-8")
    chunks = build_knowledge_chunks(knowledge, chunk_size=160, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 160 for chunk in chunks)
    assert all(chunk.unit == "Unit 4: discrete optimization" for chunk in chunks)


def test_retrieval_returns_cited_local_passage(tmp_path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "routing.md").write_text(
        "# Routing\n\n## Nearest neighbor\n\n"
        "Nearest neighbor chooses the closest unvisited customer at every step. "
        "It is fast but does not guarantee the best tour.\n\n"
        "## Ant colony\n\nAnt colony search reinforces edges from shorter routes.",
        encoding="utf-8",
    )
    matches = retrieve(
        "How does nearest neighbor choose the next customer?",
        knowledge_dir=knowledge,
        runs_dir=tmp_path / "runs",
        min_score=0.05,
    )
    assert matches
    assert matches[0].chunk.section == "Nearest neighbor"
    answer = answer_from_knowledge("nearest neighbor", matches)
    assert "closest unvisited customer" in answer
    assert "[routing.md: Nearest neighbor]" in answer


def test_offline_assistant_refuses_when_retrieval_finds_nothing() -> None:
    assert answer_from_knowledge("What is the weather?", []) == (
        "I could not find that in my knowledge base."
    )


def test_saved_run_summaries_are_added_to_retrieval(tmp_path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    runs = tmp_path / "runs"
    write_run_record(
        "routing",
        config={"customer_count": 5},
        seed=4,
        results={"genetic algorithm": {"cost_minutes": 51.5}},
        runs_dir=runs,
    )
    matches = retrieve(
        "genetic algorithm route minutes",
        knowledge_dir=knowledge,
        runs_dir=runs,
        index_path=runs / "knowledge_index.json",
        min_score=0.05,
    )
    assert matches
    assert matches[0].chunk.unit == "Recorded experiments"
    assert "51.5 minutes" in matches[0].chunk.text


def test_saved_json_index_is_reused_until_markdown_changes(tmp_path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "glossary.md").write_text(
        "# Glossary\n\n## Tour\n\nA tour returns to the hub after visiting its stops.",
        encoding="utf-8",
    )
    index = tmp_path / "runs" / "knowledge_index.json"
    written = build_index(knowledge, index_path=index)
    assert written == index
    assert json.loads(index.read_text(encoding="utf-8"))["schema_version"] == 1
    chunks = load_chunks(knowledge, runs_dir=tmp_path / "runs", index_path=index)
    assert chunks[0].source == "glossary.md"
