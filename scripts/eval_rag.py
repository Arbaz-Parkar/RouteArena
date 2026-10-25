"""Evaluate local retrieval against the small hand-written question set."""

from __future__ import annotations

import json
from pathlib import Path

from routearena.genai.rag.retrieval import build_knowledge_chunks, retrieve


def main() -> None:
    """Print retrieval hit rate and out-of-scope refusal rate."""
    question_path = Path("knowledge/eval_questions.json")
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    chunks = build_knowledge_chunks()
    answerable = [item for item in questions if item["expected_source"] is not None]
    out_of_scope = [item for item in questions if item["expected_source"] is None]

    hits = 0
    for item in answerable:
        matches = retrieve(item["question"], chunks=chunks, top_k=4, min_score=0.06)
        hits += any(match.chunk.source == item["expected_source"] for match in matches)

    refusals = 0
    for item in out_of_scope:
        matches = retrieve(item["question"], chunks=chunks, top_k=4, min_score=0.06)
        refusals += not bool(matches)

    print(f"Retrieval hit rate at k=4: {hits}/{len(answerable)} = {hits / len(answerable):.1%}")
    print(
        f"Out-of-scope refusal rate: {refusals}/{len(out_of_scope)} "
        f"= {refusals / len(out_of_scope):.1%}"
    )


if __name__ == "__main__":
    main()
