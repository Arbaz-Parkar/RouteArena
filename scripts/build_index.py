"""Build the ignored JSON snapshot used by RouteArena's offline knowledge search."""

from __future__ import annotations

from routearena.genai.rag.retrieval import build_index


def main() -> None:
    """Write the current Markdown chunks under the git-ignored runs directory."""
    destination = build_index()
    print(f"Built knowledge index with local Markdown notes: {destination}")


if __name__ == "__main__":
    main()
