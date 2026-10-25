"""Heading-based chunking and dependency-free TF-IDF retrieval."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from routearena.records import RUNS_DIR, read_run_records

LOGGER = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_DIR = Path("knowledge")
CHUNK_SIZE = 1_400
CHUNK_OVERLAP = 160
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.06
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "of",
    "on",
    "or",
    "our",
    "out",
    "please",
    "routearena",
    "should",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "under",
    "up",
    "use",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}
_UNITS_BY_FILE = {
    "optimizers.md": "Units 1 and 2: optimization methods",
    "hub-placement.md": "Unit 1: continuous optimization",
    "routing.md": "Unit 4: discrete optimization",
    "uncertainty.md": "Unit 4: uncertainty analysis",
    "surrogates.md": "Unit 3: sampling and surrogate optimization",
    "ml-demo.md": "RouteArena machine-learning demonstration",
    "assumptions-faq.md": "Project assumptions and FAQ",
    "glossary.md": "Project glossary",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    """One short passage with citation metadata."""

    source: str
    section: str
    unit: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage and its cosine similarity to the current question."""

    chunk: KnowledgeChunk
    score: float


def _split_text(text: str, *, size: int, overlap: int) -> list[str]:
    """Split a long section into bounded pieces with a small text overlap."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    content = text.strip()
    if not content:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + size, len(content))
        if end < len(content):
            boundary = content.rfind("\n", start + size // 2, end)
            if boundary == -1:
                boundary = content.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = content[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(content):
            break
        start = max(start + 1, end - overlap)
    return pieces


def build_knowledge_chunks(
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[KnowledgeChunk]:
    """Read Markdown notes, split at headings, and attach unit/source metadata."""
    directory = Path(knowledge_dir)
    chunks: list[KnowledgeChunk] = []
    if not directory.exists():
        return chunks
    for path in sorted(directory.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        unit = _UNITS_BY_FILE.get(path.name, "Project knowledge")
        current_heading = path.stem.replace("-", " ").title()
        section_lines: list[str] = []

        def flush_section() -> None:
            body = "\n".join(section_lines).strip()
            if not body:
                return
            section_text = f"{current_heading}\n{body}"
            pieces = _split_text(section_text, size=chunk_size, overlap=overlap)
            for piece_index, piece in enumerate(pieces, start=1):
                section = current_heading
                if len(pieces) > 1:
                    section = f"{current_heading} (part {piece_index})"
                chunks.append(KnowledgeChunk(path.name, section, unit, piece))

        for line in lines:
            if line.startswith("#"):
                if section_lines or current_heading != path.stem.replace("-", " ").title():
                    flush_section()
                current_heading = line.lstrip("#").strip() or current_heading
                section_lines = []
            else:
                section_lines.append(line)
        if section_lines or not chunks:
            flush_section()
    return chunks


def _run_summary_chunks(runs_dir: Path | str = RUNS_DIR) -> list[KnowledgeChunk]:
    """Turn saved experiment metadata into short passages available to retrieval."""
    chunks: list[KnowledgeChunk] = []
    for record in read_run_records(runs_dir=runs_dir, limit=None):
        run_type = record.get("run_type", "experiment")
        config = record.get("config", {})
        results = record.get("results", {})
        created = str(record.get("created_at", "unknown time"))
        filename = str(record.get("_filename", record.get("run_id", "saved-run.json")))
        if run_type == "routing":
            summaries = []
            for algorithm, value in results.items():
                if isinstance(value, dict) and "cost_minutes" in value:
                    summaries.append(f"{algorithm}: {float(value['cost_minutes']):.1f} minutes")
            detail = "; ".join(summaries) if summaries else "route results are saved"
            text = (
                f"Saved routing run from {created} used "
                f"{config.get('customer_count', 'an unknown number of')} customer stops. "
                f"{detail}."
            )
        elif run_type == "optimizer-arena":
            summaries = [
                f"{name}: objective {float(value['objective']):.4g}"
                for name, value in results.items()
                if isinstance(value, dict) and "objective" in value
            ]
            text = (
                f"Saved optimizer arena run from {created} on the "
                f"{config.get('landscape', 'unknown')} landscape. "
                f"{'; '.join(summaries) if summaries else 'Optimizer results are saved.'}"
            )
        elif run_type == "hub-placement":
            summaries = [
                f"{name}: weighted-distance objective {float(value['objective']):.4g}"
                for name, value in results.items()
                if isinstance(value, dict) and "objective" in value
            ]
            text = (
                f"Saved hub placement run from {created} used "
                f"{config.get('customer_count', 'an unknown number of')} synthetic customers. "
                f"{'; '.join(summaries) if summaries else 'Hub results are saved.'}"
            )
        elif run_type == "risk":
            traffic = config.get("traffic_level", "unknown")
            deadline = config.get("deadline_minutes", "unknown")
            probability = results.get("selected_probability_late")
            risk_text = (
                f"The selected route had a {float(probability):.1%} simulated chance of missing "
                "the deadline."
                if probability is not None
                else "Risk results are saved."
            )
            text = (
                f"Saved risk run from {created} used the {traffic} traffic assumption and a "
                f"{deadline} minute deadline. {risk_text}"
            )
        else:
            text = f"Saved {run_type} run from {created}. Configuration: {config}."
        chunks.append(
            KnowledgeChunk(
                source=filename,
                section="Saved run summary",
                unit="Recorded experiments",
                text=text,
            )
        )
    return chunks


def _read_saved_index(index_path: Path, knowledge_dir: Path) -> list[KnowledgeChunk] | None:
    """Load a JSON index only when it is at least as new as all Markdown files."""
    if not index_path.exists():
        return None
    paths = list(knowledge_dir.glob("*.md")) if knowledge_dir.exists() else []
    newest_source = max((path.stat().st_mtime for path in paths), default=0.0)
    if index_path.stat().st_mtime < newest_source:
        return None
    try:
        content = json.loads(index_path.read_text(encoding="utf-8"))
        return [KnowledgeChunk(**item) for item in content["chunks"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        LOGGER.warning("Ignoring invalid knowledge index at %s", index_path)
        return None


def load_chunks(
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    *,
    runs_dir: Path | str = RUNS_DIR,
    index_path: Path | str | None = None,
) -> list[KnowledgeChunk]:
    """Load current knowledge notes and add summaries of recent saved runs."""
    knowledge_path = Path(knowledge_dir)
    index = Path(index_path) if index_path is not None else Path(runs_dir) / "knowledge_index.json"
    chunks = _read_saved_index(index, knowledge_path)
    if chunks is None:
        chunks = build_knowledge_chunks(knowledge_path)
    return [*chunks, *_run_summary_chunks(runs_dir)]


def build_index(
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    *,
    index_path: Path | str | None = None,
) -> Path:
    """Persist a JSON snapshot of Markdown chunks under the ignored runs directory."""
    destination = Path(index_path) if index_path is not None else RUNS_DIR / "knowledge_index.json"
    chunks = build_knowledge_chunks(knowledge_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "chunks": [asdict(chunk) for chunk in chunks]}
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def _tokens(text: str) -> list[str]:
    """Normalize text into simple terms and remove common words."""
    return [token for token in _WORD_PATTERN.findall(text.lower()) if token not in _STOP_WORDS]


def _weighted_vector(tokens: list[str], inverse_document_frequency: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    return {
        term: (1.0 + math.log(count)) * inverse_document_frequency[term]
        for term, count in counts.items()
        if term in inverse_document_frequency
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity for two sparse word-weight maps."""
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight**2 for weight in left.values()))
    right_norm = math.sqrt(sum(weight**2 for weight in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
    runs_dir: Path | str = RUNS_DIR,
    index_path: Path | str | None = None,
    chunks: list[KnowledgeChunk] | None = None,
) -> list[RetrievedChunk]:
    """Return the most relevant local passages above the similarity threshold."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if chunks is None:
        corpus = load_chunks(knowledge_dir, runs_dir=runs_dir, index_path=index_path)
    else:
        corpus = chunks
    query_tokens = _tokens(query)
    documents = [_tokens(chunk.text) for chunk in corpus]
    document_frequency = Counter(term for terms in documents for term in set(terms))
    count = len(documents)
    idf = {
        term: math.log((count + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    query_vector = _weighted_vector(query_tokens, idf)
    ranked = [
        RetrievedChunk(chunk, _cosine(query_vector, _weighted_vector(tokens, idf)))
        for chunk, tokens in zip(corpus, documents, strict=True)
    ]
    ranked.sort(key=lambda item: (-item.score, item.chunk.source, item.chunk.section))
    return [item for item in ranked if item.score >= min_score][:top_k]


def answer_from_knowledge(query: str, matches: list[RetrievedChunk]) -> str:
    """Create a short extractive answer with local citations, or refuse to guess."""
    if not matches:
        return "I could not find that in my knowledge base."
    sentences: list[str] = []
    sources: list[str] = []
    for match in matches[:2]:
        text = re.sub(r"^#+\s*", "", match.chunk.text.strip(), flags=re.MULTILINE)
        candidates = [sentence.strip(" -*\n\t") for sentence in _SENTENCE_PATTERN.split(text)]
        for sentence in candidates:
            if sentence and sentence not in sentences:
                sentences.append(sentence)
            if len(sentences) >= 3:
                break
        citation = f"[{match.chunk.source}: {match.chunk.section}]"
        if citation not in sources:
            sources.append(citation)
    excerpt = " ".join(sentences)
    if len(excerpt) > 850:
        excerpt = excerpt[:847].rsplit(" ", 1)[0] + "..."
    return f"{excerpt}\n\nSources: {', '.join(sources)}"
