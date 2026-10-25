"""Local JSON records for reproducible RouteArena experiments."""

from __future__ import annotations

import json
import logging
import platform
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
RUNS_DIR = Path("runs")
_VERSION_PACKAGES = ("numpy", "plotly", "streamlit", "pydantic")


def _jsonable(value: Any) -> Any:
    """Convert common Python and NumPy values to JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot store {type(value).__name__} in a run record")


def _library_versions() -> dict[str, str]:
    """Read installed versions without assuming optional packages are present."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in _VERSION_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def write_run_record(
    run_type: str,
    *,
    config: dict[str, Any],
    seed: int | None,
    results: dict[str, Any],
    runs_dir: Path | str = RUNS_DIR,
) -> Path:
    """Write one timestamped JSON record atomically and return its path."""
    safe_type = "".join(char for char in run_type.lower() if char.isalnum() or char in "-_")
    if not safe_type:
        raise ValueError("run_type must contain a letter or number")
    created_at = datetime.now(UTC)
    record_id = uuid.uuid4().hex
    record = {
        "schema_version": 1,
        "run_id": record_id,
        "created_at": created_at.isoformat(),
        "run_type": safe_type,
        "config": _jsonable(config),
        "seed": _jsonable(seed),
        "results": _jsonable(results),
        "library_versions": _library_versions(),
    }
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{safe_type}_{record_id[:8]}.json"
    destination = directory / filename
    temporary = destination.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        LOGGER.exception("Could not save RouteArena run record to %s", destination)
        raise
    return destination


def read_run_records(
    *, runs_dir: Path | str = RUNS_DIR, limit: int | None = 20
) -> list[dict[str, Any]]:
    """Load newest valid run records; malformed files are logged and skipped."""
    directory = Path(runs_dir)
    if not directory.exists():
        return []
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Skipping unreadable run record: %s", path)
            continue
        if not isinstance(value, dict) or "run_type" not in value or "results" not in value:
            continue
        value["_filename"] = path.name
        records.append(value)
        if limit is not None and len(records) >= limit:
            break
    return records
