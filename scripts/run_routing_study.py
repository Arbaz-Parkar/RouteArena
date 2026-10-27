"""Run and save the fixed-seed TSPLIB routing study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routearena.logistics.routing.benchmark import DEFAULT_OUTPUT_DIR, run_burma14_study


def main() -> None:
    """Run the study and print its summary and artifact location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the CSV and JSON study artifacts",
    )
    args = parser.parse_args()
    report = run_burma14_study(output_dir=args.output_dir)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Saved the study artifacts under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
