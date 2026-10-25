"""Simple standard-library logging setup for the application."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise console logging once for the current process."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
