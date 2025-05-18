import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger with console output and timestamp."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    # Optionally add file handler or other handlers
    logger = logging.getLogger()
    logger.info("Logging configured. Level: %s", logging.getLevelName(level))
