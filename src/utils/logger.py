"""Logger factory helpers for consistent application logging."""

import logging
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create/configure a logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
