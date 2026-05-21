import logging
import sys


def setup_logger(name: str = "signova") -> logging.Logger:
    """Creates and returns a configured logger with console output."""
    _logger: logging.Logger = logging.getLogger(name)
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        fmt: logging.Formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        _logger.addHandler(ch)
    return _logger


logger: logging.Logger = setup_logger()
