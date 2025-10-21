import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging to stdout.
    Behavior:
      - Idempotent: if configured once, subsequent calls do nothing.
      - Format shows time, level, logger name, and the message.
    """
    if logging.getLogger().handlers:
        return  # Already configured globally.

    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
