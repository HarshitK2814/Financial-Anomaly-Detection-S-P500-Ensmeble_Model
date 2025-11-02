import logging
import os
import sys
from datetime import datetime

def setup_logging(config=None):
    """
    Robust setup for logging.
    config may be None or a dict with keys:
      - level: "INFO","DEBUG",...
      - to_file: bool
      - log_dir: path where to write logs (if to_file True)
      - filename: optional explicit filename
    """
    # default config
    if config is None:
        config = {}
    if not isinstance(config, dict):
        # fallback: if caller passed a string, treat as filename
        filename = str(config)
        config = {"to_file": True, "filename": filename}

    level_name = config.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    to_file = bool(config.get("to_file", False))

    if to_file:
        log_dir = config.get("log_dir", "artifacts/logs")
        os.makedirs(log_dir, exist_ok=True)
        filename = config.get("filename")
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(log_dir, f"run-{timestamp}.log")
        logging.basicConfig(
            level=level,
            filename=filename,
            filemode="a",
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        # stream to stdout (avoid passing dict to basicConfig)
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
        logging.getLogger().setLevel(level)

    # return root logger for convenience
    return logging.getLogger()

def log_metrics(metrics, epoch):
    """Log training and evaluation metrics."""
    logging.info(f'Epoch: {epoch}, Metrics: {metrics}')

def log_message(message):
    """Log a custom message."""
    logging.info(message)