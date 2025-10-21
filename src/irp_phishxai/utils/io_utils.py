import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Global dtype suggestions for feature CSVs to keep memory usage modest.
DTYPE_MAP_FEATURES: Dict[str, Any] = {
    "url_len": "int32",
    "host_len": "int32",
    "path_len": "int32",
    "query_len": "int32",
    "n_digits": "int32",
    "n_hyphens": "int32",
    "n_special": "int32",
    "subdomain_depth": "int32",
    "has_ip": "uint8",
    "has_at": "uint8",
    "pct_encoded": "uint8",
    "double_slash_in_path": "uint8",
    "shannon_entropy_url": "float32",
    "is_shortener": "uint8",
    "suspicious_token_present": "uint8",
    "label": "int32",
}


def ensure_dirs(cfg):
    """Create all directories defined under cfg['paths'] if they don't exist."""
    for key, path in cfg.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)


def read_csv_safely(path: str, dtype_map: Optional[Dict] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame with optional dtype hints and chunking.
    Use cases:
      - Large files: set chunksize to stream + concatenate.
      - Type stability: pass dtype_map to avoid object columns.
    """
    try:
        if chunksize:
            logger.info("Reading CSV in chunks: %s", path)
            chunks = pd.read_csv(path, dtype=dtype_map, chunksize=chunksize)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, dtype=dtype_map)
        logger.info("Read CSV: %s (rows=%d, cols=%d)", path, len(df), df.shape[1])
        return df
    except FileNotFoundError:
        logger.exception("CSV not found: %s", path)
        raise
    except Exception as e:
        logger.exception("Failed to read CSV: %s (%s)", path, e)
        raise


def write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Write a DataFrame to CSV with directory creation and logging.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Wrote CSV: %s (rows=%d, cols=%d)", path, len(df), df.shape[1])
    except Exception as e:
        logger.exception("Failed to write CSV: %s (%s)", path, e)
        raise


def save_yaml(obj: Dict, path: str) -> None:
    """
    YAML dump with safe defaults and folder auto-creation.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
        logger.info("Wrote YAML: %s", path)
    except Exception as e:
        logger.exception("Failed to write YAML: %s (%s)", path, e)
        raise


def save_json(obj: Dict, path: str) -> None:
    """
    JSON dump with indent and folder auto-creation.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        logger.info("Wrote JSON: %s", path)
    except Exception as e:
        logger.exception("Failed to write JSON: %s (%s)", path, e)
        raise


def timestamped_run_dir(base: str = "experiments/runs") -> str:
    """
    Create and return a timestamped run directory under `base`.
    Example: experiments/runs/2025-10-19_15-21-03
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base, ts)
    os.makedirs(out, exist_ok=True)
    logger.info("Created run dir: %s", out)
    return out
