import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Default configuration used if keys are absent in the YAML file.
DEFAULTS = {
    "seed": 42,
    "paths": {
        "raw": "data/raw",
        "interim": "data/interim",
        "processed": "data/processed",
        "models": "models",
        "reports": "reports",
    },
    "data": {
        "tranco": {"filename": "truncated_10000_tranco_top_1m.csv", "sample_size": 10000},
        "phishtank": {
            "filename": "truncated_10000_phishtank_verified_online.csv",
            "verified_only": True,
            "online_only": True,
            "sample_size": 10000,
        },
        "window": {"start": None, "end": None},
    },
    "split": {"test_ratio": 0.2, "domain_disjoint": True},
    "features": {"whitelist": []},
    "models": {"train": ["rf", "xgb", "lgbm", "lr", "linear_svm", "dt"], "explain": ["rf", "xgb", "lgbm"]},
    "param_grids": {},
    "shap": {"sample_size": 2000},
    "lime": {"n_examples": 2},
}


def _deep_update(d: Dict, u: Dict) -> Dict:
    """
    Recursively update dict `d` with values from `u`.
    Notes:
      - Only merges nested dicts; other types overwrite.
    """
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path_or_dict: str | Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Load YAML config (or take a dict), merge with DEFAULTS, validate essential keys,
    and ensure required directories exist.
    Raises:
      - AssertionError on missing critical keys (e.g., features.whitelist)
      - File read exceptions if YAML path invalid
    """
    cfg = DEFAULTS.copy()
    try:
        # Accept dict, path or None.
        if isinstance(path_or_dict, dict):
            user = path_or_dict
        elif isinstance(path_or_dict, str):
            with open(path_or_dict, "r") as f:
                user = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", path_or_dict)
        else:
            user = {}

        # Merge user config into defaults.
        cfg = _deep_update(cfg, user)

        # Guarantee directory structure exists (idempotent).
        for key in ["raw", "interim", "processed", "models", "reports"]:
            os.makedirs(cfg["paths"][key], exist_ok=True)

        # Basic schema assertions for common failure points.
        assert "features" in cfg and "whitelist" in cfg["features"], "features.whitelist missing"
        assert "models" in cfg and "train" in cfg["models"], "models.train missing"
        assert "param_grids" in cfg, "param_grids missing"
        return cfg
    except Exception as e:
        logger.exception("Failed to load/validate config: %s", e)
        raise
