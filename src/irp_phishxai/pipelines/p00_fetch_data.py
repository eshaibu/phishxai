import os
import logging
from ..config import load_config
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def main(cfg_path: str):
    """
    Stage p00: Validate required raw files exist.
    Fails fast with informative error if a file is missing.
    """
    setup_logging()
    cfg = load_config(cfg_path)
    tranco_fp = os.path.join(cfg["paths"]["raw"], cfg["data"]["tranco"]["filename"])
    phish_fp = os.path.join(cfg["paths"]["raw"], cfg["data"]["phishtank"]["filename"])

    for fp in [tranco_fp, phish_fp]:
        if not os.path.exists(fp):
            logger.error("Missing required raw file: %s", fp)
            raise FileNotFoundError(f"Missing raw file: {fp}")

    logger.info("[p00] Raw files present.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False, default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    main(args.config)
