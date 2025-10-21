import logging
import os

from ..config import load_config
from ..utils.feature_utils import build_feature_frame, select_feature_whitelist, cast_dtypes
from ..utils.io_utils import read_csv_safely, write_csv, DTYPE_MAP_FEATURES
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(cfg_path: str):
    """
    Stage p02: Transform aligned URLs into lexical features and save.
    Writes: data/interim/features_full.csv
    """
    setup_logging()
    cfg = load_config(cfg_path)
    inp = os.path.join(cfg["paths"]["interim"], "aligned_urls.csv")

    # Build lexical features (vectorized; pure URL-based).
    df = read_csv_safely(inp)
    feats = build_feature_frame(df, cfg)
    feats = cast_dtypes(feats)

    # Coerce to memory-safe dtypes and select whitelist.
    feats = feats.astype({k: v for k, v in DTYPE_MAP_FEATURES.items() if k in feats.columns})
    feats = select_feature_whitelist(feats, cfg["features"]["whitelist"])

    tmp = os.path.join(cfg["paths"]["interim"], "features_full.csv")
    write_csv(feats, tmp)
    logger.info("[p02] wrote features %s cols=%d rows=%d", tmp, len(feats.columns), len(feats))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    main(args.config)
