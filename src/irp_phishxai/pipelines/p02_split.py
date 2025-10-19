import os
import logging
from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv, DTYPE_MAP_FEATURES
from ..utils.split_utils import train_test_time_domain_split
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def main(cfg_path: str):
    """
    Stage p02_split: Split features into train/test ensuring domain disjointness.
    Writes: data/processed/train_features.csv and test_features.csv
    """
    setup_logging()
    cfg = load_config(cfg_path)

    # Load features with dtype hints.
    feats_fp = os.path.join(cfg["paths"]["interim"], "features_full.csv")
    df = read_csv_safely(feats_fp, dtype_map=DTYPE_MAP_FEATURES)

    # Merge in URL + etld1 + ts from aligned to enforce domain/time constraints.
    aligned = read_csv_safely(os.path.join(cfg["paths"]["interim"], "aligned_urls.csv"))[["url", "etld1", "ts"]]
    feats_with_url = df.copy()
    feats_with_url["url"] = read_csv_safely(os.path.join(cfg["paths"]["interim"], "aligned_urls.csv"))["url"]
    merged = feats_with_url.merge(aligned, on="url", how="left")

    # Split and then drop text columns before saving.
    train_df, test_df = train_test_time_domain_split(merged, cfg["split"]["test_ratio"], cfg["seed"])
    drop_cols = ["url", "etld1", "ts"]
    train_out = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_out = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Ensure final dtypes are compact and consistent.
    train_out = train_out.astype({k: v for k, v in DTYPE_MAP_FEATURES.items() if k in train_out.columns})
    test_out = test_out.astype({k: v for k, v in DTYPE_MAP_FEATURES.items() if k in test_out.columns})

    write_csv(train_out, os.path.join(cfg["paths"]["processed"], "train_features.csv"))
    write_csv(test_out, os.path.join(cfg["paths"]["processed"], "test_features.csv"))
    logger.info("[p02_split] wrote processed train/test CSVs.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    main(args.config)
