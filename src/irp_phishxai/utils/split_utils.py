import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_test_time_domain_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a domain-disjoint split (no eTLD+1 overlap) and sort test chronologically if 'ts' exists.
    Steps:
      - Get distinct eTLD+1 strings.
      - Randomly assign domains to train/test using the requested ratio.
      - Filter rows by membership.
      - If a timestamp column exists in test, sort ascending by time to simulate recency.
    """
    if "etld1" not in df.columns:
        raise ValueError("etld1 column required before splitting")

    domains = df["etld1"].dropna().unique()
    d_train, d_test = train_test_split(domains, test_size=test_ratio, random_state=seed, shuffle=True)
    train_df = df[df["etld1"].isin(d_train)].copy()
    test_df = df[df["etld1"].isin(d_test)].copy()

    if "ts" in test_df.columns and pd.api.types.is_datetime64_any_dtype(test_df["ts"]):
        test_df = test_df.sort_values("ts", ascending=True).reset_index(drop=True)

    logger.info("Split -> train: %d rows, test: %d rows (domains train=%d, test=%d)",
                len(train_df), len(test_df), len(d_train), len(d_test))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
