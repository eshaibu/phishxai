import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_exact_url_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate URLs exactly (string match).
    """
    out = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    logger.info("Dropped exact URL duplicates: %d -> %d", len(df), len(out))
    return out


def drop_domain_cross_overlap(benign_df: pd.DataFrame, phish_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove benign rows if their eTLD+1 also appears in the phishing set.
    Prevents leakage where a domain exists in both sources with conflicting labels.
    """
    phish_domains = set(phish_df["etld1"].dropna().tolist())
    mask = ~benign_df["etld1"].isin(phish_domains)
    out_b = benign_df.loc[mask].reset_index(drop=True)
    logger.info("Removed %d overlapping benign rows due to phish-domain overlap", len(benign_df) - len(out_b))
    return out_b, phish_df.reset_index(drop=True)
