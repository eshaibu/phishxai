import pandas as pd
from typing import Dict, List
import logging
from .url_utils import safe_parse_url, extract_etld1, has_ip_host, is_url_shortener, char_count_features

logger = logging.getLogger(__name__)

def _row_url_parts(url: str) -> Dict:
    """
    Parse URL into structured parts and simple metrics used as features:
      - host, path, query, etld1
      - host_len, path_len, query_len
      - subdomain_depth, has_ip, is_shortener
    Returns zeros/empties on parse failure (keeps row instead of dropping immediately).
    """
    p = safe_parse_url(url)
    if not p:
        return {"host": "", "path": "", "query": "", "etld1": "", "host_len": 0, "path_len": 0, "query_len": 0,
                "subdomain_depth": 0, "has_ip": 0, "is_shortener": 0}
    host = p["host"]
    etld1 = extract_etld1(host)
    # subdomain depth = host label count difference from etld1
    sub_depth = max(0, host.count(".") - etld1.count(".")) if etld1 else 0
    return {
        "host": host,
        "path": p["path"],
        "query": p["query"],
        "etld1": etld1,
        "host_len": len(host),
        "path_len": len(p["path"]),
        "query_len": len(p["query"]),
        "subdomain_depth": sub_depth,
        "has_ip": int(has_ip_host(host)),
        "is_shortener": int(is_url_shortener(host)),
    }

def build_feature_frame(df_urls: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Convert aligned URL rows into a numeric/boolean feature DataFrame.
    Input columns required: 'url', 'label'.
    Steps:
      1) Vectorize URL parsing to structured parts.
      2) Compute lexical char-count features.
      3) Concatenate with label and cast numeric columns compactly.
    """
    try:
        parts = df_urls["url"].apply(_row_url_parts).apply(pd.Series)
        chars = df_urls["url"].apply(char_count_features).apply(pd.Series)
        out = pd.concat([parts, chars, df_urls[["label"]]], axis=1)

        # Cast numerics to float32 for efficiency; label to int32.
        num_cols = [c for c in out.columns if c not in ["host","path","query","etld1","label"]]
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        out["label"] = out["label"].astype("int32")
        logger.info("Feature frame built (rows=%d, cols=%d)", len(out), out.shape[1])
        return out
    except KeyError as e:
        logger.exception("Input frame missing required column: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to build feature frame: %s", e)
        raise

def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast Python bools to uint8 (smaller on disk/memory).
    """
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype("uint8")
    return df

def select_feature_whitelist(df: pd.DataFrame, whitelist: List[str]) -> pd.DataFrame:
    """
    Keep only the configured feature whitelist + 'label'.
    Raises:
      - ValueError if 'label' is missing (corrupted pipeline).
    """
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in features")
    cols = [c for c in whitelist if c in df.columns]
    sel = df[cols + ["label"]].copy()
    logger.info("Selected %d whitelist features (+ label)", len(cols))
    return sel
