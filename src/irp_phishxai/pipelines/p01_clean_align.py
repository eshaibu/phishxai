import os
import pandas as pd
from dateutil import parser as dtparse
import logging
from ..config import load_config
from ..utils.logging_utils import setup_logging
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.feature_utils import _row_url_parts

logger = logging.getLogger(__name__)

def _normalize_phishtank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize PhishTank feed to unified schema: url, label=1, ts, source='phishtank'.
    Prefers verification_time; falls back to submission_time if missing.
    """
    def _pick_ts(row) -> pd.Timestamp:
        ts = row.get("verification_time") or row.get("submission_time")
        try:
            return dtparse.parse(ts) if pd.notna(ts) else pd.NaT
        except Exception:
            return pd.NaT

    out = pd.DataFrame({
        "url": df["url"],
        "label": 1,
        "ts": df.apply(_pick_ts, axis=1),
        "source": "phishtank"
    })
    return out

def _normalize_tranco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Tranco list to unified schema: url, label=0, ts (if any), source='tranco'.
    If list date exists, assign it to all rows (optional).
    """
    # second column is the domain column
    domain_col = df.columns[1]
    df = df.rename(columns={domain_col: "domain"})
    url = "https://" + df["domain"].astype(str).str.strip() + "/"
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce")
    elif "list_date" in df.columns and len(df) > 0:
        ts = pd.to_datetime(df["list_date"].iloc[0], errors="coerce")
        ts = pd.Series([ts] * len(df))
    else:
        ts = pd.Series([pd.NaT] * len(df))
    out = pd.DataFrame({"url": url, "label": 0, "ts": ts, "source": "tranco"})
    return out

def _clean_parse_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add host and eTLD+1, drop malformed and exact URL duplicates.
    """
    parts = df["url"].apply(_row_url_parts).apply(pd.Series)[["host","etld1"]]
    out = pd.concat([df.reset_index(drop=True), parts.reset_index(drop=True)], axis=1)
    out = out[out["host"] != ""].copy()  # remove malformed/hostless URLs
    out = out.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return out

def main(cfg_path: str):
    """
    Stage p01: Read raw CSVs, normalize schema, deduplicate, remove cross-domain overlaps, apply time window.
    Writes: data/interim/aligned_urls.csv
    """
    setup_logging()
    cfg = load_config(cfg_path)

    # Load raw CSVs with light dtype hints.
    tr_fp = os.path.join(cfg["paths"]["raw"], cfg["data"]["tranco"]["filename"])
    ph_fp = os.path.join(cfg["paths"]["raw"], cfg["data"]["phishtank"]["filename"])
    tranco = read_csv_safely(tr_fp)
    phish = read_csv_safely(ph_fp)

    # Optional sampling to constrain runtime (starter config).
    tsz = cfg["data"]["tranco"].get("sample_size")
    psz = cfg["data"]["phishtank"].get("sample_size")
    if tsz: tranco = tranco.head(tsz).copy()
    if psz: phish = phish.head(psz).copy()

    # Normalize raw frames to a shared minimal schema.
    phish_norm = _normalize_phishtank(phish)
    tranco_norm = _normalize_tranco(tranco)

    # Parse/clean/merge: add host/etld1; remove malformed & duplicates.
    phish_clean = _clean_parse_and_dedup(phish_norm)
    tranco_clean = _clean_parse_and_dedup(tranco_norm)

    # Remove benign rows whose eTLD+1 appears among phishing domains.
    phish_domains = set(phish_clean["etld1"].dropna().tolist())
    before = len(tranco_clean)
    tranco_clean = tranco_clean[~tranco_clean["etld1"].isin(phish_domains)].reset_index(drop=True)
    logger.info("Removed %d benign rows overlapping with phish domains", before - len(tranco_clean))

    # Apply optional date window if provided.
    start = cfg["data"]["window"].get("start")
    end = cfg["data"]["window"].get("end")
    if start:
        start_ts = pd.to_datetime(start)
        tranco_clean = tranco_clean[(tranco_clean["ts"].isna()) | (tranco_clean["ts"] >= start_ts)]
        phish_clean = phish_clean[(phish_clean["ts"].isna()) | (phish_clean["ts"] >= start_ts)]
    if end:
        end_ts = pd.to_datetime(end)
        tranco_clean = tranco_clean[(tranco_clean["ts"].isna()) | (tranco_clean["ts"] <= end_ts)]
        phish_clean = phish_clean[(phish_clean["ts"].isna()) | (phish_clean["ts"] <= end_ts)]

    # Concatenate to aligned, labeled frame.
    aligned = pd.concat([tranco_clean, phish_clean], ignore_index=True)
    out_fp = os.path.join(cfg["paths"]["interim"], "aligned_urls.csv")
    write_csv(aligned, out_fp)
    logger.info("[p01] wrote %s rows=%d", out_fp, len(aligned))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    main(args.config)
