from typing import Dict
from urllib.parse import urlparse, unquote
import re
import math
import tldextract
import logging

logger = logging.getLogger(__name__)

# Popular shortener hosts to flag (heuristic).
SHORTENERS = {"bit.ly","t.co","goo.gl","tinyurl.com","ow.ly","is.gd","buff.ly","rebrand.ly"}

def safe_parse_url(url: str) -> Dict:
    """
    Safely parse a URL into components {scheme, host, path, query}.
    Returns empty dict on malformed input instead of raising.
    """
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return {}
        return {"scheme": p.scheme, "host": p.netloc, "path": p.path or "", "query": p.query or ""}
    except Exception as e:
        logger.debug("URL parse failed: %s (%s)", url, e)
        return {}

def extract_etld1(host: str) -> str:
    """
    Extract effective TLD+1 using tldextract.
    Returns empty string if extraction fails or host is atypical.
    """
    try:
        ext = tldextract.extract(host)
        if not ext.domain or not ext.suffix:
            return ""
        return f"{ext.domain}.{ext.suffix}"
    except Exception as e:
        logger.debug("eTLD+1 extraction failed: %s (%s)", host, e)
        return ""

def shannon_entropy(s: str) -> float:
    """
    Compute Shannon entropy over characters in string s.
    Lower = more regular; higher = more random (often suspicious).
    """
    if not s:
        return 0.0
    freqs = {ch: s.count(ch)/len(s) for ch in set(s)}
    return -sum(p * math.log2(p) for p in freqs.values())

def has_ip_host(host: str) -> bool:
    """
    Check if host looks like an IPv4 address (basic regex).
    """
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

def is_url_shortener(host: str) -> bool:
    """
    Check if host is in a small whitelist of common URL shorteners.
    """
    h = host.lower()
    return h in SHORTENERS or any(h.endswith("."+s) for s in SHORTENERS)

def char_count_features(url: str) -> Dict:
    """
    Generate lexical features from the raw URL string (counts + flags).
    Notes:
      - Use unquoted URL for character counts.
      - Keep logic simple/fast; all numeric/boolean outputs.
    """
    u = unquote(url or "")
    try:
        path = urlparse(url).path
    except Exception:
        path = ""
    return {
        "url_len": len(u),
        "n_digits": sum(ch.isdigit() for ch in u),
        "n_hyphens": u.count("-"),
        "n_special": sum(ch in "@%$&*+!?" for ch in u),
        "has_at": int("@" in u),
        "pct_encoded": int("%" in (url or "")),
        "double_slash_in_path": int("//" in (path or "")),
        "shannon_entropy_url": shannon_entropy(u),
        "suspicious_token_present": int(any(tok in u.lower() for tok in ["login","verify","update","secure","account","bank"])),
    }
