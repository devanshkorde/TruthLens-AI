"""
TruthLens AI — Source Credibility Engine
==========================================
Scores any URL/domain on a 0–100 scale based on:
  - Known domain tier (trusted / fake / satire)
  - HTTPS usage
  - Domain age heuristics (mocked via domain characteristics)
  - TLD quality signals
  - Subdomain analysis
"""

import json
import re
import tldextract
from pathlib import Path
from dataclasses import dataclass, field

SOURCES_PATH = Path(__file__).parent.parent / "data" / "trusted_sources.json"

# Load once at import time
def _load_sources() -> dict[str, dict]:
    if not SOURCES_PATH.exists():
        return {}
    with open(SOURCES_PATH) as f:
        data = json.load(f)
    return {entry["domain"]: entry for entry in data}

SOURCE_DB = _load_sources()

# TLD quality tiers (mock domain age proxy)
PREMIUM_TLDS = {".com", ".org", ".gov", ".edu", ".net", ".in", ".co.in", ".co.uk"}
SUSPICIOUS_TLDS = {".xyz", ".info", ".biz", ".click", ".online", ".site", ".top", ".loan"}

VERIFICATION_SOURCES = {
    "reuters.com": ["apnews.com", "bbc.com", "theguardian.com"],
    "bbc.com": ["reuters.com", "apnews.com", "thehindu.com"],
    "ndtv.com": ["thehindu.com", "indiatoday.in", "scroll.in"],
    "thehindu.com": ["ndtv.com", "livemint.com", "reuters.com"],
    "boomlive.in": ["altnews.in", "factcheck.org", "snopes.com"],
}


@dataclass
class CredibilityResult:
    domain: str
    score: int                          # 0–100
    tier: str                           # trusted / unknown / fake / satire
    flags: list[str] = field(default_factory=list)
    is_https: bool = True
    verification_sources: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)


def score_domain(url: str) -> CredibilityResult:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    # Extract domain parts
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    full_domain = f"{extracted.subdomain}.{domain}" if extracted.subdomain else domain
    tld = f".{extracted.suffix}"

    is_https = url.startswith("https://")
    flags = []
    signals = []
    score = 50  # Start neutral

    # ─── Known domain lookup ───────────────────────
    lookup_key = domain.lower()
    known = SOURCE_DB.get(lookup_key)

    if known:
        base_score = known["score"]
        tier = known["tier"]
    else:
        base_score = 40
        tier = "unknown"
        flags.append("Unverified domain — not in TruthLens source database")

    score = base_score

    # ─── HTTPS bonus/penalty ──────────────────────
    if is_https:
        score = min(score + 5, 100)
        signals.append("🟢 HTTPS secured — encrypted connection")
    else:
        score = max(score - 20, 0)
        flags.append("HTTP only — no SSL/TLS encryption")
        signals.append("🔴 HTTP connection — site lacks basic security")

    # ─── TLD scoring ─────────────────────────────
    if tld in SUSPICIOUS_TLDS:
        score = max(score - 15, 0)
        flags.append(f"Suspicious TLD ({tld}) — often used by spam/fake sites")
        signals.append(f"🔴 Low-quality TLD ({tld}) — commonly associated with misinformation")
    elif tld in PREMIUM_TLDS:
        signals.append(f"🟢 Reputable TLD ({tld})")

    # ─── Subdomain analysis ───────────────────────
    if extracted.subdomain and extracted.subdomain not in ("www", "m", "mobile", "news", ""):
        score = max(score - 5, 0)
        flags.append(f"Non-standard subdomain ({extracted.subdomain}) — verify carefully")

    # ─── Tier-specific signals ────────────────────
    if tier == "trusted":
        signals.append("🟢 Domain is in TruthLens trusted-source database")
    elif tier == "fact-checker":
        signals.append("🟢 Independent fact-checking organization")
    elif tier == "fake":
        signals.append("🔴 Domain flagged as known misinformation source")
        flags.append("Known misinformation publisher")
    elif tier == "satire":
        signals.append("🟡 Satire website — content is intentionally fictional")
        flags.append("Satire / parody content — not factual reporting")
    elif tier == "unknown":
        signals.append("🟡 Domain not verified — exercise caution")

    # ─── Verification sources ────────────────────
    cross_sources = VERIFICATION_SOURCES.get(lookup_key, [])

    # Cap score
    score = max(0, min(100, score))

    return CredibilityResult(
        domain=full_domain,
        score=score,
        tier=tier,
        flags=flags,
        is_https=is_https,
        verification_sources=cross_sources,
        signals=signals,
    )
