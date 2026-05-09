"""
TruthLens AI — Real-Time Cross-Source Validator
================================================
Simulates cross-source verification against a curated topic database.
In production: integrate with NewsAPI, GDELT, or Google News API.
"""

import re
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Curated fact database (simulates a real-time news index)
# Topics → [verified sources that covered it]
# ──────────────────────────────────────────────────────────────
VERIFIED_TOPIC_DB = {
    # Global events
    "covid vaccine": ["reuters.com", "bbc.com", "who.int", "ndtv.com"],
    "ukraine war": ["reuters.com", "apnews.com", "bbc.com", "theguardian.com"],
    "climate change": ["bbc.com", "theguardian.com", "reuters.com", "nature.com"],
    "election": ["reuters.com", "apnews.com", "ndtv.com", "thehindu.com"],
    "ai artificial intelligence": ["reuters.com", "bbc.com", "theguardian.com"],
    "earthquake": ["reuters.com", "apnews.com", "ndtv.com"],
    "flood": ["ndtv.com", "thehindu.com", "bbc.com"],
    # India-specific
    "modi": ["thehindu.com", "ndtv.com", "indiatoday.in", "scroll.in"],
    "parliament": ["thehindu.com", "ndtv.com", "livemint.com"],
    "india pakistan": ["thehindu.com", "ndtv.com", "dawn.com", "bbc.com"],
    "aadhaar": ["thehindu.com", "livemint.com", "scroll.in"],
    "upi payment": ["livemint.com", "thehindu.com", "ndtv.com"],
    "farmer protest": ["thehindu.com", "scroll.in", "thewire.in", "ndtv.com"],
    "rupee dollar": ["livemint.com", "reuters.com", "thehindu.com"],
    # Common misinformation topics
    "5g microchip": [],  # No credible sources → misinformation signal
    "flat earth": [],
    "chemtrail": [],
    "vaccine microchip": [],
    "george soros": ["apnews.com"],  # Often misrepresented
    "deep state": [],
}

TRUSTED_SOURCE_LIST = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "nytimes.com", "thehindu.com",
    "ndtv.com", "indiatoday.in", "scroll.in", "livemint.com",
    "boomlive.in", "altnews.in", "factcheck.org", "snopes.com",
    "who.int", "nature.com", "theprint.in",
]


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for topic matching."""
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "this", "that", "and", "or"}
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in stopwords]


def find_matching_topics(text: str) -> list[tuple[str, list[str]]]:
    """Find verified topic database matches for a given text."""
    keywords = extract_keywords(text)
    matches = []
    for topic, sources in VERIFIED_TOPIC_DB.items():
        topic_words = topic.split()
        if any(kw in keywords for kw in topic_words):
            matches.append((topic, sources))
    return matches


def verify_content(text: str) -> dict:
    """
    Cross-source verification of content.
    Returns verification status and source information.
    """
    matches = find_matching_topics(text)

    if not matches:
        return {
            "status": "unverified",
            "message": "⚠️ No matching topics found in verified source database",
            "source_count": 0,
            "sources": [],
            "confidence": 0.3,
        }

    # Collect all sources from matching topics
    all_sources = []
    all_topics = []
    for topic, sources in matches:
        all_sources.extend(sources)
        all_topics.append(topic)

    unique_sources = list(set(all_sources))
    trusted_count = sum(1 for s in unique_sources if s in TRUSTED_SOURCE_LIST)

    if not unique_sources:
        return {
            "status": "disputed",
            "message": "🔴 Topic found but no credible sources have reported on this claim",
            "source_count": 0,
            "sources": [],
            "topics_matched": all_topics,
            "confidence": 0.1,
        }
    elif trusted_count >= 3:
        return {
            "status": "verified",
            "message": f"✅ Verified by {trusted_count} trusted sources",
            "source_count": trusted_count,
            "sources": unique_sources[:5],
            "topics_matched": all_topics,
            "confidence": min(0.5 + trusted_count * 0.1, 0.95),
        }
    elif trusted_count >= 1:
        return {
            "status": "partial",
            "message": f"🟡 Partially confirmed — found in {trusted_count} trusted source(s)",
            "source_count": trusted_count,
            "sources": unique_sources[:5],
            "topics_matched": all_topics,
            "confidence": 0.4 + trusted_count * 0.1,
        }
    else:
        return {
            "status": "unconfirmed",
            "message": "⚠️ No reliable confirmation found from trusted outlets",
            "source_count": 0,
            "sources": [],
            "topics_matched": all_topics,
            "confidence": 0.2,
        }
