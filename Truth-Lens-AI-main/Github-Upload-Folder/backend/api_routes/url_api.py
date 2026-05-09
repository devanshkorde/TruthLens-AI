"""
TruthLens AI — URL Analysis API Route
POST /analyze-url
"""

import time
import re
from fastapi import APIRouter
from pydantic import BaseModel

from models.credibility import score_domain
from utils.nvidia_reranker import rerank_claim
from database.db import insert_report, log_request

router = APIRouter()


class URLRequest(BaseModel):
    url: str


def fetch_page_title(url: str) -> str | None:
    """Attempt to fetch and extract the page title for analysis."""
    try:
        import requests
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "TruthLens-Bot/1.0 (Fact-checking service)"
        })
        if resp.status_code == 200:
            match = re.search(r'<title[^>]*>(.*?)</title>', resp.text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]
    except Exception:
        pass
    return None


@router.post("/analyze-url")
async def analyze_url_endpoint(req: URLRequest):
    start = time.time()

    url = req.url.strip()
    if not url:
        return {"error": "URL cannot be empty"}

    # Score domain credibility
    cred = score_domain(url)

    # Try to fetch page title for display
    page_title = fetch_page_title(url)

    # Determine base verdict from credibility score
    if cred.score >= 75:
        verdict    = "REAL"
    elif cred.score >= 45:
        verdict    = "SUSPICIOUS"
    else:
        verdict    = "FAKE"

    confidence = cred.score / 100.0

    # NVIDIA NIM Reranker — semantically check page title against fact database
    nvidia_analysis = None
    if page_title:
        nvidia_analysis = rerank_claim(page_title)
        if nvidia_analysis.get("status") == "ok":
            nv_signal     = nvidia_analysis.get("verdict_signal", "unverified")
            nv_confidence = nvidia_analysis.get("confidence", 0.0)

            if nv_signal == "misinformation" and nv_confidence >= 0.65 and verdict == "REAL":
                verdict    = "SUSPICIOUS"
                cred.signals.insert(0, "🟡 NVIDIA AI: Page title resembles known misinformation patterns")
            elif nv_signal == "verified" and nv_confidence >= 0.65 and verdict == "SUSPICIOUS":
                verdict    = "REAL"
                cred.signals.insert(0, "🟢 NVIDIA AI: Page title matches verified fact patterns")

            # Blend confidence
            confidence = round(confidence * 0.65 + nv_confidence * 0.35, 3)

    # Save report
    insert_report(
        input_type="url",
        input_summary=url[:200],
        verdict=verdict,
        confidence=confidence,
        credibility_score=cred.score,
        signals=cred.signals + cred.flags,
    )

    elapsed = (time.time() - start) * 1000
    log_request("/analyze-url", 200, elapsed)

    return {
        "url":                  url,
        "domain":               cred.domain,
        "verdict":              verdict,
        "credibility_score":    cred.score,
        "tier":                 cred.tier,
        "flags":                cred.flags,
        "signals":              cred.signals,
        "is_https":             cred.is_https,
        "page_title":           page_title,
        "verification_sources": cred.verification_sources,
        "nvidia_analysis":      nvidia_analysis,
        "processing_time_ms":   round(elapsed, 1),
    }
