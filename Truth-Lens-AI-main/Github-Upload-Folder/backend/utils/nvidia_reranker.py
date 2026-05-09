"""
TruthLens AI — NVIDIA NIM Reranker Integration
================================================
Uses nvidia/llama-3.2-nemoretriever-500m-rerank-v2 to score
how semantically relevant a news claim is against a curated
database of verified facts and known misinformation patterns.

Scoring Logic:
  - High relevance to "verified" passages  → REAL signal
  - High relevance to "misinformation" passages → FAKE signal
  - Low relevance across all passages       → SUSPICIOUS/UNVERIFIED

API Endpoint: POST https://integrate.api.nvidia.com/v1/ranking
"""

import json
import requests
from pathlib import Path

from config import ai_config

# ── Constants ─────────────────────────────────────────────────────────────────
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
RERANKER_MODEL  = "nvidia/llama-3.2-nemoretriever-500m-rerank-v2"
PASSAGES_PATH   = Path(__file__).parent.parent / "data" / "fact_passages.json"
REQUEST_TIMEOUT = 12  # seconds

# ── Load fact passages once at import time ────────────────────────────────────
def _load_passages() -> list[dict]:
    if not PASSAGES_PATH.exists():
        return []
    with open(PASSAGES_PATH, encoding="utf-8") as f:
        return json.load(f)

FACT_PASSAGES: list[dict] = _load_passages()


# ── Core reranker function ────────────────────────────────────────────────────
def rerank_claim(claim: str) -> dict:
    """
    Score a news claim against TruthLens's curated fact-passage database
    using the NVIDIA NIM reranker API.

    Args:
        claim: The news headline / article text to verify (max ~512 tokens)

    Returns:
        dict with keys:
          status          : "ok" | "skipped" | "error"
          verdict_signal  : "verified" | "misinformation" | "unverified"
          confidence      : float 0–1
          message         : human-readable summary
          top_matches     : list of top-3 closest passages with labels
          avg_verified_score  : average logit of top-K verified passages
          avg_misinfo_score   : average logit of top-K misinformation passages
          provider        : "nvidia-nemoretriever-rerank"
          model           : model name string
    """

    # ── Guard: need API key ──────────────────────────────────────────────────
    if not ai_config.has_nvidia():
        return {
            "status": "skipped",
            "reason": "NVIDIA_API_KEY not configured",
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }

    if not FACT_PASSAGES:
        return {
            "status": "skipped",
            "reason": "fact_passages.json is empty or missing",
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }

    # ── Build API request ────────────────────────────────────────────────────
    passages = [{"role": "user", "content": p["text"]} for p in FACT_PASSAGES]

    headers = {
        "Authorization": f"Bearer {ai_config.NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    payload = {
        "model":    RERANKER_MODEL,
        "query":    {"role": "user", "content": claim[:1500]},  # cap for token safety
        "passages": passages,
    }

    # ── Call NVIDIA NIM ──────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/ranking",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "reason": "NVIDIA NIM API request timed out",
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }
    except requests.exceptions.HTTPError as exc:
        return {
            "status": "error",
            "reason": f"NVIDIA API HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }

    # ── Parse rankings ───────────────────────────────────────────────────────
    rankings = data.get("rankings", [])
    if not rankings:
        return {
            "status": "error",
            "reason": "NVIDIA NIM returned empty rankings",
            "verdict_signal": "unverified",
            "confidence": 0.3,
        }

    # Attach metadata from our passage DB back to each ranking entry
    scored: list[dict] = []
    for r in rankings:
        idx   = r["index"]
        logit = r["logit"]
        if idx < len(FACT_PASSAGES):
            p = FACT_PASSAGES[idx]
            scored.append({
                "text":  p["text"][:120] + "…",
                "label": p["label"],          # "verified" | "misinformation"
                "topic": p.get("topic", "general"),
                "logit": logit,
            })

    # Sort by relevance (highest logit = most similar to the claim)
    scored.sort(key=lambda x: x["logit"], reverse=True)
    top_k = scored[:6]  # consider top-6 passages

    # ── Compute verdict signal ───────────────────────────────────────────────
    verified_scores = [s["logit"] for s in top_k if s["label"] == "verified"]
    misinfo_scores  = [s["logit"] for s in top_k if s["label"] == "misinformation"]

    avg_verified = (sum(verified_scores) / len(verified_scores)) if verified_scores else -999
    avg_misinfo  = (sum(misinfo_scores)  / len(misinfo_scores))  if misinfo_scores  else -999

    top_logit = top_k[0]["logit"] if top_k else -999

    # Thresholds determined empirically for this model (logits are typically -10…+5)
    STRONG_MATCH_THRESHOLD = -3.0   # logit above this = strong semantic match

    if top_logit < STRONG_MATCH_THRESHOLD:
        # Claim doesn't strongly resemble anything in our DB
        verdict_signal = "unverified"
        confidence     = 0.30
        message        = "⚠️ Claim does not strongly match any verified fact or known misinformation pattern"

    elif avg_verified >= avg_misinfo:
        # Claim is more semantically similar to verified/true statements
        gap         = avg_verified - avg_misinfo
        confidence  = round(min(0.52 + gap * 0.06, 0.93), 3)
        verdict_signal = "verified"
        match_count = len(verified_scores)
        message = f"✅ Closely matches {match_count} verified fact pattern(s) in TruthLens database"

    else:
        # Claim is more semantically similar to known misinformation
        gap         = avg_misinfo - avg_verified
        confidence  = round(min(0.52 + gap * 0.06, 0.93), 3)
        verdict_signal = "misinformation"
        match_count = len(misinfo_scores)
        message = f"🔴 Closely matches {match_count} known misinformation pattern(s) in TruthLens database"

    return {
        "status":              "ok",
        "verdict_signal":      verdict_signal,
        "confidence":          confidence,
        "message":             message,
        "top_matches":         top_k[:3],
        "avg_verified_score":  round(avg_verified, 4) if avg_verified != -999 else None,
        "avg_misinfo_score":   round(avg_misinfo,  4) if avg_misinfo  != -999 else None,
        "total_passages_scored": len(scored),
        "provider":            "nvidia-nemoretriever-rerank",
        "model":               RERANKER_MODEL,
    }
