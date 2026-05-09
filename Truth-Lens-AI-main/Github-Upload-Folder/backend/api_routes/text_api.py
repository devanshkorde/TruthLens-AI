import time
from fastapi import APIRouter
from pydantic import BaseModel

from models.text_model import analyze_text
from models.credibility import score_domain
from utils.explainer import generate_explanation
from utils.realtime_checker import verify_content
from utils.nvidia_reranker import rerank_claim
from database.db import insert_report, log_request

router = APIRouter()


class TextRequest(BaseModel):
    text: str
    url: str | None = None


@router.post("/analyze-text")
async def analyze_text_endpoint(req: TextRequest):
    start = time.time()

    text = req.text.strip()
    if len(text) < 10:
        return {"error": "Text too short for analysis (minimum 10 characters)"}

    # 1. Run local text model (NLP heuristics + ML)
    result = analyze_text(text)

    # 2. Run source credibility if URL provided
    credibility = None
    if req.url:
        cred_result = score_domain(req.url)
        credibility = {
            "domain":   cred_result.domain,
            "score":    cred_result.score,
            "tier":     cred_result.tier,
            "flags":    cred_result.flags,
            "signals":  cred_result.signals,
            "is_https": cred_result.is_https,
        }
        # Adjust verdict if source is known fake
        if cred_result.tier == "fake" and result.verdict != "FAKE":
            result.verdict = "FAKE"
            result.signals.insert(0, "\ud83d\udd34 Content from a known misinformation publisher")
        elif cred_result.tier == "trusted" and result.verdict == "SUSPICIOUS":
            result.verdict = "REAL"
            result.signals.insert(0, "\ud83d\udfe2 Published by a trusted, verified news source")

    # 3. Real-time cross-source check (heuristic keyword matching)
    verification = verify_content(text)

    # 4. NVIDIA NIM Reranker \u2014 semantic similarity against fact/misinfo database
    nvidia_analysis = rerank_claim(text)

    # 4a. Apply NVIDIA verdict signal to refine final verdict
    if nvidia_analysis.get("status") == "ok":
        nv_signal     = nvidia_analysis.get("verdict_signal", "unverified")
        nv_confidence = nvidia_analysis.get("confidence", 0.0)

        if nv_signal == "misinformation" and nv_confidence >= 0.65:
            if result.verdict == "REAL":
                result.verdict = "SUSPICIOUS"
                result.signals.insert(0, "\ud83d\udfe1 NVIDIA AI: Claim resembles known misinformation patterns \u2014 downgraded to Suspicious")
            elif result.verdict == "SUSPICIOUS":
                result.verdict = "FAKE"
                result.signals.insert(0, "\ud83d\udd34 NVIDIA AI: Strong match with known misinformation patterns")

        elif nv_signal == "verified" and nv_confidence >= 0.65:
            if result.verdict == "FAKE":
                result.verdict = "SUSPICIOUS"
                result.signals.insert(0, "\ud83d\udfe1 NVIDIA AI: Claim resembles verified facts \u2014 upgraded to Suspicious")
            elif result.verdict == "SUSPICIOUS":
                result.verdict = "REAL"
                result.signals.insert(0, "\ud83d\udfe2 NVIDIA AI: Claim closely matches verified fact patterns")

        # Blend confidence: 60% local model, 40% NVIDIA NIM
        result.confidence = round(result.confidence * 0.60 + nv_confidence * 0.40, 3)

    # 5. Generate XAI explanation
    explanation = generate_explanation(text, result.verdict, result.confidence, result.signals)

    # 6. Save to DB
    insert_report(
        input_type="text",
        input_summary=text[:300],
        verdict=result.verdict,
        confidence=result.confidence,
        credibility_score=credibility["score"] if credibility else None,
        signals=result.signals,
    )

    elapsed = (time.time() - start) * 1000
    log_request("/analyze-text", 200, elapsed)

    return {
        "verdict":             result.verdict,
        "confidence":          result.confidence,
        "signals":             result.signals,
        "highlighted_phrases": result.highlighted_phrases,
        "credibility":         credibility,
        "verification":        verification,
        "nvidia_analysis":     nvidia_analysis,
        "explanation":         explanation,
        "processing_time_ms":  round(elapsed, 1),
    }
