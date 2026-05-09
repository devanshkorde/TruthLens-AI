"""
TruthLens AI — Video Analysis API Route
POST /analyze-video
"""

import time
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.video_model import analyze_video
from database.db import insert_report, log_request

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_TYPES = {"video/mp4", "video/webm", "video/x-msvideo", "video/quicktime"}

@router.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    start = time.time()

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: MP4, WebM, AVI, MOV"
        )

    video_bytes = await file.read()
    if len(video_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        result = analyze_video(tmp_path)
    finally:
        os.remove(tmp_path)

    if result.verdict == "ERROR":
        raise HTTPException(status_code=422, detail=result.signals[0] if result.signals else "Video processing failed")

    insert_report(
        input_type="video",
        input_summary=file.filename or "uploaded_video",
        verdict=result.verdict,
        confidence=result.confidence,
        credibility_score=None,
        signals=result.signals,
    )

    elapsed = (time.time() - start) * 1000
    log_request("/analyze-video", 200, elapsed)

    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "artifacts_detected": result.artifacts_detected,
        "signals": result.signals,
        "analysis_scores": {
            "avg_frequency_anomaly": result.avg_frequency_anomaly,
            "avg_noise_score": result.avg_noise_score,
            "avg_blur_score": result.avg_blur_score,
        },
        "frames_analyzed": result.frames_analyzed,
        "processing_time_ms": round(elapsed, 1),
    }
