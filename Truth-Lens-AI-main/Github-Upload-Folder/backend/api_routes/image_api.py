"""
TruthLens AI — Image Analysis API Route
POST /analyze-image
"""

import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.image_model import analyze_image
from database.db import insert_report, log_request

router = APIRouter()

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}


@router.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    start = time.time()

    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, WebP, BMP"
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    # Run deepfake analysis
    result = analyze_image(image_bytes, filename=file.filename or "uploaded_image")

    if result.verdict == "ERROR":
        raise HTTPException(status_code=422, detail=result.signals[0] if result.signals else "Image processing failed")

    # Save to DB
    insert_report(
        input_type="image",
        input_summary=file.filename or "uploaded_image",
        verdict=result.verdict,
        confidence=result.confidence,
        credibility_score=None,
        signals=result.signals,
    )

    elapsed = (time.time() - start) * 1000
    log_request("/analyze-image", 200, elapsed)

    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "artifacts_detected": result.artifacts_detected,
        "signals": result.signals,
        "analysis_scores": {
            "frequency_anomaly": result.frequency_anomaly,
            "noise_score": result.noise_score,
            "blur_score": result.blur_score,
        },
        "processing_time_ms": round(elapsed, 1),
    }
