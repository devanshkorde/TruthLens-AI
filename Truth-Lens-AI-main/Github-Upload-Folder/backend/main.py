"""
TruthLens AI — FastAPI Main Application
=========================================
Entry point for the TruthLens backend API server.
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
from pathlib import Path

# Ensure all local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database.db import init_db
from api_routes.text_api import router as text_router
from api_routes.image_api import router as image_router
from api_routes.video_api import router as video_router
from api_routes.url_api import router as url_router
from api_routes.dashboard_api import router as dashboard_router

# ──────────────────────────────────────────────
# App initialization
# ──────────────────────────────────────────────

app = FastAPI(
    title="TruthLens AI API",
    description="Multi-modal fake news and deepfake detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(text_router, tags=["Text Analysis"])
app.include_router(image_router, tags=["Image Analysis"])
app.include_router(video_router, tags=["Video Analysis"])
app.include_router(url_router, tags=["URL Analysis"])
app.include_router(dashboard_router, tags=["Dashboard"])


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("  TruthLens AI — Backend Starting")
    print("=" * 50)
    init_db()
    print("[API] All routes registered.")
    print("[API] Swagger UI available at http://localhost:8000/docs")


# trigger reload
@app.get("/")
async def root():
    return {
        "service": "TruthLens AI",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "POST /analyze-text",
            "POST /analyze-image",
            "POST /analyze-video",
            "POST /analyze-url",
            "GET  /dashboard-data",
            "GET  /docs (Swagger UI)",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
