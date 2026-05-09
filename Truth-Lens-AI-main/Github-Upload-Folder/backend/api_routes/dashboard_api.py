"""
TruthLens AI — Dashboard Data API Route
GET /dashboard-data
"""

from fastapi import APIRouter
from database.db import get_dashboard_data

router = APIRouter()


@router.get("/dashboard-data")
async def dashboard_data():
    data = get_dashboard_data()

    # Compute fake percentage
    total = data["total"] or 1
    fake_pct = round(data["fake_count"] / total * 100, 1)
    real_pct = round(data["real_count"] / total * 100, 1)
    suspicious_pct = round(data["suspicious_count"] / total * 100, 1)

    return {
        **data,
        "fake_percentage": fake_pct,
        "real_percentage": real_pct,
        "suspicious_percentage": suspicious_pct,
    }
