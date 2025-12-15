"""
API endpoints for KPIs
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from app.models.kpi import KPISummary, KPI, KPITrend
from app.services.kpi_service import KPIService
from app.database.postgresql import get_postgresql_service, PostgreSQLService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/summary", response_model=KPISummary)
async def get_kpi_summary(
    asset_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get KPI summary (MTBF, MTTR, OEE)"""
    try:
        kpi_service = KPIService(db)
        summary = await kpi_service.get_kpi_summary(asset_id, days)
        return summary
    except Exception as e:
        logger.error(f"Erreur lors du calcul des KPIs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trend/{metric_name}", response_model=KPITrend)
async def get_kpi_trend(
    metric_name: str,
    asset_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get KPI trend over time"""
    try:
        from datetime import datetime, timedelta
        
        start_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT value, timestamp, unit
            FROM kpis_history
            WHERE metric_name = $1
            AND timestamp >= $2
        """
        params = [metric_name, start_date]
        
        if asset_id:
            query += " AND asset_id = $3"
            params.append(asset_id)
        
        query += " ORDER BY timestamp ASC"
        
        rows = await db.fetch(query, *params)
        
        values = []
        timestamps = []
        unit = None
        
        for row in rows:
            values.append(float(row['value']))
            timestamps.append(row['timestamp'])
            if not unit and row['unit']:
                unit = row['unit']
        
        return KPITrend(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            unit=unit
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la tendance KPI: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

