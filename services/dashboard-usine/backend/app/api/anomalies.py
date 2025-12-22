"""
API endpoints for anomalies
"""
import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from app.models.anomaly import AnomalyListResponse
from app.services.anomaly_service import AnomalyService

logger = logging.getLogger(__name__)

router = APIRouter()
anomaly_service = AnomalyService()


@router.get("/", response_model=AnomalyListResponse)
async def get_anomalies(
    asset_id: Optional[str] = Query(None),
    sensor_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    criticality: Optional[str] = Query(None),
    is_anomaly: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0)
):
    """Get list of anomalies"""
    try:
        result = await anomaly_service.get_anomalies(
            asset_id=asset_id,
            sensor_id=sensor_id,
            start_date=start_date,
            end_date=end_date,
            criticality=criticality,
            is_anomaly=is_anomaly,
            limit=limit,
            offset=offset
        )
        
        # Convert items to Anomaly models
        # Note: detection-anomalies API returns 'anomalies' key, not 'items'
        from app.models.anomaly import Anomaly
        anomalies = []
        raw_anomalies = result.get('anomalies', result.get('items', []))
        logger.info(f"Processing {len(raw_anomalies) if raw_anomalies else 0} anomalies from API")
        
        for item in raw_anomalies:
            try:
                timestamp_str = item.get('timestamp')
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                # Parse created_at (when the anomaly was detected/saved)
                created_at_str = item.get('created_at')
                created_at = None
                if created_at_str:
                    if isinstance(created_at_str, str):
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    else:
                        created_at = created_at_str
                
                # Map 'criticality' to 'severity' (detection-anomalies uses 'criticality')
                severity = item.get('severity') or item.get('criticality', 'medium')
                
                # Map 'final_score' to 'anomaly_score'
                anomaly_score = item.get('anomaly_score') or item.get('final_score')
                
                # Convert id to string if it's an integer
                anomaly_id = str(item.get('id')) if item.get('id') is not None else None
                
                anomalies.append(Anomaly(
                    id=anomaly_id,
                    asset_id=item.get('asset_id'),
                    sensor_id=item.get('sensor_id'),
                    timestamp=timestamp,
                    created_at=created_at,
                    severity=severity,
                    is_anomaly=item.get('is_anomaly', True),
                    anomaly_score=anomaly_score,
                    model_used=item.get('model_used'),
                    features=item.get('features'),
                    metadata=item.get('metadata')
                ))
                logger.debug(f"Successfully converted anomaly: {anomaly_id}")
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion d'une anomalie: {e}")
                continue
        
        return AnomalyListResponse(
            anomalies=anomalies,
            total=result.get('total', 0),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{anomaly_id}")
async def get_anomaly(anomaly_id: str):
    """Get a specific anomaly by ID"""
    try:
        result = await anomaly_service.get_anomaly_by_id(anomaly_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'anomalie: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

