"""
API endpoints for RUL predictions
"""
import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from app.models.rul import RULListResponse
from app.services.rul_service import RULService

logger = logging.getLogger(__name__)

router = APIRouter()
rul_service = RULService()


@router.get("/", response_model=RULListResponse)
async def get_rul_predictions(
    asset_id: Optional[str] = Query(None),
    sensor_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get RUL predictions"""
    try:
        result = await rul_service.get_rul_predictions(
            asset_id=asset_id,
            sensor_id=sensor_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        # Convert predictions to RULPrediction models
        from app.models.rul import RULPrediction
        predictions = []
        for pred in result.get('predictions', []):
            try:
                timestamp_str = pred.get('timestamp')
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                # Convert id to string if it's an integer
                pred_id = str(pred.get('id')) if pred.get('id') is not None else None
                
                predictions.append(RULPrediction(
                    id=pred_id,
                    asset_id=pred.get('asset_id'),
                    sensor_id=pred.get('sensor_id'),
                    timestamp=timestamp,
                    rul_prediction=float(pred.get('rul_prediction', 0.0)),
                    confidence_interval_lower=pred.get('confidence_interval_lower'),
                    confidence_interval_upper=pred.get('confidence_interval_upper'),
                    confidence_level=pred.get('confidence_level'),
                    uncertainty=pred.get('uncertainty'),
                    model_used=pred.get('model_used'),
                    model_version=pred.get('model_version'),
                    model_scores=pred.get('model_scores'),
                    features=pred.get('features'),
                    metadata=pred.get('metadata')
                ))
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion d'une prédiction RUL: {e}, data: {pred}")
                continue
        
        return RULListResponse(
            predictions=predictions,
            total=result.get('total', 0),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prédictions RUL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}/latest")
async def get_latest_rul(asset_id: str):
    """Get latest RUL prediction for an asset"""
    try:
        result = await rul_service.get_latest_rul(asset_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"No RUL prediction found for asset {asset_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la dernière prédiction RUL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

