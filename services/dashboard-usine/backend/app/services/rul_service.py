"""
Service client for prediction-rul service
"""
import logging
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class RULService:
    """Client for prediction-rul service"""
    
    def __init__(self):
        self.base_url = settings.prediction_rul_url
        self.timeout = 30.0
    
    async def get_rul_predictions(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get RUL predictions"""
        try:
            params = {"limit": limit, "offset": offset}
            if asset_id:
                params["asset_id"] = asset_id
            if sensor_id:
                params["sensor_id"] = sensor_id
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/rul/",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération des prédictions RUL: {e}")
            return {"predictions": [], "total": 0, "limit": limit, "offset": offset}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prédictions RUL: {e}", exc_info=True)
            return {"predictions": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_latest_rul(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get latest RUL prediction for an asset"""
        try:
            result = await self.get_rul_predictions(asset_id=asset_id, limit=1)
            predictions = result.get("predictions", [])
            if predictions:
                return predictions[0]
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la dernière prédiction RUL: {e}", exc_info=True)
            return None
    
    async def predict_rul(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Request a new RUL prediction"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/rul/predict",
                    json=request_data
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la prédiction RUL: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction RUL: {e}", exc_info=True)
            return None
    
    async def get_status(self) -> Dict[str, Any]:
        """Get prediction-rul service status"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/rul/status")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Impossible de récupérer le statut du service prediction-rul: {e}")
            return {"status": "unknown"}

