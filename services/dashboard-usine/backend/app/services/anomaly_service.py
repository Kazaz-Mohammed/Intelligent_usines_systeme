"""
Service client for detection-anomalies service
"""
import logging
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class AnomalyService:
    """Client for detection-anomalies service"""
    
    def __init__(self):
        self.base_url = settings.detection_anomalies_url
        self.timeout = 30.0
    
    async def get_anomalies(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        criticality: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get anomalies"""
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
            if criticality:
                params["criticality"] = criticality
            if is_anomaly is not None:
                params["is_anomaly"] = str(is_anomaly).lower()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/anomalies/",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération des anomalies: {e}")
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des anomalies: {e}", exc_info=True)
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_anomaly_by_id(self, anomaly_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific anomaly by ID"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v1/anomalies/{anomaly_id}")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération de l'anomalie {anomaly_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'anomalie: {e}", exc_info=True)
            return None
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detection-anomalies service status"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/anomalies/status")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Impossible de récupérer le statut du service detection-anomalies: {e}")
            return {"status": "unknown"}

