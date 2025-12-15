"""
Service client for extraction-features service
"""
import logging
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class ExtractionService:
    """Client for extraction-features service"""
    
    def __init__(self):
        self.base_url = settings.extraction_features_url
        self.timeout = 30.0
    
    async def get_features(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get features for an asset"""
        try:
            params = {"limit": limit}
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()
            if feature_names:
                params["feature_names"] = ",".join(feature_names)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/features/features/{asset_id}",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération des features pour {asset_id}: {e}")
            return {"asset_id": asset_id, "count": 0, "features": []}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des features: {e}", exc_info=True)
            return {"asset_id": asset_id, "count": 0, "features": []}
    
    async def get_feature_vector(self, asset_id: str, feature_vector_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feature vector for an asset"""
        try:
            params = {}
            if feature_vector_id:
                params["feature_vector_id"] = feature_vector_id
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/features/features/{asset_id}/vector",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération du vecteur de features: {e}")
            return {}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du vecteur de features: {e}", exc_info=True)
            return {}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get extraction-features service status"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/features/status")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Impossible de récupérer le statut du service extraction-features: {e}")
            return {"status": "unknown"}

