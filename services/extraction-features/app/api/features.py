"""
API REST pour le service Extraction Features
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.asset_service import AssetService
from app.database.timescaledb import TimescaleDBService
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()


class FeatureExtractionServiceSingleton:
    """Singleton pour le service d'extraction de features"""
    _instance: Optional[FeatureExtractionService] = None
    
    @classmethod
    def get_instance(cls) -> FeatureExtractionService:
        if cls._instance is None:
            cls._instance = FeatureExtractionService()
        return cls._instance
    
    @classmethod
    def set_instance(cls, instance: FeatureExtractionService):
        cls._instance = instance


def get_feature_extraction_service() -> FeatureExtractionService:
    """Retourne l'instance du service d'extraction de features"""
    return FeatureExtractionServiceSingleton.get_instance()


@router.get("/health")
async def health_check():
    """Endpoint de santé"""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": "0.1.0"
    }


@router.get("/status")
async def get_status():
    """Retourne le statut du service"""
    try:
        service = get_feature_extraction_service()
        statistics = service.get_statistics()
        
        return {
            "status": "running",
            "service": settings.service_name,
            "version": "0.1.0",
            "statistics": statistics,
            "configuration": {
                "temporal_features": settings.enable_temporal_features,
                "frequency_features": settings.enable_frequency_features,
                "wavelet_features": settings.enable_wavelet_features,
                "standardization": settings.enable_standardization,
                "feast": settings.feast_enable
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{asset_id}")
async def get_features(
    asset_id: str,
    start_time: Optional[datetime] = Query(None, description="Timestamp de début"),
    end_time: Optional[datetime] = Query(None, description="Timestamp de fin"),
    feature_names: Optional[List[str]] = Query(None, description="Liste de noms de features"),
    limit: int = Query(100, description="Nombre maximum de features à retourner")
):
    """
    Récupère les features pour un asset donné
    
    Args:
        asset_id: ID de l'actif
        start_time: Timestamp de début (optionnel)
        end_time: Timestamp de fin (optionnel)
        feature_names: Liste de noms de features (optionnel)
        limit: Nombre maximum de features à retourner
    
    Returns:
        Liste de features
    """
    try:
        timescale_db_service = TimescaleDBService()
        features = timescale_db_service.get_features_by_asset(
            asset_id,
            limit
        )
        
        # Filtrer par start_time, end_time et feature_names si spécifiés
        if start_time:
            features = [f for f in features if f.metadata.get("timestamp") and f.metadata.get("timestamp") >= start_time]
        if end_time:
            features = [f for f in features if f.metadata.get("timestamp") and f.metadata.get("timestamp") <= end_time]
        if feature_names:
            features = [f for f in features if f.name in feature_names]
        
        # Limiter le nombre de résultats
        features = features[:limit]
        
        return {
            "asset_id": asset_id,
            "count": len(features),
            "features": [feature.model_dump() for feature in features]
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{asset_id}/vector")
async def get_feature_vector(
    asset_id: str,
    feature_vector_id: Optional[str] = Query(None, description="ID du vecteur de features")
):
    """
    Récupère un vecteur de features pour un asset donné
    
    Args:
        asset_id: ID de l'actif
        feature_vector_id: ID du vecteur de features (optionnel)
    
    Returns:
        Vecteur de features
    """
    try:
        # Pour l'instant, on retourne les features récentes
        # TODO: Implémenter la récupération d'un vecteur spécifique
        timescale_db_service = TimescaleDBService()
        features = await timescale_db_service.get_features_by_asset(
            asset_id,
            None,
            None,
            None
        )
        
        if not features:
            raise HTTPException(status_code=404, detail=f"Aucune feature trouvée pour asset_id={asset_id}")
        
        # Créer un vecteur de features à partir des features récentes
        features_dict = {feature.name: feature.value for feature in features}
        
        # Extract timestamp, asset_id from metadata
        first_timestamp = features[0].metadata.get("timestamp") if features[0].metadata else None
        last_timestamp = features[-1].metadata.get("timestamp") if features[-1].metadata else None
        
        feature_vector = ExtractedFeaturesVector(
            feature_vector_id=feature_vector_id or f"fv_{asset_id}_{uuid.uuid4().hex[:8]}",
            timestamp=first_timestamp or datetime.now(),
            asset_id=asset_id,
            start_time=last_timestamp or datetime.now(),
            end_time=first_timestamp or datetime.now(),
            features=features_dict,
            feature_metadata={
                "num_features": len(features),
                "feature_types": list(set(f.feature_type for f in features))
            }
        )
        
        return feature_vector.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du vecteur de features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{asset_id}")
async def get_asset_info(asset_id: str):
    """
    Récupère les informations sur un actif
    
    Args:
        asset_id: ID de l'actif
    
    Returns:
        Informations sur l'actif
    """
    try:
        asset_service = AssetService()
        asset_info = asset_service.get_asset_info(asset_id)
        
        if not asset_info:
            raise HTTPException(status_code=404, detail=f"Actif {asset_id} non trouvé")
        
        return asset_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations de l'actif: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{asset_id}/type")
async def get_asset_type(asset_id: str):
    """
    Récupère le type d'actif
    
    Args:
        asset_id: ID de l'actif
    
    Returns:
        Type d'actif
    """
    try:
        asset_service = AssetService()
        asset_type = asset_service.get_asset_type(asset_id)
        
        if not asset_type:
            raise HTTPException(status_code=404, detail=f"Actif {asset_id} non trouvé")
        
        return {
            "asset_id": asset_id,
            "type": asset_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du type d'actif: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Retourne les métriques du service"""
    try:
        service = get_feature_extraction_service()
        statistics = service.get_statistics()
        
        return {
            "buffers": statistics.get("buffers", {}),
            "last_processed": statistics.get("last_processed", {}),
            "services": statistics.get("services", {})
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute")
async def compute_features(
    asset_id: str,
    sensor_ids: List[str],
    start_time: datetime,
    end_time: datetime
):
    """
    Calcule les features pour un asset et des capteurs donnés
    
    Args:
        asset_id: ID de l'actif
        sensor_ids: Liste d'IDs de capteurs
        start_time: Timestamp de début
        end_time: Timestamp de fin
    
    Returns:
        Liste de features calculées
    """
    try:
        # Récupérer les données depuis TimescaleDB
        timescale_db_service = TimescaleDBService()
        # TODO: Implémenter la récupération des données prétraitées depuis TimescaleDB
        # Pour l'instant, on retourne une réponse vide
        
        return {
            "asset_id": asset_id,
            "sensor_ids": sensor_ids,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "message": "Feature computation not yet implemented for manual requests"
        }
    except Exception as e:
        logger.error(f"Erreur lors du calcul des features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

