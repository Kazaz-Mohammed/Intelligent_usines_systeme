"""
API endpoints pour la détection d'anomalies
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.anomaly_data import (
    AnomalyDetectionRequest,
    AnomalyDetectionResult,
    TrainingRequest,
    TrainingResult,
    CriticalityLevel
)
from app.services.anomaly_detection_service import AnomalyDetectionService
from app.database.postgresql import PostgreSQLService
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Créer le router
router = APIRouter(prefix="/api/v1/anomalies", tags=["Anomalies"])

# Instances singletons des services
_anomaly_detection_service: Optional[AnomalyDetectionService] = None
_postgresql_service: Optional[PostgreSQLService] = None


def get_anomaly_detection_service() -> AnomalyDetectionService:
    """Dependency pour obtenir l'instance du service de détection"""
    global _anomaly_detection_service
    if _anomaly_detection_service is None:
        _anomaly_detection_service = AnomalyDetectionService()
    return _anomaly_detection_service


def get_postgresql_service() -> PostgreSQLService:
    """Dependency pour obtenir l'instance du service PostgreSQL"""
    global _postgresql_service
    if _postgresql_service is None:
        _postgresql_service = PostgreSQLService()
    return _postgresql_service


@router.post("/detect", response_model=AnomalyDetectionResult, status_code=200)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    service: AnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Détecte une anomalie à partir de features
    
    Args:
        request: Requête de détection avec features
        service: Service de détection d'anomalies
    
    Returns:
        Résultat de la détection avec scores et criticité
    """
    try:
        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Aucun modèle n'est entraîné. Veuillez entraîner les modèles d'abord."
            )
        
        result = service.detect_anomaly(request)
        
        # Journaliser l'anomalie si détectée
        if result.is_anomaly:
            try:
                db_service = get_postgresql_service()
                db_service.insert_anomaly(result)
            except Exception as e:
                # Ne pas faire échouer la requête si la journalisation échoue
                logger.warning(f"Erreur lors de la journalisation de l'anomalie: {e}")
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la détection d'anomalie: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/detect/batch", response_model=List[AnomalyDetectionResult], status_code=200)
async def detect_anomalies_batch(
    requests: List[AnomalyDetectionRequest],
    service: AnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Détecte des anomalies pour un batch de requêtes
    
    Args:
        requests: Liste de requêtes de détection
        service: Service de détection d'anomalies
    
    Returns:
        Liste de résultats de détection
    """
    try:
        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Aucun modèle n'est entraîné. Veuillez entraîner les modèles d'abord."
            )
        
        results = service.detect_anomalies_batch(requests)
        return results
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la détection batch d'anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/train", response_model=Dict[str, TrainingResult], status_code=200)
async def train_models(
    training_data: Dict[str, Any],
    service: AnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Entraîne les modèles de détection d'anomalies
    
    Body attendu:
    {
        "data": [[feature1, feature2, ...], ...],  # Array 2D numpy-compatible
        "feature_names": ["feature1", "feature2", ...],  # Optionnel
        "model_names": ["isolation_forest", "one_class_svm", "lstm_autoencoder"]  # Optionnel, tous par défaut
    }
    
    Args:
        training_data: Données d'entraînement
        service: Service de détection d'anomalies
    
    Returns:
        Résultats de l'entraînement pour chaque modèle
    """
    try:
        import numpy as np
        
        # Extraire les données
        if "data" not in training_data:
            raise HTTPException(status_code=400, detail="Le champ 'data' est requis")
        
        data = np.array(training_data["data"])
        feature_names = training_data.get("feature_names")
        
        if len(data.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Les données doivent être un array 2D, reçu shape: {data.shape}"
            )
        
        # Entraîner tous les modèles
        results = service.train_all_models(data, feature_names)
        
        # Convertir en format TrainingResult
        training_results = {}
        for model_name, result in results.items():
            # Skip non-model entries like 'models_saved'
            if not isinstance(result, dict):
                continue
            
            if result["status"] == "success":
                training_results[model_name] = TrainingResult(
                    model_name=model_name,
                    status="success",
                    message="Modèle entraîné avec succès",
                    metrics=result.get("metrics", {}),
                    model_version="v1.0"
                )
            else:
                training_results[model_name] = TrainingResult(
                    model_name=model_name,
                    status="error",
                    message=result.get("error", "Erreur inconnue")
                )
        
        return training_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint pour le service de détection d'anomalies
    
    Returns:
        Status du service
    """
    return {
        "status": "healthy",
        "service": "detection-anomalies",
        "version": "0.1.0"
    }


@router.get("/status", status_code=200)
async def get_status(
    service: AnomalyDetectionService = Depends(get_anomaly_detection_service)
):
    """
    Retourne le statut des modèles
    
    Returns:
        Statut de tous les modèles (entraînés ou non)
    """
    try:
        status = service.get_model_status()
        is_ready = service.is_ready()
        
        return {
            "ready": is_ready,
            "models": status
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/", status_code=200)
async def get_anomalies(
    asset_id: Optional[str] = Query(None, description="Filtrer par asset_id"),
    sensor_id: Optional[str] = Query(None, description="Filtrer par sensor_id"),
    start_date: Optional[datetime] = Query(None, description="Date de début"),
    end_date: Optional[datetime] = Query(None, description="Date de fin"),
    criticality: Optional[str] = Query(None, description="Filtrer par criticité (low, medium, high, critical)"),
    is_anomaly: Optional[bool] = Query(None, description="Filtrer par is_anomaly (true/false)"),
    limit: int = Query(100, ge=1, le=5000, description="Nombre maximum de résultats"),
    offset: int = Query(0, ge=0, description="Offset pour la pagination"),
    db_service: PostgreSQLService = Depends(get_postgresql_service)
):
    """
    Récupère l'historique des anomalies détectées depuis la base de données
    
    Args:
        asset_id: ID de l'actif (optionnel)
        sensor_id: ID du capteur (optionnel)
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)
        criticality: Niveau de criticité (optionnel)
        is_anomaly: Filtrer par anomalie détectée ou non (optionnel)
        limit: Nombre maximum de résultats
        offset: Offset pour la pagination
        db_service: Service PostgreSQL
    
    Returns:
        Dict avec 'items' (liste des anomalies), 'total', 'limit', 'offset'
    """
    try:
        # Convertir criticality string en enum si fourni
        criticality_enum = None
        if criticality:
            try:
                criticality_enum = CriticalityLevel(criticality.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Criticalité invalide: {criticality}. Valeurs acceptées: low, medium, high, critical"
                )
        
        # Récupérer les anomalies
        anomalies = db_service.get_anomalies(
            asset_id=asset_id,
            sensor_id=sensor_id,
            start_date=start_date,
            end_date=end_date,
            criticality=str(criticality_enum.value) if criticality_enum else None,
            is_anomaly=is_anomaly,
            limit=limit,
            offset=offset
        )
        
        # Compter le total (sans pagination)
        total = db_service.get_anomaly_count(
            asset_id=asset_id,
            sensor_id=sensor_id,
            start_date=start_date,
            end_date=end_date,
            is_anomaly=is_anomaly,
            criticality=str(criticality_enum.value) if criticality_enum else None
        )
        
        return {
            "anomalies": anomalies,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "asset_id": asset_id,
                "sensor_id": sensor_id,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "is_anomaly": is_anomaly,
                "criticality": criticality
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/metrics", status_code=200)
async def get_metrics():
    """
    Retourne les métriques MLflow
    
    Note: Cette fonctionnalité nécessite l'intégration MLflow.
    Pour l'instant, retourne un message indiquant que c'est à implémenter.
    
    Returns:
        Métriques MLflow
    """
    # TODO: Implémenter avec MLflow
    return {
        "message": "Fonctionnalité à implémenter avec MLflow",
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "mlflow_experiment_name": settings.mlflow_experiment_name
    }



