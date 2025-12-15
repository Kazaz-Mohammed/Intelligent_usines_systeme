"""
API endpoints pour la prédiction RUL
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict

from app.models.rul_data import (
    RULPredictionRequest,
    RULPredictionResult,
    TrainingRequest,
    TrainingResult
)
from app.services.rul_prediction_service import RULPredictionService
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Instance singleton du service
_rul_prediction_service: Optional[RULPredictionService] = None


def get_rul_prediction_service() -> RULPredictionService:
    """Dependency pour obtenir l'instance du service de prédiction RUL"""
    global _rul_prediction_service
    if _rul_prediction_service is None:
        _rul_prediction_service = RULPredictionService()
    return _rul_prediction_service


@router.post("/predict", response_model=RULPredictionResult, status_code=200)
async def predict_rul(
    request: RULPredictionRequest,
    model_name: Optional[str] = None,
    use_ensemble: bool = True,
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Prédit la RUL (Remaining Useful Life) pour un actif
    
    Args:
        request: Requête avec features et données séquentielles
        model_name: Nom du modèle à utiliser (lstm, gru, tcn, xgboost) - si None, utilise ensemble
        use_ensemble: Si True, agrège les prédictions de tous les modèles
        service: Service de prédiction RUL
    
    Returns:
        Résultat de la prédiction avec intervalle de confiance
    """
    try:
        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Aucun modèle n'est entraîné. Utilisez POST /api/v1/rul/train pour entraîner les modèles."
            )
        
        result = service.predict_rul(request, model_name=model_name, use_ensemble=use_ensemble)
        
        # Journaliser dans PostgreSQL
        try:
            from app.database.postgresql import PostgreSQLService
            postgresql_service = PostgreSQLService()
            postgresql_service.insert_rul_prediction(result)
        except Exception as e:
            logger.warning(f"Impossible de journaliser la prédiction RUL dans PostgreSQL: {e}")
            # Ne pas lever d'exception, la prédiction est réussie même si la journalisation échoue
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction RUL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/predict/batch", response_model=List[RULPredictionResult], status_code=200)
async def predict_rul_batch(
    requests: List[RULPredictionRequest],
    model_name: Optional[str] = None,
    use_ensemble: bool = True,
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Prédit la RUL pour plusieurs actifs (batch)
    
    Args:
        requests: Liste de requêtes
        model_name: Nom du modèle à utiliser (optionnel)
        use_ensemble: Si True, agrège les prédictions
        service: Service de prédiction RUL
    
    Returns:
        Liste des résultats de prédiction
    """
    try:
        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Aucun modèle n'est entraîné. Utilisez POST /api/v1/rul/train pour entraîner les modèles."
            )
        
        if not requests:
            raise HTTPException(status_code=400, detail="Liste de requêtes vide")
        
        results = service.predict_rul_batch(requests, model_name=model_name, use_ensemble=use_ensemble)
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction RUL batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/transfer-learning/load", status_code=200)
async def load_pretrained_model(
    model_name: str,
    model_path: str,
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Charge un modèle pré-entraîné pour le transfer learning
    
    Args:
        model_name: Nom du modèle ('lstm', 'gru', 'tcn')
        model_path: Chemin vers le fichier du modèle
        service: Service de prédiction RUL
    
    Returns:
        Résultat du chargement
    """
    try:
        result = service.transfer_learning_service.load_pretrained_model(
            model_name,
            model_path=model_path
        )
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Modèle pré-entraîné non trouvé ou erreur de chargement"
            )
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_path": model_path,
            "message": f"Modèle {model_name} chargé avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle pré-entraîné: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/transfer-learning/info", status_code=200)
async def get_transfer_learning_info(
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Retourne des informations sur le transfer learning
    
    Args:
        service: Service de prédiction RUL
    
    Returns:
        Informations sur les modèles pré-entraînés
    """
    try:
        info = service.transfer_learning_service.get_pretrained_info()
        return info
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/calibrate", status_code=200)
async def calibrate_models(
    predictions: List[float],
    actuals: List[float],
    method: Optional[str] = None,
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Calibre les modèles RUL avec des données de validation
    
    Args:
        predictions: Liste des prédictions
        actuals: Liste des valeurs réelles
        method: Méthode de calibration ('isotonic', 'platt', 'temperature_scaling')
        service: Service de prédiction RUL
    
    Returns:
        Résultat de la calibration avec métriques
    """
    try:
        import numpy as np
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        result = service.calibration_service.fit_calibration(
            predictions_array,
            actuals_array,
            method=method
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la calibration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.post("/train", response_model=Dict[str, TrainingResult], status_code=200)
async def train_models(
    request: TrainingRequest,
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Entraîne un ou tous les modèles RUL
    
    Args:
        request: Requête d'entraînement avec données
        service: Service de prédiction RUL
    
    Returns:
        Résultat de l'entraînement pour chaque modèle
    """
    try:
        # Préparer les données
        if request.training_data is None or request.target_data is None:
            raise HTTPException(
                status_code=400,
                detail="training_data et target_data sont requis"
            )
        
        import numpy as np
        X_train = np.array(request.training_data)
        y_train = np.array(request.target_data)
        
        if len(X_train) != len(y_train):
            raise HTTPException(
                status_code=400,
                detail="training_data et target_data doivent avoir la même longueur"
            )
        
        # Entraîner tous les modèles ou un modèle spécifique
        if request.model_name:
            # Entraîner un modèle spécifique
            if request.model_name not in service.models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modèle {request.model_name} non disponible"
                )
            
            model = service.models[request.model_name]
            if not hasattr(model, 'train'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Modèle {request.model_name} ne supporte pas l'entraînement"
                )
            
            # Paramètres personnalisés
            train_kwargs = {}
            if 'epochs' in model.train.__code__.co_varnames:
                train_kwargs['epochs'] = request.parameters.get('epochs')
            if 'batch_size' in model.train.__code__.co_varnames:
                train_kwargs['batch_size'] = request.parameters.get('batch_size')
            
            metrics = model.train(
                X_train,
                y_train,
                feature_names=request.feature_names,
                **train_kwargs
            )
            
            results = {
                request.model_name: TrainingResult(
                    model_name=request.model_name,
                    status="success",
                    message="Modèle entraîné avec succès",
                    metrics=metrics,
                    model_version="v1.0"
                )
            }
        else:
            # Entraîner tous les modèles
            training_results = service.train_all_models(
                X_train,
                y_train,
                feature_names=request.feature_names,
                epochs=request.parameters.get('epochs'),
                batch_size=request.parameters.get('batch_size')
            )
            
            results = {}
            for model_name, result in training_results.items():
                if result.get("status") == "success":
                    results[model_name] = TrainingResult(
                        model_name=model_name,
                        status="success",
                        message="Modèle entraîné avec succès",
                        metrics=result.get("metrics", {}),
                        model_version="v1.0"
                    )
                else:
                    results[model_name] = TrainingResult(
                        model_name=model_name,
                        status="error",
                        message=result.get("error", "Erreur inconnue")
                    )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint pour le service de prédiction RUL
    
    Returns:
        Status du service
    """
    return {
        "status": "healthy",
        "service": "prediction-rul",
        "version": "0.1.0"
    }


@router.get("/status", status_code=200)
async def get_status(
    service: RULPredictionService = Depends(get_rul_prediction_service)
):
    """
    Retourne le statut des modèles et de la calibration
    
    Args:
        service: Service de prédiction RUL
    
    Returns:
        Statut de tous les modèles (entraînés ou non) et de la calibration
    """
    try:
        status = service.get_model_status()
        is_ready = service.is_ready()
        calibration_info = service.calibration_service.get_calibration_info()
        
        transfer_learning_info = service.transfer_learning_service.get_pretrained_info()
        
        # Informations MLflow (non-blocking)
        mlflow_info = {}
        try:
            from app.services.mlflow_service import MLflowService
            mlflow_service = MLflowService()
            mlflow_info = mlflow_service.get_service_info()
        except Exception as mlflow_error:
            logger.debug(f"MLflow info skipped: {mlflow_error}")
            mlflow_info = {"enabled": False, "error": "MLflow unavailable"}
        
        return {
            "ready": is_ready,
            "models": status,
            "calibration": calibration_info,
            "transfer_learning": transfer_learning_info,
            "mlflow": mlflow_info
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/", status_code=200)
async def get_rul_predictions(
    asset_id: Optional[str] = None,
    sensor_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Récupère l'historique des prédictions RUL
    
    Args:
        asset_id: Filtrer par asset_id
        sensor_id: Filtrer par sensor_id
        limit: Nombre maximum de résultats
        offset: Offset pour la pagination
    
    Returns:
        Liste des prédictions RUL
    """
    try:
        from app.database.postgresql import PostgreSQLService
        postgresql_service = PostgreSQLService()
        
        result = postgresql_service.get_rul_predictions(
            asset_id=asset_id,
            sensor_id=sensor_id,
            limit=limit,
            offset=offset
        )
        
        # Convert predictions to serializable format
        predictions = []
        for pred in result.get("predictions", []):
            if hasattr(pred, 'dict'):
                pred_dict = pred.dict()
            elif hasattr(pred, '__dict__'):
                pred_dict = {k: v for k, v in pred.__dict__.items() if not k.startswith('_')}
            else:
                pred_dict = dict(pred)
            
            # Ensure timestamp is ISO format string
            if 'timestamp' in pred_dict and hasattr(pred_dict['timestamp'], 'isoformat'):
                pred_dict['timestamp'] = pred_dict['timestamp'].isoformat()
            if 'created_at' in pred_dict and hasattr(pred_dict['created_at'], 'isoformat'):
                pred_dict['created_at'] = pred_dict['created_at'].isoformat()
            
            predictions.append(pred_dict)
        
        return {
            "predictions": predictions,
            "total": result.get("total", 0),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prédictions RUL: {e}", exc_info=True)
        # Return empty list on error (database might not be available)
        return {
            "predictions": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

