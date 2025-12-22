"""
Service principal d'orchestration pour la prédiction RUL
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import os

from app.config import settings

# Lazy imports to avoid slow PyTorch initialization
# These will be imported only when needed
LSTMService = None
GRUService = None
TCNService = None
CalibrationService = None
TransferLearningService = None

# XGBoost optionnel
try:
    from app.services.xgboost_service import XGBoostService
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostService = None

from app.models.rul_data import RULPredictionRequest, RULPredictionResult

logger = logging.getLogger(__name__)


class RULPredictionService:
    """Service principal pour orchestrer les modèles de prédiction RUL"""
    
    def __init__(self):
        """Initialise le service avec tous les modèles"""
        logger.info("Initialisation de RULPredictionService...")
        self.models: Dict[str, Any] = {}
        
        # Lazy import CalibrationService (imports PyTorch)
        global CalibrationService
        if CalibrationService is None:
            try:
                logger.debug("Importing CalibrationService (may take time if PyTorch not preloaded)...")
                from app.services.calibration_service import CalibrationService
                logger.debug("✓ CalibrationService imported")
            except Exception as e:
                logger.warning(f"Erreur lors de l'import CalibrationService: {e}")
                CalibrationService = None
        
        try:
            logger.info("Initialisation Calibration Service...")
            if CalibrationService:
                self.calibration_service = CalibrationService()
                logger.info("✓ Calibration Service initialisé")
            else:
                self.calibration_service = None
                logger.info("Calibration Service non disponible (PyTorch non disponible)")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation Calibration Service: {e}")
            self.calibration_service = None
        
        # Lazy import TransferLearningService (imports PyTorch)
        global TransferLearningService
        if TransferLearningService is None:
            try:
                logger.debug("Importing TransferLearningService (may take time if PyTorch not preloaded)...")
                from app.services.transfer_learning_service import TransferLearningService
                logger.debug("✓ TransferLearningService imported")
            except Exception as e:
                logger.warning(f"Erreur lors de l'import TransferLearningService: {e}")
                TransferLearningService = None
        
        try:
            logger.info("Initialisation Transfer Learning Service...")
            if TransferLearningService:
                self.transfer_learning_service = TransferLearningService()
                logger.info("✓ Transfer Learning Service initialisé")
            else:
                self.transfer_learning_service = None
                logger.info("Transfer Learning Service non disponible (PyTorch non disponible)")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation Transfer Learning Service: {e}")
            self.transfer_learning_service = None
        
        # Initialiser les modèles selon la configuration
        if settings.enable_lstm:
            # Lazy import LSTMService
            global LSTMService
            if LSTMService is None:
                try:
                    logger.debug("Importing LSTMService...")
                    from app.services.lstm_service import LSTMService
                    logger.debug("✓ LSTMService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import LSTMService: {e}")
                    LSTMService = None
            
            if LSTMService:
                try:
                    logger.info("Initialisation LSTM service (peut prendre du temps si PyTorch n'est pas préchargé)...")
                    lstm_service = LSTMService(transfer_learning_service=self.transfer_learning_service)
                    # Appliquer transfer learning si disponible
                    if (settings.transfer_learning_enabled and 
                        self.transfer_learning_service and
                        self.transfer_learning_service.load_pretrained_model("lstm") is not None):
                        logger.info("Transfer learning disponible pour LSTM")
                    self.models["lstm"] = lstm_service
                    logger.info("✓ LSTM Service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation LSTM: {e}")
            else:
                logger.warning("LSTMService non disponible")
        
        if settings.enable_gru:
            # Lazy import GRUService
            global GRUService
            if GRUService is None:
                try:
                    logger.debug("Importing GRUService...")
                    from app.services.gru_service import GRUService
                    logger.debug("✓ GRUService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import GRUService: {e}")
                    GRUService = None
            
            if GRUService:
                try:
                    logger.info("Initialisation GRU service (peut prendre du temps si PyTorch n'est pas préchargé)...")
                    if settings.transfer_learning_enabled and self.transfer_learning_service:
                        self.transfer_learning_service.load_pretrained_model("gru")
                    gru_service = GRUService(transfer_learning_service=self.transfer_learning_service)
                    self.models["gru"] = gru_service
                    logger.info("✓ GRU Service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation GRU: {e}")
            else:
                logger.warning("GRUService non disponible")
        
        if settings.enable_tcn:
            # Lazy import TCNService
            global TCNService
            if TCNService is None:
                try:
                    logger.debug("Importing TCNService...")
                    from app.services.tcn_service import TCNService
                    logger.debug("✓ TCNService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import TCNService: {e}")
                    TCNService = None
            
            if TCNService:
                try:
                    logger.info("Initialisation TCN service (peut prendre du temps si PyTorch n'est pas préchargé)...")
                    if settings.transfer_learning_enabled and self.transfer_learning_service:
                        self.transfer_learning_service.load_pretrained_model("tcn")
                    tcn_service = TCNService(transfer_learning_service=self.transfer_learning_service)
                    self.models["tcn"] = tcn_service
                    logger.info("✓ TCN Service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation TCN: {e}")
            else:
                logger.warning("TCNService non disponible")
        
        if settings.enable_xgboost and XGBOOST_AVAILABLE:
            try:
                logger.info("Initialisation XGBoost service...")
                self.models["xgboost"] = XGBoostService()
                logger.info("✓ XGBoost Service initialisé")
            except Exception as e:
                logger.warning(f"Erreur lors de l'initialisation XGBoost: {e}")
        elif settings.enable_xgboost and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost demandé mais non disponible (module non installé)")
        
        logger.info(f"✓ RULPredictionService initialisé avec {len(self.models)} modèles")
        
        # Charger les modèles sauvegardés si activé
        if settings.auto_load_models:
            logger.info("Tentative de chargement des modèles sauvegardés...")
            loaded_count = self.load_all_models()
            if loaded_count == 0:
                logger.info("Aucun modèle sauvegardé trouvé. Les modèles devront être entraînés.")
            else:
                logger.info(f"✓ {loaded_count} modèle(s) chargé(s) avec succès depuis le disque")
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Entraîne tous les modèles activés
        
        Args:
            X_train: Données d'entraînement
            y_train: Targets d'entraînement
            X_val: Données de validation (optionnel)
            y_val: Targets de validation (optionnel)
            feature_names: Noms des features
            epochs: Nombre d'epochs (override config)
            batch_size: Taille de batch (override config)
        
        Returns:
            Dict avec métriques de chaque modèle
        """
        logger.info(f"Début de l'entraînement de {len(self.models)} modèles")
        
        results = {}
        
        for model_name, model_service in self.models.items():
            try:
                logger.info(f"Entraînement du modèle {model_name}...")
                
                # Entraîner le modèle
                if hasattr(model_service, 'train'):
                    # Appliquer transfer learning si disponible
                    if (settings.transfer_learning_enabled and 
                        model_name in self.transfer_learning_service.pretrained_models):
                        # Le transfer learning sera appliqué dans la méthode train du service
                        logger.info(f"Transfer learning sera appliqué pour {model_name}")
                    
                    # Passer epochs et batch_size si disponibles
                    train_kwargs = {}
                    if epochs is not None and 'epochs' in model_service.train.__code__.co_varnames:
                        train_kwargs['epochs'] = epochs
                    if batch_size is not None and 'batch_size' in model_service.train.__code__.co_varnames:
                        train_kwargs['batch_size'] = batch_size
                    
                    metrics = model_service.train(
                        X_train,
                        y_train,
                        X_val=X_val,
                        y_val=y_val,
                        feature_names=feature_names,
                        **train_kwargs
                    )
                    
                    results[model_name] = {
                        "status": "success",
                        "metrics": metrics
                    }
                    logger.info(f"Modèle {model_name} entraîné avec succès")
                else:
                    results[model_name] = {
                        "status": "error",
                        "error": "Méthode train() non disponible"
                    }
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}", exc_info=True)
                results[model_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        logger.info(f"Entraînement terminé. {sum(1 for r in results.values() if r.get('status') == 'success')}/{len(results)} modèles entraînés")
        
        # Sauvegarder les modèles entraînés
        saved_count = self.save_all_models()
        if saved_count > 0:
            logger.info(f"{saved_count} modèles sauvegardés sur disque")
        
        return results
    
    def predict_rul(
        self,
        request: RULPredictionRequest,
        model_name: Optional[str] = None,
        use_ensemble: bool = True
    ) -> RULPredictionResult:
        """
        Prédit la RUL pour un actif
        
        Args:
            request: Requête de prédiction
            model_name: Nom du modèle à utiliser (si None, utilise ensemble)
            use_ensemble: Si True, agrège les prédictions de tous les modèles
        
        Returns:
            Résultat de prédiction avec intervalle de confiance
        """
        if not self.is_ready():
            raise RuntimeError("Aucun modèle n'est entraîné. Appelez train_all_models() d'abord.")
        
        # Convertir features en array numpy
        feature_values = list(request.features.values())
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # Aligner les features avec la taille attendue par les modèles
        # Les modèles peuvent avoir été entraînés avec différentes tailles (88, 90, 92)
        # On aligne en prenant la taille attendue du premier modèle disponible
        expected_size = None
        for name, model in self.models.items():
            if self._is_model_trained(model):
                # XGBoost models
                if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
                    expected_size = model.model.n_features_in_
                    break
                # LSTM/GRU/TCN models
                elif hasattr(model, 'input_size'):
                    expected_size = model.input_size
                    break
                # Direct XGBoost model
                elif hasattr(model, 'n_features_in_'):
                    expected_size = model.n_features_in_
                    break
        
        if expected_size is None:
            # Si on ne peut pas déterminer, utiliser la taille actuelle
            expected_size = feature_array.shape[1]
            logger.warning(f"Could not determine expected feature size, using current size: {expected_size}")
        
        # Aligner les features: padding avec 0 si trop court, troncature si trop long
        current_size = feature_array.shape[1]
        if current_size < expected_size:
            # Padding avec des zéros
            padding = np.zeros((1, expected_size - current_size))
            feature_array = np.hstack([feature_array, padding])
            logger.debug(f"Features padded from {current_size} to {expected_size}")
        elif current_size > expected_size:
            # Troncature
            feature_array = feature_array[:, :expected_size]
            logger.debug(f"Features truncated from {current_size} to {expected_size}")
        
        # Si sequence_data fourni, l'utiliser (optional field)
        sequence_data = getattr(request, 'sequence_data', None)
        if sequence_data:
            sequence_array = np.array([
                list(seq.values()) for seq in sequence_data
            ])
            # Aligner aussi les séquences
            if sequence_array.shape[2] != expected_size:
                if sequence_array.shape[2] < expected_size:
                    padding = np.zeros((sequence_array.shape[0], sequence_array.shape[1], expected_size - sequence_array.shape[2]))
                    sequence_array = np.concatenate([sequence_array, padding], axis=2)
                else:
                    sequence_array = sequence_array[:, :, :expected_size]
        else:
            # Créer une séquence à partir des features actuelles
            sequence_array = feature_array
        
        predictions = {}
        model_scores = {}
        
        # Prédictions avec chaque modèle entraîné
        trained_models = {name: model for name, model in self.models.items() if self._is_model_trained(model)}
        
        if not trained_models:
            raise RuntimeError("Aucun modèle entraîné disponible")
        
        # Si model_name spécifié, utiliser seulement ce modèle
        if model_name and model_name in trained_models:
            models_to_use = {model_name: trained_models[model_name]}
        elif model_name:
            raise ValueError(f"Modèle {model_name} non disponible ou non entraîné")
        else:
            models_to_use = trained_models
        
        for name, model in models_to_use.items():
            try:
                # Try prediction first
                pred = model.predict(sequence_array)
                predictions[name] = float(pred[0]) if len(pred) > 0 else 0.0
                model_scores[name] = predictions[name]
            except ValueError as e:
                # Handle feature shape mismatch - align and retry
                if "Feature shape mismatch" in str(e) or "feature" in str(e).lower():
                    # Extract expected size from error or model
                    model_expected_size = None
                    if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
                        model_expected_size = model.model.n_features_in_
                    elif hasattr(model, 'input_size'):
                        model_expected_size = model.input_size
                    elif hasattr(model, 'n_features_in_'):
                        model_expected_size = model.n_features_in_
                    
                    if model_expected_size:
                        # Align sequence_array to expected size
                        current_size = sequence_array.shape[-1]
                        if current_size < model_expected_size:
                            # Pad with zeros
                            if len(sequence_array.shape) == 2:
                                padding = np.zeros((sequence_array.shape[0], model_expected_size - current_size))
                                sequence_array = np.hstack([sequence_array, padding])
                            else:  # 3D
                                padding = np.zeros((sequence_array.shape[0], sequence_array.shape[1], model_expected_size - current_size))
                                sequence_array = np.concatenate([sequence_array, padding], axis=2)
                            logger.debug(f"Aligned features for {name}: {current_size} -> {model_expected_size} (padded)")
                        elif current_size > model_expected_size:
                            # Truncate
                            if len(sequence_array.shape) == 2:
                                sequence_array = sequence_array[:, :model_expected_size]
                            else:  # 3D
                                sequence_array = sequence_array[:, :, :model_expected_size]
                            logger.debug(f"Aligned features for {name}: {current_size} -> {model_expected_size} (truncated)")
                        
                        # Retry prediction
                        try:
                            pred = model.predict(sequence_array)
                            predictions[name] = float(pred[0]) if len(pred) > 0 else 0.0
                            model_scores[name] = predictions[name]
                        except Exception as e2:
                            logger.warning(f"Erreur lors de la prédiction avec {name} après alignement: {e2}")
                            continue
                    else:
                        logger.warning(f"Erreur lors de la prédiction avec {name}: {e} (could not determine expected size)")
                        continue
                else:
                    logger.warning(f"Erreur lors de la prédiction avec {name}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction avec {name}: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("Aucune prédiction réussie")
        
        # Agrégation des prédictions
        if use_ensemble and len(predictions) > 1:
            # Moyenne pondérée (poids égaux pour l'instant)
            predictions_array = np.array(list(predictions.values()))
            final_rul = float(np.mean(predictions_array))
            
            # Calcul de l'incertitude avec le service de calibration
            if self.calibration_service:
                uncertainty, confidence_interval_lower, confidence_interval_upper = \
                    self.calibration_service.compute_uncertainty(
                        predictions_array.reshape(-1, 1),
                        method="std"
                    )
            else:
                # Fallback si calibration service n'est pas disponible
                std_dev = float(np.std(predictions_array))
                uncertainty = std_dev
                confidence_interval_lower = final_rul - 1.96 * std_dev
                confidence_interval_upper = final_rul + 1.96 * std_dev
            uncertainty = float(uncertainty[0])
            confidence_interval_lower = float(confidence_interval_lower[0])
            confidence_interval_upper = float(confidence_interval_upper[0])
            
            # Appliquer calibration si disponible
            if self.calibration_service and self.calibration_service.is_calibrated:
                calibrated_rul = self.calibration_service.calibrate_predictions(
                    np.array([final_rul])
                )
                final_rul = float(calibrated_rul[0])
                # Recalculer l'intervalle de confiance avec la prédiction calibrée
                confidence_interval_lower, confidence_interval_upper = \
                    self.calibration_service.compute_confidence_interval(
                        final_rul, uncertainty, confidence_level=0.95
                    )
            
            confidence_level = 0.95
            model_used = "ensemble"
        else:
            # Utiliser la première prédiction disponible
            model_used = list(predictions.keys())[0]
            final_rul = predictions[model_used]
            
            # Appliquer calibration si disponible
            if self.calibration_service and self.calibration_service.is_calibrated:
                calibrated_rul = self.calibration_service.calibrate_predictions(
                    np.array([final_rul])
                )
                final_rul = float(calibrated_rul[0])
            
            # Calculer l'incertitude
            if self.calibration_service:
                uncertainty, confidence_interval_lower, confidence_interval_upper = \
                    self.calibration_service.compute_uncertainty(
                        np.array([final_rul]),
                        method="std"
                    )
            else:
                # Fallback si calibration service n'est pas disponible
                uncertainty = 0.1 * abs(final_rul)  # 10% uncertainty
                confidence_interval_lower = final_rul - 1.96 * uncertainty
                confidence_interval_upper = final_rul + 1.96 * uncertainty
            uncertainty = float(uncertainty[0])
            confidence_interval_lower = float(confidence_interval_lower[0])
            confidence_interval_upper = float(confidence_interval_upper[0])
            
            confidence_level = 0.95
        
        # S'assurer que RUL est positive
        final_rul = max(0.0, final_rul)
        confidence_interval_lower = max(0.0, confidence_interval_lower)
        confidence_interval_upper = max(0.0, confidence_interval_upper)
        
        return RULPredictionResult(
            asset_id=request.asset_id,
            sensor_id=request.sensor_id,
            timestamp=request.timestamp or datetime.now(timezone.utc),
            rul_prediction=float(final_rul),
            confidence_interval_lower=float(confidence_interval_lower),
            confidence_interval_upper=float(confidence_interval_upper),
            confidence_level=confidence_level,
            uncertainty=float(uncertainty),
            model_used=model_used,
            model_scores=model_scores,
            features=request.features,
            metadata=request.metadata or {}
        )
    
    def predict_rul_batch(
        self,
        requests: List[RULPredictionRequest],
        model_name: Optional[str] = None,
        use_ensemble: bool = True
    ) -> List[RULPredictionResult]:
        """
        Prédit la RUL pour plusieurs actifs (batch)
        
        Args:
            requests: Liste de requêtes
            model_name: Nom du modèle à utiliser
            use_ensemble: Si True, agrège les prédictions
        
        Returns:
            Liste des résultats de prédiction
        """
        results = []
        
        for request in requests:
            try:
                result = self.predict_rul(request, model_name, use_ensemble)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour {request.asset_id}: {e}", exc_info=True)
                # Créer un résultat d'erreur
                results.append(RULPredictionResult(
                    asset_id=request.asset_id,
                    sensor_id=request.sensor_id,
                    timestamp=request.timestamp or datetime.now(timezone.utc),
                    rul_prediction=0.0,
                    confidence_interval_lower=0.0,
                    confidence_interval_upper=0.0,
                    confidence_level=0.95,
                    uncertainty=0.0,
                    model_used="error",
                    model_scores={},
                    features=request.features,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def _is_model_trained(self, model: Any) -> bool:
        """Vérifie si un modèle est entraîné"""
        return hasattr(model, 'is_trained') and model.is_trained
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne le statut de tous les modèles
        
        Returns:
            Dict avec statut de chaque modèle
        """
        status = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'get_model_info'):
                    info = model.get_model_info()
                    status[name] = {
                        "available": True,
                        "trained": self._is_model_trained(model),
                        **info
                    }
                else:
                    status[name] = {
                        "available": True,
                        "trained": self._is_model_trained(model)
                    }
            except Exception as e:
                status[name] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
    def is_ready(self) -> bool:
        """
        Vérifie si au moins un modèle est entraîné et prêt
        
        Returns:
            True si au moins un modèle est prêt
        """
        return any(self._is_model_trained(model) for model in self.models.values())
    
    def get_best_model(self) -> Optional[str]:
        """
        Retourne le nom du meilleur modèle basé sur les métriques
        
        Returns:
            Nom du meilleur modèle ou None
        """
        # Pour l'instant, retourner le premier modèle entraîné
        # Plus tard, on pourra utiliser les métriques de validation
        for name, model in self.models.items():
            if self._is_model_trained(model):
                return name
        return None
    
    def _get_model_path(self, model_name: str) -> Path:
        """
        Retourne le chemin de sauvegarde pour un modèle
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Chemin Path pour le modèle
        """
        # Use absolute path relative to the service directory
        # Get the directory where this service is located
        service_dir = Path(__file__).parent.parent.parent  # app/services -> app -> service root
        models_dir = service_dir / settings.models_save_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Models directory: {models_dir.absolute()}")
        
        # Extension selon le type de modèle
        if model_name in ['lstm', 'gru', 'tcn']:
            extension = '.pth'
        elif model_name == 'xgboost':
            extension = '.model'  # XGBoost format (peut être JSON ou binaire)
        else:
            extension = '.pkl'
        
        return models_dir / f"{model_name}_model{extension}"
    
    def save_all_models(self) -> int:
        """
        Sauvegarde tous les modèles entraînés sur disque
        
        Returns:
            Nombre de modèles sauvegardés avec succès
        """
        saved_count = 0
        
        for model_name, model_service in self.models.items():
            if not self._is_model_trained(model_service):
                logger.debug(f"Modèle {model_name} non entraîné, ignoré pour la sauvegarde")
                continue
            
            if not hasattr(model_service, 'save_model'):
                logger.warning(f"Modèle {model_name} n'a pas de méthode save_model()")
                continue
            
            try:
                model_path = self._get_model_path(model_name)
                if model_service.save_model(str(model_path)):
                    saved_count += 1
                    logger.info(f"✓ Modèle {model_name} sauvegardé dans {model_path}")
                else:
                    logger.warning(f"Échec de la sauvegarde du modèle {model_name}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du modèle {model_name}: {e}", exc_info=True)
        
        return saved_count
    
    def load_all_models(self) -> int:
        """
        Charge tous les modèles sauvegardés depuis le disque
        
        Returns:
            Nombre de modèles chargés avec succès
        """
        if not settings.auto_load_models:
            logger.debug("Auto-load des modèles désactivé")
            return 0
        
        loaded_count = 0
        
        for model_name, model_service in self.models.items():
            if not hasattr(model_service, 'load_model'):
                logger.debug(f"Modèle {model_name} n'a pas de méthode load_model()")
                continue
            
            try:
                model_path = self._get_model_path(model_name)
                if model_path.exists():
                    if model_service.load_model(str(model_path)):
                        loaded_count += 1
                        logger.info(f"✓ Modèle {model_name} chargé depuis {model_path}")
                    else:
                        logger.warning(f"Échec du chargement du modèle {model_name}")
                else:
                    logger.debug(f"Modèle {model_name} non trouvé à {model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}", exc_info=True)
        
        if loaded_count > 0:
            logger.info(f"{loaded_count} modèles chargés depuis le disque")
        else:
            logger.info("Aucun modèle sauvegardé trouvé sur le disque")
        
        return loaded_count

