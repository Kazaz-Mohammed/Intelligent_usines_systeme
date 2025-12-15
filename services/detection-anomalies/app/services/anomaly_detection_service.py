"""
Service principal de détection d'anomalies - Orchestration des modèles
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid

# Lazy imports to avoid slow PyTorch initialization
# These will be imported only when needed
IsolationForestService = None
OneClassSVMService = None
LSTMAutoencoderService = None
from app.models.anomaly_data import (
    AnomalyDetectionRequest,
    AnomalyDetectionResult,
    AnomalyScore,
    CriticalityLevel
)
from app.config import settings

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """Service principal pour la détection d'anomalies orchestrant tous les modèles"""
    
    def __init__(self):
        """Initialise le service de détection d'anomalies"""
        # Declare all globals FIRST before any use
        global IsolationForestService
        global OneClassSVMService
        global LSTMAutoencoderService
        
        import sys
        print("Initialisation de AnomalyDetectionService...", file=sys.stderr)
        logger.info("Initialisation de AnomalyDetectionService...")
        self.isolation_forest_service: Optional[IsolationForestService] = None
        self.one_class_svm_service: Optional[OneClassSVMService] = None
        self.lstm_autoencoder_service: Optional[LSTMAutoencoderService] = None
        
        # Initialiser les services selon la configuration
        if settings.enable_isolation_forest:
            print("  Initialisation Isolation Forest...", file=sys.stderr)
            # Lazy import IsolationForestService
            if IsolationForestService is None:
                try:
                    logger.debug("Importing IsolationForestService...")
                    from app.services.isolation_forest_service import IsolationForestService
                    logger.debug("✓ IsolationForestService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import IsolationForestService: {e}")
                    IsolationForestService = None
            
            if IsolationForestService:
                try:
                    logger.info("Initialisation Isolation Forest service...")
                    self.isolation_forest_service = IsolationForestService()
                    print("  ✓ Isolation Forest service initialisé", file=sys.stderr)
                    logger.info("✓ Isolation Forest service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation Isolation Forest: {e}")
            else:
                logger.warning("IsolationForestService non disponible")
        
        if settings.enable_one_class_svm:
            print("  Initialisation One-Class SVM...", file=sys.stderr)
            # Lazy import OneClassSVMService
            if OneClassSVMService is None:
                try:
                    logger.debug("Importing OneClassSVMService...")
                    from app.services.one_class_svm_service import OneClassSVMService
                    logger.debug("✓ OneClassSVMService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import OneClassSVMService: {e}")
                    OneClassSVMService = None
            
            if OneClassSVMService:
                try:
                    logger.info("Initialisation One-Class SVM service...")
                    self.one_class_svm_service = OneClassSVMService()
                    print("  ✓ One-Class SVM service initialisé", file=sys.stderr)
                    logger.info("✓ One-Class SVM service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation One-Class SVM: {e}")
            else:
                logger.warning("OneClassSVMService non disponible")
        
        if settings.enable_lstm_autoencoder:
            print("  Initialisation LSTM Autoencoder...", file=sys.stderr)
            # Lazy import LSTMAutoencoderService (imports PyTorch - can be slow)
            if LSTMAutoencoderService is None:
                try:
                    logger.debug("Importing LSTMAutoencoderService (may take time if PyTorch not preloaded)...")
                    from app.services.lstm_autoencoder_service import LSTMAutoencoderService
                    logger.debug("✓ LSTMAutoencoderService imported")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'import LSTMAutoencoderService: {e}")
                    LSTMAutoencoderService = None
            
            if LSTMAutoencoderService:
                try:
                    logger.info("Initialisation LSTM Autoencoder service (peut prendre du temps si PyTorch n'est pas préchargé)...")
                    self.lstm_autoencoder_service = LSTMAutoencoderService()
                    logger.info("✓ LSTM Autoencoder service initialisé")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'initialisation LSTM Autoencoder: {e}")
            else:
                logger.warning("LSTMAutoencoderService non disponible")
        
        print("✓ AnomalyDetectionService initialisé", file=sys.stderr)
        logger.info("AnomalyDetectionService initialisé")
        
        # Auto-charger les modèles sauvegardés si configuré
        if settings.auto_load_models:
            loaded = self.load_all_models()
            if loaded > 0:
                logger.info(f"✓ {loaded} modèle(s) de détection d'anomalies chargé(s) automatiquement")
            else:
                logger.info("Aucun modèle de détection d'anomalies sauvegardé trouvé")
    
    def train_all_models(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entraîne tous les modèles activés
        
        Args:
            X: Données d'entraînement (n_samples, n_features)
            feature_names: Noms des features (optionnel)
        
        Returns:
            Dict avec les métriques de tous les modèles
        """
        results = {}
        
        if self.isolation_forest_service is not None:
            try:
                logger.info("Entraînement Isolation Forest...")
                metrics = self.isolation_forest_service.train(X, feature_names)
                results["isolation_forest"] = {
                    "status": "success",
                    "metrics": metrics
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement Isolation Forest: {e}", exc_info=True)
                results["isolation_forest"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        if self.one_class_svm_service is not None:
            try:
                logger.info("Entraînement One-Class SVM...")
                metrics = self.one_class_svm_service.train(X, feature_names)
                results["one_class_svm"] = {
                    "status": "success",
                    "metrics": metrics
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement One-Class SVM: {e}", exc_info=True)
                results["one_class_svm"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        if self.lstm_autoencoder_service is not None:
            try:
                logger.info("Entraînement LSTM Autoencoder...")
                metrics = self.lstm_autoencoder_service.train(X, feature_names)
                results["lstm_autoencoder"] = {
                    "status": "success",
                    "metrics": metrics
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement LSTM Autoencoder: {e}", exc_info=True)
                results["lstm_autoencoder"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Sauvegarder tous les modèles après l'entraînement
        saved_count = self.save_all_models()
        results["models_saved"] = saved_count
        logger.info(f"✓ {saved_count} modèle(s) sauvegardé(s) après entraînement")
        
        return results
    
    def _aggregate_scores(self, scores: List[AnomalyScore]) -> float:
        """
        Agrège les scores de plusieurs modèles
        
        Args:
            scores: Liste des scores des différents modèles
        
        Returns:
            Score final agrégé (0-1)
        """
        if not scores:
            return 0.0
        
        # Méthode : moyenne pondérée (tous les modèles ont le même poids pour l'instant)
        # On peut améliorer avec des poids basés sur la performance
        total_score = sum(score.score for score in scores)
        return total_score / len(scores)
    
    def _determine_criticality(self, final_score: float, thresholds: Optional[Dict[str, float]] = None) -> CriticalityLevel:
        """
        Détermine le niveau de criticité basé sur le score final
        
        Args:
            final_score: Score final agrégé (0-1)
            thresholds: Seuils personnalisés (optionnel)
        
        Returns:
            Niveau de criticité
        """
        if thresholds is None:
            # Seuils par défaut
            thresholds = {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3
            }
        
        if final_score >= thresholds.get("critical", 0.9):
            return CriticalityLevel.CRITICAL
        elif final_score >= thresholds.get("high", 0.7):
            return CriticalityLevel.HIGH
        elif final_score >= thresholds.get("medium", 0.5):
            return CriticalityLevel.MEDIUM
        else:
            return CriticalityLevel.LOW
    
    def detect_anomaly(
        self,
        request: AnomalyDetectionRequest,
        thresholds: Optional[Dict[str, float]] = None
    ) -> AnomalyDetectionResult:
        """
        Détecte une anomalie en utilisant tous les modèles activés
        
        Args:
            request: Requête de détection d'anomalie
            thresholds: Seuils personnalisés par modèle (optionnel)
        
        Returns:
            Résultat de la détection avec scores de tous les modèles
        """
        scores = []
        
        # Isolation Forest
        if self.isolation_forest_service is not None and self.isolation_forest_service.is_trained:
            try:
                threshold = thresholds.get("isolation_forest") if thresholds else None
                result = self.isolation_forest_service.detect_anomaly(
                    request.features,
                    threshold=threshold
                )
                scores.append(AnomalyScore(
                    score=result["score"],
                    model_name="isolation_forest",
                    threshold=result["threshold"],
                    is_anomaly=result["is_anomaly"]
                ))
            except Exception as e:
                logger.warning(f"Erreur lors de la détection avec Isolation Forest: {e}")
        
        # One-Class SVM
        if self.one_class_svm_service is not None and self.one_class_svm_service.is_trained:
            try:
                threshold = thresholds.get("one_class_svm") if thresholds else None
                result = self.one_class_svm_service.detect_anomaly(
                    request.features,
                    threshold=threshold
                )
                scores.append(AnomalyScore(
                    score=result["score"],
                    model_name="one_class_svm",
                    threshold=result["threshold"],
                    is_anomaly=result["is_anomaly"]
                ))
            except Exception as e:
                logger.warning(f"Erreur lors de la détection avec One-Class SVM: {e}")
        
        # LSTM Autoencoder
        if self.lstm_autoencoder_service is not None and self.lstm_autoencoder_service.is_trained:
            try:
                threshold = thresholds.get("lstm_autoencoder") if thresholds else None
                result = self.lstm_autoencoder_service.detect_anomaly(
                    request.features,
                    threshold=threshold
                )
                scores.append(AnomalyScore(
                    score=result["score"],
                    model_name="lstm_autoencoder",
                    threshold=result["threshold"],
                    is_anomaly=result["is_anomaly"]
                ))
            except Exception as e:
                logger.warning(f"Erreur lors de la détection avec LSTM Autoencoder: {e}")
        
        # Agréger les scores
        final_score = self._aggregate_scores(scores)
        
        # Déterminer si c'est une anomalie
        is_anomaly = final_score >= settings.anomaly_score_threshold
        
        # Déterminer la criticité
        criticality = self._determine_criticality(final_score)
        
        # Créer le résultat
        result = AnomalyDetectionResult(
            asset_id=request.asset_id,
            sensor_id=request.sensor_id,
            timestamp=request.timestamp or datetime.now(timezone.utc),
            scores=scores,
            final_score=final_score,
            is_anomaly=is_anomaly,
            criticality=criticality,
            features=request.features,
            metadata=request.metadata
        )
        
        return result
    
    def detect_anomalies_batch(
        self,
        requests: List[AnomalyDetectionRequest],
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[AnomalyDetectionResult]:
        """
        Détecte des anomalies pour un batch de requêtes
        
        Args:
            requests: Liste de requêtes de détection
            thresholds: Seuils personnalisés (optionnel)
        
        Returns:
            Liste de résultats de détection
        """
        results = []
        for request in requests:
            result = self.detect_anomaly(request, thresholds)
            results.append(result)
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Retourne le statut de tous les modèles
        
        Returns:
            Dict avec le statut de chaque modèle
        """
        status = {}
        
        if self.isolation_forest_service is not None:
            status["isolation_forest"] = self.isolation_forest_service.get_model_info()
        
        if self.one_class_svm_service is not None:
            status["one_class_svm"] = self.one_class_svm_service.get_model_info()
        
        if self.lstm_autoencoder_service is not None:
            status["lstm_autoencoder"] = self.lstm_autoencoder_service.get_model_info()
        
        return status
    
    def is_ready(self) -> bool:
        """
        Vérifie si au moins un modèle est entraîné et prêt
        
        Returns:
            True si au moins un modèle est prêt
        """
        ready = False
        
        if self.isolation_forest_service is not None:
            ready = ready or self.isolation_forest_service.is_trained
        
        if self.one_class_svm_service is not None:
            ready = ready or self.one_class_svm_service.is_trained
        
        if self.lstm_autoencoder_service is not None:
            ready = ready or self.lstm_autoencoder_service.is_trained
        
        return ready
    
    def _get_model_path(self, model_name: str) -> Path:
        """
        Retourne le chemin de sauvegarde pour un modèle
        
        Args:
            model_name: Nom du modèle (isolation_forest, one_class_svm, lstm_autoencoder)
        
        Returns:
            Chemin absolu du fichier modèle
        """
        # Construire un chemin absolu relatif au répertoire racine du service
        service_root = Path(__file__).parent.parent.parent.resolve()
        models_dir = service_root / settings.models_save_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Extension selon le type de modèle
        if model_name == 'lstm_autoencoder':
            extension = '.pth'  # PyTorch format
        else:
            extension = '.pkl'  # Joblib/pickle format
        
        return models_dir / f"{model_name}_model{extension}"
    
    def save_all_models(self) -> int:
        """
        Sauvegarde tous les modèles entraînés sur disque
        
        Returns:
            Nombre de modèles sauvegardés avec succès
        """
        saved_count = 0
        
        # Sauvegarder Isolation Forest
        if self.isolation_forest_service is not None and self.isolation_forest_service.is_trained:
            try:
                path = self._get_model_path("isolation_forest")
                if self.isolation_forest_service.save_model(path):
                    saved_count += 1
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde Isolation Forest: {e}")
        
        # Sauvegarder One-Class SVM
        if self.one_class_svm_service is not None and self.one_class_svm_service.is_trained:
            try:
                path = self._get_model_path("one_class_svm")
                if self.one_class_svm_service.save_model(path):
                    saved_count += 1
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde One-Class SVM: {e}")
        
        # Sauvegarder LSTM Autoencoder (si implémenté)
        if self.lstm_autoencoder_service is not None and self.lstm_autoencoder_service.is_trained:
            try:
                path = self._get_model_path("lstm_autoencoder")
                if hasattr(self.lstm_autoencoder_service, 'save_model'):
                    if self.lstm_autoencoder_service.save_model(path):
                        saved_count += 1
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde LSTM Autoencoder: {e}")
        
        logger.info(f"Sauvegarde terminée: {saved_count} modèle(s) sauvegardé(s)")
        return saved_count
    
    def load_all_models(self) -> int:
        """
        Charge tous les modèles depuis le disque
        
        Returns:
            Nombre de modèles chargés avec succès
        """
        loaded_count = 0
        
        # Charger Isolation Forest
        if self.isolation_forest_service is not None:
            try:
                path = self._get_model_path("isolation_forest")
                if self.isolation_forest_service.load_model(path):
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Erreur lors du chargement Isolation Forest: {e}")
        
        # Charger One-Class SVM
        if self.one_class_svm_service is not None:
            try:
                path = self._get_model_path("one_class_svm")
                if self.one_class_svm_service.load_model(path):
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Erreur lors du chargement One-Class SVM: {e}")
        
        # Charger LSTM Autoencoder (si implémenté)
        if self.lstm_autoencoder_service is not None:
            try:
                path = self._get_model_path("lstm_autoencoder")
                if hasattr(self.lstm_autoencoder_service, 'load_model'):
                    if self.lstm_autoencoder_service.load_model(path):
                        loaded_count += 1
            except Exception as e:
                logger.error(f"Erreur lors du chargement LSTM Autoencoder: {e}")
        
        logger.info(f"Chargement terminé: {loaded_count} modèle(s) chargé(s)")
        return loaded_count

