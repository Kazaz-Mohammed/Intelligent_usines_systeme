"""
Service Isolation Forest pour la détection d'anomalies
"""
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib
from pyod.models.iforest import IForest
from app.config import settings
# Lazy import MLflowService to avoid slow initialization
MLflowService = None

logger = logging.getLogger(__name__)


class IsolationForestService:
    """Service pour la détection d'anomalies avec Isolation Forest"""
    
    def __init__(self):
        """Initialise le service Isolation Forest"""
        self.model: Optional[IForest] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False
        # Lazy initialization of MLflowService (can be slow)
        self.mlflow_service: Optional[MLflowService] = None
        
        # Paramètres depuis la config
        self.contamination = settings.isolation_forest_contamination
        self.n_estimators = settings.isolation_forest_n_estimators
        self.max_samples = settings.isolation_forest_max_samples
        
        logger.info(
            f"IsolationForestService initialisé avec contamination={self.contamination}, "
            f"n_estimators={self.n_estimators}, max_samples={self.max_samples}"
        )
    
    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Entraîne le modèle Isolation Forest
        
        Args:
            X: Données d'entraînement (n_samples, n_features)
            feature_names: Noms des features (optionnel)
        
        Returns:
            Dict avec les métriques d'entraînement
        """
        try:
            logger.info(f"Entraînement Isolation Forest sur {X.shape[0]} échantillons, {X.shape[1]} features")
            
            # Initialiser le modèle
            self.model = IForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=42,
                n_jobs=-1
            )
            
            # Entraîner
            self.model.fit(X)
            
            # Sauvegarder les noms de features
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            self.is_trained = True
            
            # Calculer quelques métriques sur les données d'entraînement
            train_scores = self.model.decision_scores_
            train_predictions = self.model.labels_
            
            n_anomalies = np.sum(train_predictions == 1)
            anomaly_rate = n_anomalies / len(train_predictions)
            
            metrics = {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_anomalies_detected": int(n_anomalies),
                "anomaly_rate": float(anomaly_rate),
                "mean_score": float(np.mean(train_scores)),
                "std_score": float(np.std(train_scores)),
                "min_score": float(np.min(train_scores)),
                "max_score": float(np.max(train_scores))
            }
            
            # Lazy initialization of MLflowService (non-blocking)
            # Skip MLflow if it's not available or causes delays
            try:
                if self.mlflow_service is None:
                    global MLflowService
                    if MLflowService is None:
                        from app.services.mlflow_service import MLflowService
                    self.mlflow_service = MLflowService()
                
                # Logging MLflow (non-blocking, skip if it fails)
                if self.mlflow_service and self.mlflow_service.enabled:
                    run = self.mlflow_service.start_run(run_name=f"isolation_forest_{X.shape[0]}_samples")
                    if run:
                        try:
                            # Log parameters
                            self.mlflow_service.log_params({
                                "model_type": "isolation_forest",
                                "contamination": self.contamination,
                                "n_estimators": self.n_estimators,
                                "max_samples": str(self.max_samples),
                                "n_samples": X.shape[0],
                                "n_features": X.shape[1]
                            })
                            
                            # Log metrics
                            self.mlflow_service.log_metrics({
                                "n_anomalies_detected": metrics["n_anomalies_detected"],
                                "anomaly_rate": metrics["anomaly_rate"],
                                "mean_score": metrics["mean_score"],
                                "std_score": metrics["std_score"]
                            })
                            
                            # Log model (skip if it's slow)
                            try:
                                self.mlflow_service.log_model_sklearn(
                                    self.model,
                                    artifact_path="isolation_forest_model",
                                    registered_model_name="isolation_forest"
                                )
                            except Exception as model_log_error:
                                logger.debug(f"MLflow model logging skipped: {model_log_error}")
                        finally:
                            self.mlflow_service.end_run()
            except Exception as mlflow_error:
                # MLflow is optional - don't fail training if it doesn't work
                logger.debug(f"MLflow logging skipped: {mlflow_error}")
            
            logger.info(f"Isolation Forest entraîné avec succès. Métriques: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement Isolation Forest: {e}", exc_info=True)
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les labels d'anomalie (0=normal, 1=anomalie)
        
        Args:
            X: Données à prédire (n_samples, n_features)
        
        Returns:
            Labels (0 ou 1)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction Isolation Forest: {e}", exc_info=True)
            raise
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores d'anomalie (plus élevé = plus anormal)
        
        Args:
            X: Données à scorer (n_samples, n_features)
        
        Returns:
            Scores d'anomalie
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            scores = self.model.decision_function(X)
            # Normaliser les scores entre 0 et 1
            # Isolation Forest retourne des scores négatifs pour les anomalies
            # On inverse et normalise
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores_normalized
        except Exception as e:
            logger.error(f"Erreur lors du scoring Isolation Forest: {e}", exc_info=True)
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités d'anomalie (0=normal, 1=anomalie)
        
        Args:
            X: Données à scorer (n_samples, n_features)
        
        Returns:
            Probabilités [prob_normal, prob_anomaly] pour chaque échantillon
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            scores = self.model.decision_function(X)
            # Normaliser les scores entre 0 et 1
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Probabilité d'anomalie = score normalisé
            prob_anomaly = scores_normalized
            prob_normal = 1 - prob_anomaly
            
            # Retourner [prob_normal, prob_anomaly]
            return np.column_stack([prob_normal, prob_anomaly])
        except Exception as e:
            logger.error(f"Erreur lors du calcul de probabilité Isolation Forest: {e}", exc_info=True)
            raise
    
    def detect_anomaly(self, features: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Détecte une anomalie à partir d'un dictionnaire de features
        
        Args:
            features: Dictionnaire {nom_feature: valeur}
            threshold: Seuil personnalisé (optionnel, utilise contamination par défaut)
        
        Returns:
            Dict avec score, is_anomaly, etc.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            # Convertir le dictionnaire en array numpy
            if self.feature_names is None:
                raise ValueError("Les noms de features ne sont pas définis")
            
            # Créer un array avec les features dans le bon ordre
            X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            
            # Get raw decision function score
            # Isolation Forest: negative = anomaly, positive = normal
            raw_score = self.model.decision_function(X)[0]
            
            # Convert to anomaly score (0-1 scale, higher = more anomalous)
            # Use sigmoid-like transformation centered around 0
            # Negative raw_score -> high anomaly score
            score = 1.0 / (1.0 + np.exp(raw_score * 5))  # Scale factor of 5 for sensitivity
            
            # Predict (-1 = anomaly, 1 = normal)
            prediction = self.predict(X)[0]
            
            # Utiliser le seuil fourni ou celui du modèle
            if threshold is None:
                threshold = self.contamination
            
            # FIX: prediction == -1 means anomaly (not 1!)
            is_anomaly = score >= threshold or prediction == -1
            
            logger.debug(f"Isolation Forest: raw_score={raw_score:.4f}, score={score:.4f}, prediction={prediction}, is_anomaly={is_anomaly}")
            
            return {
                "score": float(score),
                "is_anomaly": bool(is_anomaly),
                "threshold": float(threshold),
                "prediction": int(prediction)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie: {e}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "model_type": "isolation_forest",
            "is_trained": self.is_trained,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names
        }
    
    def save_model(self, path: Path) -> bool:
        """
        Sauvegarde le modèle entraîné sur disque
        
        Args:
            path: Chemin du fichier de sauvegarde
        
        Returns:
            True si sauvegarde réussie, False sinon
        """
        if not self.is_trained or self.model is None:
            logger.warning("Impossible de sauvegarder: modèle Isolation Forest non entraîné")
            return False
        
        try:
            # Créer le répertoire parent si nécessaire
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Préparer les données à sauvegarder
            save_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'is_trained': self.is_trained
            }
            
            # Sauvegarder avec joblib
            joblib.dump(save_data, path)
            
            logger.info(f"✓ Modèle Isolation Forest sauvegardé: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle Isolation Forest: {e}", exc_info=True)
            return False
    
    def load_model(self, path: Path) -> bool:
        """
        Charge un modèle depuis le disque
        
        Args:
            path: Chemin du fichier de modèle
        
        Returns:
            True si chargement réussi, False sinon
        """
        if not path.exists():
            logger.info(f"Aucun modèle Isolation Forest trouvé à: {path}")
            return False
        
        try:
            # Charger les données
            save_data = joblib.load(path)
            
            # Restaurer l'état
            self.model = save_data['model']
            self.feature_names = save_data['feature_names']
            self.contamination = save_data.get('contamination', self.contamination)
            self.n_estimators = save_data.get('n_estimators', self.n_estimators)
            self.max_samples = save_data.get('max_samples', self.max_samples)
            self.is_trained = save_data.get('is_trained', True)
            
            logger.info(f"✓ Modèle Isolation Forest chargé: {path} ({len(self.feature_names)} features)")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Isolation Forest: {e}", exc_info=True)
            return False

