"""
Service One-Class SVM pour la détection d'anomalies
"""
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib
from pyod.models.ocsvm import OCSVM
from app.config import settings
# Lazy import MLflowService to avoid slow initialization
MLflowService = None

logger = logging.getLogger(__name__)


class OneClassSVMService:
    """Service pour la détection d'anomalies avec One-Class SVM"""
    
    def __init__(self):
        """Initialise le service One-Class SVM"""
        self.model: Optional[OCSVM] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False
        # Lazy initialization of MLflowService (can be slow)
        self.mlflow_service: Optional[MLflowService] = None
        
        # Paramètres depuis la config
        self.nu = settings.one_class_svm_nu
        self.kernel = settings.one_class_svm_kernel
        self.gamma = settings.one_class_svm_gamma
        
        logger.info(
            f"OneClassSVMService initialisé avec nu={self.nu}, "
            f"kernel={self.kernel}, gamma={self.gamma}"
        )
    
    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Entraîne le modèle One-Class SVM
        
        Args:
            X: Données d'entraînement (n_samples, n_features)
            feature_names: Noms des features (optionnel)
        
        Returns:
            Dict avec les métriques d'entraînement
        """
        try:
            logger.info(f"Entraînement One-Class SVM sur {X.shape[0]} échantillons, {X.shape[1]} features")
            
            # Convertir gamma si nécessaire
            gamma_value = self.gamma
            if isinstance(self.gamma, str):
                if self.gamma == "scale":
                    gamma_value = "scale"
                elif self.gamma == "auto":
                    gamma_value = "auto"
                else:
                    try:
                        gamma_value = float(self.gamma)
                    except ValueError:
                        gamma_value = "scale"
            
            # Initialiser le modèle
            self.model = OCSVM(
                nu=self.nu,
                kernel=self.kernel,
                gamma=gamma_value,
                contamination=0.1  # PyOD utilise contamination pour le seuil
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
                    run = self.mlflow_service.start_run(run_name=f"one_class_svm_{X.shape[0]}_samples")
                    if run:
                        try:
                            # Log parameters
                            self.mlflow_service.log_params({
                                "model_type": "one_class_svm",
                                "nu": self.nu,
                                "kernel": self.kernel,
                                "gamma": str(self.gamma),
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
                                    artifact_path="one_class_svm_model",
                                    registered_model_name="one_class_svm"
                                )
                            except Exception as model_log_error:
                                logger.debug(f"MLflow model logging skipped: {model_log_error}")
                        finally:
                            self.mlflow_service.end_run()
            except Exception as mlflow_error:
                # MLflow is optional - don't fail training if it doesn't work
                logger.debug(f"MLflow logging skipped: {mlflow_error}")
            
            logger.info(f"One-Class SVM entraîné avec succès. Métriques: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement One-Class SVM: {e}", exc_info=True)
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
            logger.error(f"Erreur lors de la prédiction One-Class SVM: {e}", exc_info=True)
            raise
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores d'anomalie (plus élevé = plus anormal)
        
        Args:
            X: Données à scorer (n_samples, n_features)
        
        Returns:
            Scores d'anomalie normalisés entre 0 et 1
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            scores = self.model.decision_function(X)
            # Normaliser les scores entre 0 et 1
            # One-Class SVM retourne des scores négatifs pour les anomalies
            # On inverse et normalise
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores_normalized
        except Exception as e:
            logger.error(f"Erreur lors du scoring One-Class SVM: {e}", exc_info=True)
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
            logger.error(f"Erreur lors du calcul de probabilité One-Class SVM: {e}", exc_info=True)
            raise
    
    def detect_anomaly(self, features: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Détecte une anomalie à partir d'un dictionnaire de features
        
        Args:
            features: Dictionnaire {nom_feature: valeur}
            threshold: Seuil personnalisé (optionnel, utilise nu par défaut)
        
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
            
            # Prédire
            score = self.predict_scores(X)[0]
            prediction = self.predict(X)[0]
            
            # Utiliser le seuil fourni ou celui du modèle
            if threshold is None:
                threshold = self.nu
            
            is_anomaly = score >= threshold or prediction == 1
            
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
            "model_type": "one_class_svm",
            "is_trained": self.is_trained,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
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
            logger.warning("Impossible de sauvegarder: modèle One-Class SVM non entraîné")
            return False
        
        try:
            # Créer le répertoire parent si nécessaire
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Préparer les données à sauvegarder
            save_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'nu': self.nu,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'is_trained': self.is_trained
            }
            
            # Sauvegarder avec joblib
            joblib.dump(save_data, path)
            
            logger.info(f"✓ Modèle One-Class SVM sauvegardé: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle One-Class SVM: {e}", exc_info=True)
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
            logger.info(f"Aucun modèle One-Class SVM trouvé à: {path}")
            return False
        
        try:
            # Charger les données
            save_data = joblib.load(path)
            
            # Restaurer l'état
            self.model = save_data['model']
            self.feature_names = save_data['feature_names']
            self.nu = save_data.get('nu', self.nu)
            self.kernel = save_data.get('kernel', self.kernel)
            self.gamma = save_data.get('gamma', self.gamma)
            self.is_trained = save_data.get('is_trained', True)
            
            logger.info(f"✓ Modèle One-Class SVM chargé: {path} ({len(self.feature_names)} features)")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle One-Class SVM: {e}", exc_info=True)
            return False

