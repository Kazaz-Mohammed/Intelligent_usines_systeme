"""
Service XGBoost pour la prédiction RUL
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import xgboost as xgb
# MLflow imports are optional
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from app.config import settings

logger = logging.getLogger(__name__)


class XGBoostService:
    """Service pour la prédiction RUL avec XGBoost"""
    
    def __init__(self):
        """Initialise le service XGBoost"""
        self.model: Optional[xgb.XGBRegressor] = None
        self.is_trained = False
        
        logger.info("XGBoost Service initialisé")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle XGBoost
        
        Args:
            X_train: Données d'entraînement (n_samples, n_features)
            y_train: Targets d'entraînement (n_samples,)
            X_val: Données de validation (optionnel)
            y_val: Targets de validation (optionnel)
            feature_names: Noms des features
        
        Returns:
            Dict avec métriques d'entraînement
        """
        logger.info("Début de l'entraînement XGBoost")
        
        # Si données 3D, prendre la dernière observation de chaque séquence
        if len(X_train.shape) == 3:
            X_train = X_train[:, -1, :]  # (n_samples, n_features)
        
        # Créer le modèle
        self.model = xgb.XGBRegressor(
            n_estimators=settings.xgboost_n_estimators,
            max_depth=settings.xgboost_max_depth,
            learning_rate=settings.xgboost_learning_rate,
            subsample=settings.xgboost_subsample,
            random_state=42,
            n_jobs=-1
        )
        
        # Entraînement
        eval_set = []
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 3:
                X_val = X_val[:, -1, :]
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Évaluation
        train_predictions = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        train_r2 = r2_score(y_train, train_predictions)
        
        metrics = {
            "train_mae": float(train_mae),
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2)
        }
        
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_predictions))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_predictions)))
            metrics["val_r2"] = float(r2_score(y_val, val_predictions))
        
        self.is_trained = True
        logger.info(f"Entraînement XGBoost terminé. MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        
        # MLflow logging
        if settings.mlflow_enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.log_params({
                    "model": "xgboost",
                    "n_estimators": settings.xgboost_n_estimators,
                    "max_depth": settings.xgboost_max_depth,
                    "learning_rate": settings.xgboost_learning_rate,
                    "subsample": settings.xgboost_subsample
                })
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(self.model, "xgboost_model")
                logger.info("Modèle XGBoost enregistré dans MLflow")
            except Exception as e:
                logger.warning(f"Erreur lors du logging MLflow: {e}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la RUL
        
        Args:
            X: Données de forme (n_samples, n_features) ou (n_samples, sequence_length, n_features)
        
        Returns:
            Prédictions RUL (n_samples,)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez train() d'abord.")
        
        # Si données 3D, prendre la dernière observation
        if len(X.shape) == 3:
            X = X[:, -1, :]
        
        # Si données 2D avec une seule observation, reshape
        if len(X.shape) == 2 and X.shape[0] == 1:
            pass  # OK
        
        predictions = self.model.predict(X)
        
        # RUL doit être positive
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "model_type": "xgboost",
            "is_trained": self.is_trained,
            "n_estimators": settings.xgboost_n_estimators,
            "max_depth": settings.xgboost_max_depth,
            "learning_rate": settings.xgboost_learning_rate
        }
    
    def save_model(self, save_path: str) -> bool:
        """
        Sauvegarde le modèle entraîné sur disque
        
        Args:
            save_path: Chemin où sauvegarder le modèle
            
        Returns:
            True si succès, False sinon
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modèle non entraîné, impossible de sauvegarder")
            return False
        
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # XGBoost a sa propre méthode de sauvegarde
            self.model.save_model(str(save_path))
            logger.info(f"Modèle XGBoost sauvegardé dans {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle XGBoost: {e}", exc_info=True)
            return False
    
    def load_model(self, load_path: str) -> bool:
        """
        Charge un modèle entraîné depuis le disque
        
        Args:
            load_path: Chemin du modèle à charger
            
        Returns:
            True si succès, False sinon
        """
        try:
            load_path = Path(load_path)
            if not load_path.exists():
                logger.warning(f"Fichier modèle non trouvé: {load_path}")
                return False
            
            # Créer un nouveau modèle XGBoost avec les mêmes paramètres
            self.model = xgb.XGBRegressor(
                n_estimators=settings.xgboost_n_estimators,
                max_depth=settings.xgboost_max_depth,
                learning_rate=settings.xgboost_learning_rate,
                subsample=settings.xgboost_subsample,
                random_state=42,
                n_jobs=-1
            )
            
            # Charger les poids depuis le fichier
            self.model.load_model(str(load_path))
            
            self.is_trained = True
            logger.info(f"Modèle XGBoost chargé depuis {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle XGBoost: {e}", exc_info=True)
            return False

