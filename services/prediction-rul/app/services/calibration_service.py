"""
Service de calibration pour améliorer les prédictions RUL et quantifier l'incertitude
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from app.config import settings

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Module PyTorch pour temperature scaling"""
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """Applique temperature scaling aux logits"""
        return logits / self.temperature


class CalibrationService:
    """Service pour calibrer les prédictions RUL et quantifier l'incertitude"""
    
    def __init__(self):
        """Initialise le service de calibration"""
        self.calibration_method = settings.calibration_method
        self.calibration_enabled = settings.calibration_enabled
        
        # Modèles de calibration
        self.isotonic_regressor: Optional[IsotonicRegression] = None
        self.platt_scaler: Optional[LogisticRegression] = None
        self.temperature_scaler: Optional[TemperatureScaling] = None
        
        # Données de calibration
        self.calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.is_calibrated: bool = False
        
        logger.info(f"CalibrationService initialisé avec méthode: {self.calibration_method}")
    
    def fit_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle de calibration sur les prédictions et valeurs réelles
        
        Args:
            predictions: Prédictions du modèle (1D array)
            actuals: Valeurs réelles (1D array)
            method: Méthode de calibration ('isotonic', 'platt', 'temperature_scaling')
        
        Returns:
            Dictionnaire avec métriques de calibration
        """
        if not self.calibration_enabled:
            logger.warning("Calibration désactivée dans la configuration")
            return {"status": "disabled"}
        
        method = method or self.calibration_method
        
        if len(predictions) != len(actuals):
            raise ValueError("predictions et actuals doivent avoir la même longueur")
        
        if len(predictions) < 10:
            logger.warning("Pas assez de données pour la calibration (minimum 10)")
            return {"status": "insufficient_data"}
        
        # Stocker les données de calibration
        self.calibration_data = (predictions.copy(), actuals.copy())
        
        try:
            if method == "isotonic":
                self._fit_isotonic(predictions, actuals)
            elif method == "platt":
                self._fit_platt(predictions, actuals)
            elif method == "temperature_scaling":
                self._fit_temperature_scaling(predictions, actuals)
            else:
                raise ValueError(f"Méthode de calibration inconnue: {method}")
            
            # Calculer les métriques de calibration
            calibrated_preds = self.calibrate_predictions(predictions)
            metrics = self._compute_calibration_metrics(calibrated_preds, actuals)
            
            self.is_calibrated = True
            logger.info(f"Calibration {method} terminée avec succès. MAE: {metrics.get('mae', 0):.4f}")
            
            return {
                "status": "success",
                "method": method,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la calibration: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _fit_isotonic(self, predictions: np.ndarray, actuals: np.ndarray):
        """Entraîne un régresseur isotonique"""
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(predictions, actuals)
        logger.debug("Régresseur isotonique entraîné")
    
    def _fit_platt(self, predictions: np.ndarray, actuals: np.ndarray):
        """Entraîne un scaler Platt (logistic regression)"""
        # Platt scaling nécessite des probabilités, on normalise les prédictions
        # Pour RUL, on convertit en probabilités de survie
        # Approche simplifiée : normaliser entre 0 et 1
        predictions_normalized = self._normalize_predictions(predictions)
        actuals_normalized = self._normalize_predictions(actuals)
        
        # Utiliser une régression logistique pour mapper prédictions -> actuals
        self.platt_scaler = LogisticRegression()
        # Reshape pour sklearn
        X = predictions_normalized.reshape(-1, 1)
        y = (actuals_normalized > 0.5).astype(int)  # Binariser pour classification
        
        self.platt_scaler.fit(X, y)
        logger.debug("Scaler Platt entraîné")
    
    def _fit_temperature_scaling(self, predictions: np.ndarray, actuals: np.ndarray):
        """Entraîne temperature scaling (nécessite des logits)"""
        # Pour RUL, on convertit les prédictions en logits
        # Approche : utiliser les prédictions comme logits et optimiser la température
        predictions_tensor = torch.FloatTensor(predictions)
        actuals_tensor = torch.FloatTensor(actuals)
        
        self.temperature_scaler = TemperatureScaling()
        optimizer = torch.optim.LBFGS([self.temperature_scaler.temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            scaled = self.temperature_scaler(predictions_tensor.unsqueeze(1))
            loss = nn.MSELoss()(scaled.squeeze(), actuals_tensor)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        logger.debug(f"Temperature scaling entraîné. Température: {self.temperature_scaler.temperature.item():.4f}")
    
    def calibrate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibre les prédictions en utilisant le modèle entraîné
        
        Args:
            predictions: Prédictions à calibrer (1D array)
        
        Returns:
            Prédictions calibrées
        """
        if not self.is_calibrated:
            logger.warning("Calibration non entraînée, retour des prédictions originales")
            return predictions
        
        try:
            if self.calibration_method == "isotonic" and self.isotonic_regressor is not None:
                calibrated = self.isotonic_regressor.predict(predictions)
                return np.clip(calibrated, 0, None)  # RUL doit être >= 0
            
            elif self.calibration_method == "platt" and self.platt_scaler is not None:
                # Pour Platt, on inverse la normalisation
                predictions_normalized = self._normalize_predictions(predictions)
                X = predictions_normalized.reshape(-1, 1)
                calibrated_normalized = self.platt_scaler.predict_proba(X)[:, 1]
                # Dénormaliser (approximation)
                calibrated = self._denormalize_predictions(calibrated_normalized, predictions)
                return np.clip(calibrated, 0, None)
            
            elif self.calibration_method == "temperature_scaling" and self.temperature_scaler is not None:
                predictions_tensor = torch.FloatTensor(predictions)
                with torch.no_grad():
                    calibrated_tensor = self.temperature_scaler(predictions_tensor.unsqueeze(1))
                calibrated = calibrated_tensor.squeeze().numpy()
                return np.clip(calibrated, 0, None)
            
            else:
                logger.warning(f"Méthode {self.calibration_method} non disponible, retour des prédictions originales")
                return predictions
                
        except Exception as e:
            logger.error(f"Erreur lors de la calibration: {e}", exc_info=True)
            return predictions
    
    def compute_uncertainty(
        self,
        predictions: np.ndarray,
        method: str = "std"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule l'incertitude des prédictions
        
        Args:
            predictions: Prédictions (peut être un array 2D avec plusieurs modèles)
            method: Méthode de calcul ('std', 'quantile', 'ensemble')
        
        Returns:
            Tuple (uncertainty, confidence_interval_lower, confidence_interval_upper)
        """
        if predictions.ndim == 1:
            # Une seule prédiction, utiliser une estimation basée sur la variance
            uncertainty = predictions * 0.1  # 10% par défaut
            confidence_lower = predictions - 1.96 * uncertainty
            confidence_upper = predictions + 1.96 * uncertainty
        else:
            # Plusieurs prédictions (ensemble de modèles)
            if method == "std":
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                uncertainty = std_pred
                confidence_lower = mean_pred - 1.96 * std_pred
                confidence_upper = mean_pred + 1.96 * std_pred
            elif method == "quantile":
                mean_pred = np.mean(predictions, axis=0)
                confidence_lower = np.percentile(predictions, 2.5, axis=0)
                confidence_upper = np.percentile(predictions, 97.5, axis=0)
                uncertainty = (confidence_upper - confidence_lower) / 2
            else:  # ensemble
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                uncertainty = std_pred
                confidence_lower = mean_pred - 1.96 * std_pred
                confidence_upper = mean_pred + 1.96 * std_pred
        
        # S'assurer que RUL est >= 0
        confidence_lower = np.clip(confidence_lower, 0, None)
        confidence_upper = np.clip(confidence_upper, 0, None)
        uncertainty = np.clip(uncertainty, 0, None)
        
        return uncertainty, confidence_lower, confidence_upper
    
    def compute_confidence_interval(
        self,
        prediction: float,
        uncertainty: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calcule l'intervalle de confiance pour une prédiction
        
        Args:
            prediction: Prédiction RUL
            uncertainty: Incertitude (écart-type)
            confidence_level: Niveau de confiance (0.95 = 95%)
        
        Returns:
            Tuple (lower, upper)
        """
        z_score = self._get_z_score(confidence_level)
        margin = z_score * uncertainty
        
        lower = max(0.0, prediction - margin)
        upper = prediction + margin
        
        return lower, upper
    
    def _normalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Normalise les prédictions entre 0 et 1"""
        min_val = np.min(predictions)
        max_val = np.max(predictions)
        if max_val == min_val:
            return np.ones_like(predictions) * 0.5
        return (predictions - min_val) / (max_val - min_val)
    
    def _denormalize_predictions(
        self,
        normalized: np.ndarray,
        original_range: np.ndarray
    ) -> np.ndarray:
        """Dénormalise les prédictions"""
        min_val = np.min(original_range)
        max_val = np.max(original_range)
        return normalized * (max_val - min_val) + min_val
    
    def _compute_calibration_metrics(
        self,
        calibrated_preds: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calcule les métriques de calibration"""
        mae = np.mean(np.abs(calibrated_preds - actuals))
        rmse = np.sqrt(np.mean((calibrated_preds - actuals) ** 2))
        mape = np.mean(np.abs((actuals - calibrated_preds) / (actuals + 1e-8))) * 100
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Retourne le z-score pour un niveau de confiance donné"""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.96)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Retourne des informations sur l'état de la calibration"""
        return {
            "enabled": self.calibration_enabled,
            "method": self.calibration_method,
            "is_calibrated": self.is_calibrated,
            "has_calibration_data": self.calibration_data is not None
        }

