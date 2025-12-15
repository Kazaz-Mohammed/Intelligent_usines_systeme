"""
Tests pour le service de calibration
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from app.services.calibration_service import CalibrationService, TemperatureScaling
from app.config import settings


@pytest.fixture
def calibration_service():
    """Fixture pour créer une instance du service"""
    with patch('app.services.calibration_service.settings') as mock_settings:
        mock_settings.calibration_enabled = True
        mock_settings.calibration_method = "isotonic"
        service = CalibrationService()
        return service


@pytest.fixture
def sample_calibration_data():
    """Fixture pour créer des données de calibration"""
    np.random.seed(42)
    n_samples = 100
    # Prédictions avec un biais systématique
    predictions = np.random.rand(n_samples) * 200 + 50
    # Valeurs réelles (ajouter un biais pour tester la calibration)
    actuals = predictions * 0.9 + np.random.randn(n_samples) * 10
    return predictions, actuals


class TestCalibrationService:
    """Tests pour CalibrationService"""
    
    def test_init(self, calibration_service):
        """Test initialisation"""
        assert calibration_service.calibration_enabled is True
        assert calibration_service.is_calibrated is False
        assert calibration_service.calibration_data is None
    
    def test_fit_calibration_isotonic(self, calibration_service, sample_calibration_data):
        """Test entraînement calibration isotonic"""
        predictions, actuals = sample_calibration_data
        
        result = calibration_service.fit_calibration(predictions, actuals, method="isotonic")
        
        assert result["status"] == "success"
        assert result["method"] == "isotonic"
        assert "metrics" in result
        assert calibration_service.is_calibrated is True
        assert calibration_service.isotonic_regressor is not None
    
    def test_fit_calibration_platt(self, calibration_service, sample_calibration_data):
        """Test entraînement calibration platt"""
        predictions, actuals = sample_calibration_data
        
        result = calibration_service.fit_calibration(predictions, actuals, method="platt")
        
        assert result["status"] == "success"
        assert result["method"] == "platt"
        assert calibration_service.is_calibrated is True
        assert calibration_service.platt_scaler is not None
    
    def test_fit_calibration_temperature_scaling(self, calibration_service, sample_calibration_data):
        """Test entraînement calibration temperature scaling"""
        predictions, actuals = sample_calibration_data
        
        result = calibration_service.fit_calibration(predictions, actuals, method="temperature_scaling")
        
        assert result["status"] == "success"
        assert result["method"] == "temperature_scaling"
        assert calibration_service.is_calibrated is True
        assert calibration_service.temperature_scaler is not None
    
    def test_fit_calibration_insufficient_data(self, calibration_service):
        """Test avec données insuffisantes"""
        predictions = np.array([1.0, 2.0, 3.0])  # Seulement 3 échantillons
        actuals = np.array([1.0, 2.0, 3.0])
        
        result = calibration_service.fit_calibration(predictions, actuals)
        
        assert result["status"] == "insufficient_data"
    
    def test_calibrate_predictions_isotonic(self, calibration_service, sample_calibration_data):
        """Test calibration des prédictions avec isotonic"""
        predictions, actuals = sample_calibration_data
        
        # Entraîner la calibration
        calibration_service.fit_calibration(predictions, actuals, method="isotonic")
        
        # Tester la calibration
        test_predictions = np.array([100.0, 150.0, 200.0])
        calibrated = calibration_service.calibrate_predictions(test_predictions)
        
        assert len(calibrated) == len(test_predictions)
        assert all(cal >= 0 for cal in calibrated)  # RUL doit être >= 0
    
    def test_calibrate_predictions_not_trained(self, calibration_service):
        """Test calibration sans entraînement"""
        predictions = np.array([100.0, 150.0])
        calibrated = calibration_service.calibrate_predictions(predictions)
        
        # Devrait retourner les prédictions originales
        assert np.array_equal(calibrated, predictions)
    
    def test_compute_uncertainty_single(self, calibration_service):
        """Test calcul incertitude pour une seule prédiction"""
        predictions = np.array([100.0])
        
        uncertainty, lower, upper = calibration_service.compute_uncertainty(predictions)
        
        assert uncertainty >= 0
        assert lower >= 0
        assert upper >= 0
        assert lower <= upper
    
    def test_compute_uncertainty_ensemble(self, calibration_service):
        """Test calcul incertitude pour un ensemble"""
        # Simuler plusieurs prédictions de modèles différents
        predictions = np.array([
            [100.0, 150.0, 200.0],
            [105.0, 155.0, 205.0],
            [95.0, 145.0, 195.0]
        ])
        
        uncertainty, lower, upper = calibration_service.compute_uncertainty(
            predictions, method="std"
        )
        
        assert len(uncertainty) == 3
        assert all(u >= 0 for u in uncertainty)
        assert all(l >= 0 for l in lower)
        assert all(u >= 0 for u in upper)
    
    def test_compute_uncertainty_quantile(self, calibration_service):
        """Test calcul incertitude avec méthode quantile"""
        predictions = np.array([
            [100.0, 150.0],
            [105.0, 155.0],
            [95.0, 145.0]
        ])
        
        uncertainty, lower, upper = calibration_service.compute_uncertainty(
            predictions, method="quantile"
        )
        
        assert len(uncertainty) == 2
        assert all(l <= u for l, u in zip(lower, upper))
    
    def test_compute_confidence_interval(self, calibration_service):
        """Test calcul intervalle de confiance"""
        prediction = 100.0
        uncertainty = 10.0
        
        lower, upper = calibration_service.compute_confidence_interval(
            prediction, uncertainty, confidence_level=0.95
        )
        
        assert lower >= 0
        assert upper >= 0
        assert lower <= prediction <= upper
    
    def test_get_calibration_info(self, calibration_service):
        """Test récupération des informations"""
        info = calibration_service.get_calibration_info()
        
        assert "enabled" in info
        assert "method" in info
        assert "is_calibrated" in info
        assert "has_calibration_data" in info


class TestTemperatureScaling:
    """Tests pour TemperatureScaling"""
    
    def test_init(self):
        """Test initialisation"""
        scaler = TemperatureScaling()
        assert scaler.temperature.item() == 1.0
    
    def test_forward(self):
        """Test forward pass"""
        scaler = TemperatureScaling()
        scaler.temperature.data = torch.tensor([2.0])
        
        logits = torch.tensor([10.0, 20.0, 30.0])
        scaled = scaler(logits)
        
        assert torch.allclose(scaled, logits / 2.0)

