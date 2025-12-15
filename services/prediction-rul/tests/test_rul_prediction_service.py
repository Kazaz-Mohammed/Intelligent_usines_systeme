"""
Tests pour le service de prédiction RUL principal
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from app.services.rul_prediction_service import RULPredictionService
from app.models.rul_data import RULPredictionRequest, RULPredictionResult


@pytest.fixture
def rul_prediction_service():
    """Fixture pour créer une instance du service"""
    with patch('app.services.rul_prediction_service.settings') as mock_settings:
        mock_settings.enable_lstm = True
        mock_settings.enable_gru = True
        mock_settings.enable_tcn = True
        mock_settings.enable_xgboost = True
        service = RULPredictionService()
        return service


@pytest.fixture
def sample_training_data():
    """Fixture pour créer des données d'entraînement"""
    np.random.seed(42)
    n_samples = 30
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 200
    return X, y


@pytest.fixture
def sample_prediction_request():
    """Fixture pour créer une requête de prédiction"""
    return RULPredictionRequest(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        features={"rms": 10.5, "kurtosis": 2.3, "crest_factor": 3.1},
        timestamp=datetime.now(timezone.utc)
    )


class TestRULPredictionService:
    """Tests pour RULPredictionService"""
    
    def test_init(self, rul_prediction_service):
        """Test initialisation"""
        assert len(rul_prediction_service.models) > 0
        assert "lstm" in rul_prediction_service.models or "gru" in rul_prediction_service.models
    
    def test_train_all_models(self, rul_prediction_service, sample_training_data):
        """Test entraînement de tous les modèles"""
        X_train, y_train = sample_training_data
        
        # Mock les méthodes train des modèles
        for name, model in rul_prediction_service.models.items():
            model.train = Mock(return_value={"train_mae": 5.0, "train_rmse": 7.0})
            model.is_trained = False
        
        results = rul_prediction_service.train_all_models(
            X_train,
            y_train,
            epochs=1,
            batch_size=8
        )
        
        assert len(results) == len(rul_prediction_service.models)
        for name, result in results.items():
            assert "status" in result
            if result["status"] == "success":
                assert "metrics" in result
    
    def test_is_ready_false(self, rul_prediction_service):
        """Test is_ready quand aucun modèle n'est entraîné"""
        for model in rul_prediction_service.models.values():
            model.is_trained = False
        
        assert rul_prediction_service.is_ready() is False
    
    def test_is_ready_true(self, rul_prediction_service):
        """Test is_ready quand au moins un modèle est entraîné"""
        # Marquer le premier modèle comme entraîné
        first_model = list(rul_prediction_service.models.values())[0]
        first_model.is_trained = True
        
        assert rul_prediction_service.is_ready() is True
    
    def test_get_model_status(self, rul_prediction_service):
        """Test récupération du statut des modèles"""
        # Mock get_model_info
        for model in rul_prediction_service.models.values():
            model.get_model_info = Mock(return_value={"model_type": "test", "is_trained": False})
            model.is_trained = False
        
        status = rul_prediction_service.get_model_status()
        
        assert len(status) == len(rul_prediction_service.models)
        for name, model_status in status.items():
            assert "available" in model_status
            assert "trained" in model_status
    
    def test_predict_rul_not_ready(self, rul_prediction_service, sample_prediction_request):
        """Test prédiction quand aucun modèle n'est entraîné"""
        for model in rul_prediction_service.models.values():
            model.is_trained = False
        
        with pytest.raises(RuntimeError, match="Aucun modèle n'est entraîné"):
            rul_prediction_service.predict_rul(sample_prediction_request)
    
    def test_predict_rul_single_model(self, rul_prediction_service, sample_prediction_request):
        """Test prédiction avec un modèle entraîné"""
        # Entraîner un modèle
        first_name = list(rul_prediction_service.models.keys())[0]
        first_model = rul_prediction_service.models[first_name]
        first_model.is_trained = True
        first_model.predict = Mock(return_value=np.array([150.5]))
        
        result = rul_prediction_service.predict_rul(sample_prediction_request, use_ensemble=False)
        
        assert isinstance(result, RULPredictionResult)
        assert result.asset_id == "ASSET001"
        assert result.rul_prediction >= 0
        assert result.confidence_interval_lower >= 0
        assert result.confidence_interval_upper >= 0
    
    def test_predict_rul_ensemble(self, rul_prediction_service, sample_prediction_request):
        """Test prédiction avec ensemble de modèles"""
        # Entraîner plusieurs modèles
        predictions = [150.0, 155.0, 145.0]
        for i, (name, model) in enumerate(rul_prediction_service.models.items()):
            if i < len(predictions):
                model.is_trained = True
                model.predict = Mock(return_value=np.array([predictions[i]]))
        
        result = rul_prediction_service.predict_rul(sample_prediction_request, use_ensemble=True)
        
        assert isinstance(result, RULPredictionResult)
        assert result.model_used == "ensemble"
        assert len(result.model_scores) > 1
        assert result.rul_prediction >= 0
    
    def test_predict_rul_batch(self, rul_prediction_service):
        """Test prédiction batch"""
        requests = [
            RULPredictionRequest(
                asset_id=f"ASSET{i:03d}",
                features={"rms": 10.0 + i, "kurtosis": 2.0, "crest_factor": 3.0}
            )
            for i in range(3)
        ]
        
        # Entraîner un modèle
        first_model = list(rul_prediction_service.models.values())[0]
        first_model.is_trained = True
        first_model.predict = Mock(return_value=np.array([150.0]))
        
        results = rul_prediction_service.predict_rul_batch(requests)
        
        assert len(results) == 3
        assert all(isinstance(r, RULPredictionResult) for r in results)
    
    def test_get_best_model(self, rul_prediction_service):
        """Test récupération du meilleur modèle"""
        # Aucun modèle entraîné
        for model in rul_prediction_service.models.values():
            model.is_trained = False
        
        assert rul_prediction_service.get_best_model() is None
        
        # Entraîner le premier modèle
        first_name = list(rul_prediction_service.models.keys())[0]
        first_model = rul_prediction_service.models[first_name]
        first_model.is_trained = True
        
        best_model = rul_prediction_service.get_best_model()
        assert best_model == first_name

