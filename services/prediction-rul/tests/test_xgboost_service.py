"""
Tests pour le service XGBoost
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# XGBoost peut ne pas être installé, donc on le rend optionnel
try:
    from app.services.xgboost_service import XGBoostService
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@pytest.fixture
def xgboost_service():
    """Fixture pour créer une instance du service XGBoost"""
    if not XGBOOST_AVAILABLE:
        pytest.skip("XGBoost non disponible")
    return XGBoostService()


@pytest.fixture
def sample_training_data():
    """Fixture pour créer des données d'entraînement"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 200
    return X, y


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost non disponible")
class TestXGBoostService:
    """Tests pour XGBoostService"""
    
    def test_init(self, xgboost_service):
        """Test initialisation"""
        assert xgboost_service.model is None
        assert xgboost_service.is_trained is False
    
    @patch('app.services.xgboost_service.mlflow')
    @patch('app.services.xgboost_service.settings')
    def test_train(self, mock_settings, mock_mlflow, xgboost_service, sample_training_data):
        """Test entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock settings pour accélérer
        mock_settings.xgboost_n_estimators = 5  # Réduire à 5 arbres
        mock_settings.xgboost_max_depth = 3
        mock_settings.xgboost_learning_rate = 0.1
        mock_settings.xgboost_subsample = 0.8
        mock_settings.mlflow_enabled = False
        
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        mock_mlflow.xgboost.log_model = MagicMock()
        
        metrics = xgboost_service.train(X_train, y_train)
        
        assert xgboost_service.is_trained is True
        assert xgboost_service.model is not None
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics
    
    def test_train_with_validation(self, xgboost_service, sample_training_data):
        """Test entraînement avec validation"""
        X_train, y_train = sample_training_data
        X_val = np.random.randn(20, 5)
        y_val = np.random.rand(20) * 200
        
        with patch('app.services.xgboost_service.mlflow'):
            metrics = xgboost_service.train(X_train, y_train, X_val, y_val)
        
        assert "val_mae" in metrics
        assert "val_rmse" in metrics
    
    def test_predict_after_train(self, xgboost_service, sample_training_data):
        """Test prédiction après entraînement"""
        X_train, y_train = sample_training_data
        
        with patch('app.services.xgboost_service.mlflow'):
            xgboost_service.train(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        predictions = xgboost_service.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)  # Toutes positives
    
    def test_predict_with_3d_data(self, xgboost_service, sample_training_data):
        """Test prédiction avec données 3D"""
        X_train, y_train = sample_training_data
        
        with patch('app.services.xgboost_service.mlflow'):
            xgboost_service.train(X_train, y_train)
        
        # Données 3D (séquences)
        X_test = np.random.randn(5, 10, 5)
        predictions = xgboost_service.predict(X_test)
        
        assert len(predictions) == 5
        assert all(p >= 0 for p in predictions)
    
    def test_get_model_info(self, xgboost_service):
        """Test récupération des informations du modèle"""
        info = xgboost_service.get_model_info()
        
        assert info["model_type"] == "xgboost"
        assert info["is_trained"] is False
        assert "n_estimators" in info

