"""
Tests pour le service TCN
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.tcn_service import TCNService


@pytest.fixture
def tcn_service():
    """Fixture pour créer une instance du service TCN"""
    return TCNService()


@pytest.fixture
def sample_training_data():
    """Fixture pour créer des données d'entraînement"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 200
    return X, y


class TestTCNService:
    """Tests pour TCNService"""
    
    def test_init(self, tcn_service):
        """Test initialisation"""
        assert tcn_service.model is None
        assert tcn_service.is_trained is False
    
    @patch('app.services.tcn_service.mlflow')
    @patch('app.services.tcn_service.settings')
    def test_train(self, mock_settings, mock_mlflow, tcn_service, sample_training_data):
        """Test entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock settings pour accélérer
        mock_settings.tcn_sequence_length = 5
        mock_settings.tcn_num_channels = [16, 32]  # Réduire
        mock_settings.tcn_kernel_size = 2
        mock_settings.tcn_dropout = 0.1
        mock_settings.tcn_batch_size = 16
        mock_settings.tcn_epochs = 2
        mock_settings.tcn_learning_rate = 0.001
        mock_settings.mlflow_enabled = False
        
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        mock_mlflow.pytorch.log_model = MagicMock()
        
        tcn_service.sequence_length = 5
        
        metrics = tcn_service.train(X_train, y_train)
        
        assert tcn_service.is_trained is True
        assert "train_mae" in metrics
    
    @patch('app.services.tcn_service.settings')
    def test_predict_after_train(self, mock_settings, tcn_service, sample_training_data):
        """Test prédiction après entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock settings pour accélérer
        mock_settings.tcn_sequence_length = 5
        mock_settings.tcn_num_channels = [16, 32]
        mock_settings.tcn_kernel_size = 2
        mock_settings.tcn_dropout = 0.1
        mock_settings.tcn_batch_size = 16
        mock_settings.tcn_epochs = 2
        mock_settings.tcn_learning_rate = 0.001
        mock_settings.mlflow_enabled = False
        
        tcn_service.sequence_length = 5
        
        with patch('app.services.tcn_service.mlflow'):
            tcn_service.train(X_train, y_train)
        
        X_test = np.random.randn(20, 5)
        predictions = tcn_service.predict(X_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0

