"""
Tests pour le service GRU
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.gru_service import GRUService


@pytest.fixture
def gru_service():
    """Fixture pour créer une instance du service GRU"""
    return GRUService()


@pytest.fixture
def sample_training_data():
    """Fixture pour créer des données d'entraînement (réduites pour tests rapides)"""
    np.random.seed(42)
    n_samples = 30
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 200
    return X, y


class TestGRUService:
    """Tests pour GRUService"""
    
    def test_init(self, gru_service):
        """Test initialisation"""
        assert gru_service.model is None
        assert gru_service.is_trained is False
    
    @patch('app.services.gru_service.mlflow')
    @patch('app.services.gru_service.settings')
    def test_train(self, mock_settings, mock_mlflow, gru_service, sample_training_data):
        """Test entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock settings pour accélérer
        mock_settings.gru_sequence_length = 5
        mock_settings.gru_hidden_size = 16
        mock_settings.gru_num_layers = 1
        mock_settings.gru_dropout = 0.1
        mock_settings.gru_batch_size = 16
        mock_settings.gru_epochs = 2
        mock_settings.gru_learning_rate = 0.001
        mock_settings.mlflow_enabled = False
        
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        mock_mlflow.pytorch.log_model = MagicMock()
        
        gru_service.sequence_length = 3
        
        metrics = gru_service.train(X_train, y_train, epochs=1, batch_size=8)
        
        assert gru_service.is_trained is True
        assert "train_mae" in metrics
    
    @patch('app.services.gru_service.settings')
    def test_predict_after_train(self, mock_settings, gru_service, sample_training_data):
        """Test prédiction après entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock settings pour accélérer
        mock_settings.gru_sequence_length = 5
        mock_settings.gru_hidden_size = 16
        mock_settings.gru_num_layers = 1
        mock_settings.gru_dropout = 0.1
        mock_settings.gru_batch_size = 16
        mock_settings.gru_epochs = 2
        mock_settings.gru_learning_rate = 0.001
        mock_settings.mlflow_enabled = False
        
        gru_service.sequence_length = 3
        
        with patch('app.services.gru_service.mlflow'):
            gru_service.train(X_train, y_train, epochs=1, batch_size=8)
        
        X_test = np.random.randn(20, 5)
        predictions = gru_service.predict(X_test)
        
        assert len(predictions) == 1
        assert predictions[0] >= 0

