"""
Tests pour le service LSTM
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.lstm_service import LSTMService, LSTMModel


@pytest.fixture
def lstm_service():
    """Fixture pour créer une instance du service LSTM"""
    return LSTMService()


@pytest.fixture
def sample_training_data():
    """Fixture pour créer des données d'entraînement (réduites pour tests rapides)"""
    np.random.seed(42)
    n_samples = 30  # Réduire à 30 échantillons
    n_features = 3  # Réduire à 3 features
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 200  # RUL entre 0 et 200
    return X, y


class TestLSTMService:
    """Tests pour LSTMService"""
    
    def test_init(self, lstm_service):
        """Test initialisation"""
        assert lstm_service.model is None
        assert lstm_service.is_trained is False
        assert lstm_service.sequence_length > 0
    
    def test_create_sequences(self, lstm_service):
        """Test création de séquences"""
        data = np.random.randn(50, 5)
        sequence_length = 10
        
        X_seq, y_seq = lstm_service._create_sequences(data, sequence_length)
        
        assert len(X_seq) == len(y_seq)
        assert X_seq.shape[1] == sequence_length
        assert X_seq.shape[2] == 5
    
    def test_create_sequences_insufficient_data(self, lstm_service):
        """Test avec données insuffisantes"""
        data = np.random.randn(5, 5)
        sequence_length = 10
        
        with pytest.raises(ValueError, match="Données insuffisantes"):
            lstm_service._create_sequences(data, sequence_length)
    
    @patch('app.services.lstm_service.mlflow')
    def test_train(self, mock_mlflow, lstm_service, sample_training_data):
        """Test entraînement"""
        X_train, y_train = sample_training_data
        
        # Mock MLflow
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        mock_mlflow.pytorch.log_model = MagicMock()
        
        # Utiliser des paramètres réduits pour accélérer les tests
        lstm_service.sequence_length = 3  # Réduire à 3
        
        # Passer epochs=1 et batch_size=8 pour accélérer au maximum
        metrics = lstm_service.train(X_train, y_train, epochs=1, batch_size=8)
        
        assert lstm_service.is_trained is True
        assert lstm_service.model is not None
        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics
        assert metrics["train_mae"] >= 0
    
    def test_train_not_called_predict(self, lstm_service):
        """Test prédiction sans entraînement"""
        X = np.random.randn(1, 5)
        
        with pytest.raises(RuntimeError, match="Modèle non entraîné"):
            lstm_service.predict(X)
    
    def test_predict_after_train(self, lstm_service, sample_training_data):
        """Test prédiction après entraînement"""
        X_train, y_train = sample_training_data
        
        lstm_service.sequence_length = 3
        
        with patch('app.services.lstm_service.mlflow'):
            lstm_service.train(X_train, y_train, epochs=1, batch_size=8)
        
        # Prédiction avec données 2D
        X_test = np.random.randn(20, 5)
        predictions = lstm_service.predict(X_test)
        
        assert len(predictions) == 1  # Une prédiction pour la séquence
        assert predictions[0] >= 0  # RUL positive
    
    def test_get_model_info(self, lstm_service):
        """Test récupération des informations du modèle"""
        info = lstm_service.get_model_info()
        
        assert info["model_type"] == "lstm"
        assert info["is_trained"] is False
        assert "sequence_length" in info
        assert "hidden_size" in info

