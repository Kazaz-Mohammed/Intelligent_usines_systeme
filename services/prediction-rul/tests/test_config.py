"""
Tests pour la configuration
"""
import pytest
from app.config import settings


def test_settings_loaded():
    """Test que les settings sont chargés"""
    assert settings.service_name == "prediction-rul-service"
    assert settings.service_port == 8085
    assert settings.log_level == "INFO"


def test_kafka_config():
    """Test configuration Kafka"""
    assert settings.kafka_bootstrap_servers == "localhost:9092"
    assert settings.kafka_consumer_group == "prediction-rul-service"
    assert settings.kafka_topic_input_features == "extracted-features"
    assert settings.kafka_topic_output_rul == "rul-predictions"


def test_database_config():
    """Test configuration base de données"""
    assert settings.database_host == "localhost"
    assert settings.database_port == 5432
    assert settings.database_name == "predictive_maintenance"


def test_models_config():
    """Test configuration des modèles"""
    assert settings.enable_lstm is True
    assert settings.enable_gru is True
    assert settings.enable_tcn is True
    assert settings.enable_xgboost is True


def test_lstm_parameters():
    """Test paramètres LSTM"""
    assert settings.lstm_hidden_size == 64
    assert settings.lstm_num_layers == 2
    assert settings.lstm_sequence_length == 20


def test_mlflow_config():
    """Test configuration MLflow"""
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.mlflow_experiment_name == "rul-prediction"
    assert settings.mlflow_enabled is True

