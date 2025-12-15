"""
Tests pour la configuration
"""
from app.config import settings


def test_settings_loaded():
    """Test que les settings sont chargés"""
    assert settings.service_name == "detection-anomalies-service"
    assert settings.service_port == 8084
    assert settings.kafka_bootstrap_servers == "localhost:9092"
    assert settings.kafka_topic_input_features == "extracted-features"
    assert settings.kafka_topic_output_anomalies == "anomalies-detected"


def test_model_flags():
    """Test que les flags des modèles sont définis"""
    assert isinstance(settings.enable_isolation_forest, bool)
    assert isinstance(settings.enable_one_class_svm, bool)
    assert isinstance(settings.enable_lstm_autoencoder, bool)


def test_isolation_forest_params():
    """Test des paramètres Isolation Forest"""
    assert 0.0 <= settings.isolation_forest_contamination <= 1.0
    assert settings.isolation_forest_n_estimators > 0


def test_one_class_svm_params():
    """Test des paramètres One-Class SVM"""
    assert 0.0 < settings.one_class_svm_nu <= 1.0
    assert settings.one_class_svm_kernel in ["rbf", "linear", "poly", "sigmoid"]


def test_lstm_autoencoder_params():
    """Test des paramètres LSTM Autoencoder"""
    assert len(settings.lstm_autoencoder_encoder_layers) > 0
    assert len(settings.lstm_autoencoder_decoder_layers) > 0
    assert settings.lstm_autoencoder_sequence_length > 0
    assert settings.lstm_autoencoder_batch_size > 0
    assert settings.lstm_autoencoder_epochs > 0
    assert settings.lstm_autoencoder_learning_rate > 0
    assert 0.0 <= settings.lstm_autoencoder_threshold_percentile <= 100.0

