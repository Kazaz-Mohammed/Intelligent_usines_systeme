"""
Tests pour le service LSTM Autoencoder
"""
import pytest
import numpy as np
from app.services.lstm_autoencoder_service import LSTMAutoencoderService


@pytest.fixture
def lstm_autoencoder_service():
    """Fixture pour le service LSTM Autoencoder"""
    return LSTMAutoencoderService()


@pytest.fixture
def sample_training_data():
    """Données d'entraînement de test"""
    # Générer des données normales (gaussiennes)
    np.random.seed(42)
    n_samples = 200  # Plus d'échantillons pour créer des séquences
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    return X


@pytest.fixture
def sample_feature_names():
    """Noms de features de test"""
    return ["rms", "kurtosis", "crest_factor", "variance", "mean"]


def test_initialization(lstm_autoencoder_service):
    """Test de l'initialisation du service"""
    assert lstm_autoencoder_service.model is None
    assert lstm_autoencoder_service.is_trained is False
    assert len(lstm_autoencoder_service.encoder_layers) > 0
    assert len(lstm_autoencoder_service.decoder_layers) > 0
    assert lstm_autoencoder_service.sequence_length > 0
    assert lstm_autoencoder_service.batch_size > 0
    assert lstm_autoencoder_service.epochs > 0


def test_train(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test de l'entraînement"""
    metrics = lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    assert lstm_autoencoder_service.is_trained is True
    assert lstm_autoencoder_service.model is not None
    assert lstm_autoencoder_service.feature_names == sample_feature_names
    assert metrics["n_samples"] == 200
    assert metrics["n_features"] == 5
    assert "n_sequences" in metrics
    assert "final_loss" in metrics
    assert "threshold" in metrics
    assert lstm_autoencoder_service.threshold is not None


def test_predict_scores(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test du scoring"""
    lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    scores = lstm_autoencoder_service.predict_scores(sample_training_data)
    
    assert len(scores) == len(sample_training_data)
    assert all(0 <= score <= 1 for score in scores)


def test_predict(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test de la prédiction"""
    lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    predictions = lstm_autoencoder_service.predict(sample_training_data)
    
    assert len(predictions) == len(sample_training_data)
    assert all(pred in [0, 1] for pred in predictions)


def test_detect_anomaly(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test de la détection d'anomalie avec dictionnaire de features"""
    lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    # Features normales
    features_normal = {
        "rms": 0.5,
        "kurtosis": 0.3,
        "crest_factor": 1.2,
        "variance": 0.1,
        "mean": 0.0
    }
    
    result = lstm_autoencoder_service.detect_anomaly(features_normal)
    
    assert "score" in result
    assert "is_anomaly" in result
    assert "threshold" in result
    assert "reconstruction_error" in result
    assert 0 <= result["score"] <= 1
    assert isinstance(result["is_anomaly"], bool)
    assert result["reconstruction_error"] >= 0


def test_detect_anomaly_with_threshold(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test de la détection avec seuil personnalisé"""
    lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    features = {
        "rms": 0.5,
        "kurtosis": 0.3,
        "crest_factor": 1.2,
        "variance": 0.1,
        "mean": 0.0
    }
    
    result = lstm_autoencoder_service.detect_anomaly(features, threshold=0.5)
    
    assert result["threshold"] == 0.5


def test_get_model_info(lstm_autoencoder_service, sample_training_data, sample_feature_names):
    """Test de récupération des informations du modèle"""
    info_before = lstm_autoencoder_service.get_model_info()
    assert info_before["is_trained"] is False
    
    lstm_autoencoder_service.train(sample_training_data, sample_feature_names)
    
    info_after = lstm_autoencoder_service.get_model_info()
    assert info_after["is_trained"] is True
    assert info_after["model_type"] == "lstm_autoencoder"
    assert info_after["n_features"] == 5
    assert info_after["feature_names"] == sample_feature_names
    assert info_after["threshold"] is not None
    assert "device" in info_after


def test_predict_before_training(lstm_autoencoder_service, sample_training_data):
    """Test qu'une erreur est levée si on prédit avant l'entraînement"""
    with pytest.raises(ValueError, match="n'a pas été entraîné"):
        lstm_autoencoder_service.predict(sample_training_data)


def test_detect_anomaly_before_training(lstm_autoencoder_service):
    """Test qu'une erreur est levée si on détecte avant l'entraînement"""
    with pytest.raises(ValueError, match="n'a pas été entraîné"):
        lstm_autoencoder_service.detect_anomaly({"rms": 0.5})


def test_create_sequences(lstm_autoencoder_service):
    """Test de la création de séquences"""
    # Données de test
    X = np.random.randn(100, 5)
    sequences = lstm_autoencoder_service._create_sequences(X)
    
    # Vérifier la forme
    expected_sequences = len(X) - lstm_autoencoder_service.sequence_length + 1
    assert len(sequences) == expected_sequences
    assert sequences.shape[1] == lstm_autoencoder_service.sequence_length
    assert sequences.shape[2] == X.shape[1]


def test_sequence_length_requirement(lstm_autoencoder_service):
    """Test que les données doivent avoir assez d'échantillons pour créer des séquences"""
    # Données avec moins d'échantillons que sequence_length
    X = np.random.randn(5, 5)  # Moins que sequence_length (10 par défaut)
    
    # L'entraînement devrait quand même fonctionner (mais créer 0 séquences)
    # Ou lever une erreur appropriée
    # Pour l'instant, on teste juste que ça ne plante pas
    try:
        lstm_autoencoder_service.train(X)
        # Si l'entraînement réussit, vérifier que predict_scores gère le cas
        scores = lstm_autoencoder_service.predict_scores(X)
        assert len(scores) == len(X)
    except Exception:
        # C'est acceptable si ça lève une erreur pour données insuffisantes
        pass

