"""
Tests pour le service Isolation Forest
"""
import pytest
import numpy as np
from app.services.isolation_forest_service import IsolationForestService


@pytest.fixture
def isolation_forest_service():
    """Fixture pour le service Isolation Forest"""
    return IsolationForestService()


@pytest.fixture
def sample_training_data():
    """Données d'entraînement de test"""
    # Générer des données normales (gaussiennes)
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    return X


@pytest.fixture
def sample_feature_names():
    """Noms de features de test"""
    return ["rms", "kurtosis", "crest_factor", "variance", "mean"]


def test_initialization(isolation_forest_service):
    """Test de l'initialisation du service"""
    assert isolation_forest_service.model is None
    assert isolation_forest_service.is_trained is False
    assert isolation_forest_service.contamination > 0
    assert isolation_forest_service.n_estimators > 0


def test_train(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test de l'entraînement"""
    metrics = isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    assert isolation_forest_service.is_trained is True
    assert isolation_forest_service.model is not None
    assert isolation_forest_service.feature_names == sample_feature_names
    assert metrics["n_samples"] == 100
    assert metrics["n_features"] == 5
    assert "n_anomalies_detected" in metrics
    assert "anomaly_rate" in metrics


def test_predict(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test de la prédiction"""
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    # Tester sur les mêmes données
    predictions = isolation_forest_service.predict(sample_training_data)
    
    assert len(predictions) == len(sample_training_data)
    assert all(pred in [0, 1] for pred in predictions)


def test_predict_scores(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test du scoring"""
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    scores = isolation_forest_service.predict_scores(sample_training_data)
    
    assert len(scores) == len(sample_training_data)
    assert all(0 <= score <= 1 for score in scores)


def test_predict_proba(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test des probabilités"""
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    proba = isolation_forest_service.predict_proba(sample_training_data)
    
    assert proba.shape == (len(sample_training_data), 2)
    assert all(0 <= p <= 1 for row in proba for p in row)
    # Vérifier que prob_normal + prob_anomaly = 1
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_detect_anomaly(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test de la détection d'anomalie avec dictionnaire de features"""
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    # Features normales
    features_normal = {
        "rms": 0.5,
        "kurtosis": 0.3,
        "crest_factor": 1.2,
        "variance": 0.1,
        "mean": 0.0
    }
    
    result = isolation_forest_service.detect_anomaly(features_normal)
    
    assert "score" in result
    assert "is_anomaly" in result
    assert "threshold" in result
    assert "prediction" in result
    assert 0 <= result["score"] <= 1
    assert isinstance(result["is_anomaly"], bool)


def test_detect_anomaly_with_threshold(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test de la détection avec seuil personnalisé"""
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    features = {
        "rms": 0.5,
        "kurtosis": 0.3,
        "crest_factor": 1.2,
        "variance": 0.1,
        "mean": 0.0
    }
    
    result = isolation_forest_service.detect_anomaly(features, threshold=0.5)
    
    assert result["threshold"] == 0.5


def test_get_model_info(isolation_forest_service, sample_training_data, sample_feature_names):
    """Test de récupération des informations du modèle"""
    info_before = isolation_forest_service.get_model_info()
    assert info_before["is_trained"] is False
    
    isolation_forest_service.train(sample_training_data, sample_feature_names)
    
    info_after = isolation_forest_service.get_model_info()
    assert info_after["is_trained"] is True
    assert info_after["model_type"] == "isolation_forest"
    assert info_after["n_features"] == 5
    assert info_after["feature_names"] == sample_feature_names


def test_predict_before_training(isolation_forest_service, sample_training_data):
    """Test qu'une erreur est levée si on prédit avant l'entraînement"""
    with pytest.raises(ValueError, match="n'a pas été entraîné"):
        isolation_forest_service.predict(sample_training_data)


def test_detect_anomaly_before_training(isolation_forest_service):
    """Test qu'une erreur est levée si on détecte avant l'entraînement"""
    with pytest.raises(ValueError, match="n'a pas été entraîné"):
        isolation_forest_service.detect_anomaly({"rms": 0.5})

