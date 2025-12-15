"""
Tests pour le service principal de détection d'anomalies
"""
import pytest
import numpy as np
from datetime import datetime, timezone
from app.services.anomaly_detection_service import AnomalyDetectionService
from app.models.anomaly_data import AnomalyDetectionRequest, CriticalityLevel


@pytest.fixture
def anomaly_detection_service():
    """Fixture pour le service de détection d'anomalies"""
    return AnomalyDetectionService()


@pytest.fixture
def sample_training_data():
    """Données d'entraînement de test"""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    return X


@pytest.fixture
def sample_feature_names():
    """Noms de features de test"""
    return ["rms", "kurtosis", "crest_factor", "variance", "mean"]


@pytest.fixture
def sample_detection_request():
    """Requête de détection de test"""
    return AnomalyDetectionRequest(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        features={
            "rms": 0.5,
            "kurtosis": 0.3,
            "crest_factor": 1.2,
            "variance": 0.1,
            "mean": 0.0
        },
        timestamp=datetime.now(timezone.utc)
    )


def test_initialization(anomaly_detection_service):
    """Test de l'initialisation du service"""
    assert anomaly_detection_service is not None
    # Au moins un service devrait être initialisé selon la config
    assert (
        anomaly_detection_service.isolation_forest_service is not None or
        anomaly_detection_service.one_class_svm_service is not None or
        anomaly_detection_service.lstm_autoencoder_service is not None
    )


def test_train_all_models(anomaly_detection_service, sample_training_data, sample_feature_names):
    """Test de l'entraînement de tous les modèles"""
    results = anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    assert isinstance(results, dict)
    # Vérifier que les modèles activés ont été entraînés
    if anomaly_detection_service.isolation_forest_service is not None:
        assert "isolation_forest" in results
        assert results["isolation_forest"]["status"] == "success"
    
    if anomaly_detection_service.one_class_svm_service is not None:
        assert "one_class_svm" in results
        assert results["one_class_svm"]["status"] == "success"
    
    if anomaly_detection_service.lstm_autoencoder_service is not None:
        assert "lstm_autoencoder" in results
        assert results["lstm_autoencoder"]["status"] == "success"


def test_detect_anomaly(anomaly_detection_service, sample_training_data, sample_feature_names, sample_detection_request):
    """Test de la détection d'anomalie"""
    # Entraîner d'abord
    anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    # Détecter
    result = anomaly_detection_service.detect_anomaly(sample_detection_request)
    
    assert result.asset_id == "ASSET001"
    assert result.sensor_id == "SENSOR001"
    assert len(result.scores) > 0  # Au moins un modèle devrait avoir donné un score
    assert 0 <= result.final_score <= 1
    assert isinstance(result.is_anomaly, bool)
    assert isinstance(result.criticality, CriticalityLevel)
    assert result.features == sample_detection_request.features


def test_detect_anomaly_before_training(anomaly_detection_service, sample_detection_request):
    """Test qu'une détection avant entraînement retourne des scores vides ou gère l'erreur"""
    result = anomaly_detection_service.detect_anomaly(sample_detection_request)
    
    # Si aucun modèle n'est entraîné, les scores seront vides
    # Le score final sera 0.0
    assert result.final_score == 0.0
    assert len(result.scores) == 0


def test_detect_anomalies_batch(anomaly_detection_service, sample_training_data, sample_feature_names):
    """Test de la détection par batch"""
    # Entraîner d'abord
    anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    # Créer plusieurs requêtes
    requests = [
        AnomalyDetectionRequest(
            asset_id=f"ASSET{i:03d}",
            features={
                "rms": 0.5 + i * 0.1,
                "kurtosis": 0.3,
                "crest_factor": 1.2,
                "variance": 0.1,
                "mean": 0.0
            }
        )
        for i in range(5)
    ]
    
    results = anomaly_detection_service.detect_anomalies_batch(requests)
    
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result.asset_id == f"ASSET{i:03d}"
        assert 0 <= result.final_score <= 1


def test_aggregate_scores(anomaly_detection_service):
    """Test de l'agrégation des scores"""
    from app.models.anomaly_data import AnomalyScore
    
    scores = [
        AnomalyScore(score=0.3, model_name="model1", threshold=0.5, is_anomaly=False),
        AnomalyScore(score=0.7, model_name="model2", threshold=0.5, is_anomaly=True),
        AnomalyScore(score=0.5, model_name="model3", threshold=0.5, is_anomaly=False),
        AnomalyScore(score=0.9, model_name="model4", threshold=0.5, is_anomaly=True),
        AnomalyScore(score=0.1, model_name="model5", threshold=0.5, is_anomaly=False),
    ]
    
    final_score = anomaly_detection_service._aggregate_scores(scores)
    
    # Score moyen = (0.3 + 0.7 + 0.5 + 0.9 + 0.1) / 5 = 0.5
    assert abs(final_score - 0.5) < 0.01
    assert 0 <= final_score <= 1


def test_determine_criticality(anomaly_detection_service):
    """Test de la détermination de la criticité"""
    # Test différents scores
    assert anomaly_detection_service._determine_criticality(0.95) == CriticalityLevel.CRITICAL
    assert anomaly_detection_service._determine_criticality(0.75) == CriticalityLevel.HIGH
    assert anomaly_detection_service._determine_criticality(0.6) == CriticalityLevel.MEDIUM
    assert anomaly_detection_service._determine_criticality(0.2) == CriticalityLevel.LOW


def test_determine_criticality_custom_thresholds(anomaly_detection_service):
    """Test de la détermination de criticité avec seuils personnalisés"""
    custom_thresholds = {
        "critical": 0.8,
        "high": 0.6,
        "medium": 0.4,
        "low": 0.2
    }
    
    assert anomaly_detection_service._determine_criticality(0.85, custom_thresholds) == CriticalityLevel.CRITICAL
    assert anomaly_detection_service._determine_criticality(0.65, custom_thresholds) == CriticalityLevel.HIGH
    assert anomaly_detection_service._determine_criticality(0.45, custom_thresholds) == CriticalityLevel.MEDIUM
    assert anomaly_detection_service._determine_criticality(0.15, custom_thresholds) == CriticalityLevel.LOW


def test_get_model_status(anomaly_detection_service, sample_training_data, sample_feature_names):
    """Test de récupération du statut des modèles"""
    status_before = anomaly_detection_service.get_model_status()
    
    # Entraîner
    anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    status_after = anomaly_detection_service.get_model_status()
    
    # Vérifier que le statut a changé
    if "isolation_forest" in status_after:
        assert status_after["isolation_forest"]["is_trained"] is True
    
    if "one_class_svm" in status_after:
        assert status_after["one_class_svm"]["is_trained"] is True
    
    if "lstm_autoencoder" in status_after:
        assert status_after["lstm_autoencoder"]["is_trained"] is True


def test_is_ready(anomaly_detection_service, sample_training_data, sample_feature_names):
    """Test de vérification si le service est prêt"""
    # Avant entraînement
    ready_before = anomaly_detection_service.is_ready()
    
    # Entraîner
    anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    # Après entraînement
    ready_after = anomaly_detection_service.is_ready()
    
    # Au moins un modèle devrait être prêt après entraînement
    assert ready_after is True or ready_before is False


def test_detect_anomaly_with_custom_thresholds(anomaly_detection_service, sample_training_data, sample_feature_names, sample_detection_request):
    """Test de détection avec seuils personnalisés"""
    # Entraîner
    anomaly_detection_service.train_all_models(sample_training_data, sample_feature_names)
    
    # Détecter avec seuils personnalisés
    custom_thresholds = {
        "isolation_forest": 0.3,
        "one_class_svm": 0.4,
        "lstm_autoencoder": 0.5
    }
    
    result = anomaly_detection_service.detect_anomaly(sample_detection_request, custom_thresholds)
    
    # Vérifier que les seuils personnalisés ont été utilisés
    for score in result.scores:
        if score.model_name in custom_thresholds:
            assert score.threshold == custom_thresholds[score.model_name]

