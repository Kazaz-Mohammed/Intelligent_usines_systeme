"""
Tests pour les modèles de données
"""
from datetime import datetime, timezone
from app.models.anomaly_data import (
    AnomalyDetectionRequest,
    AnomalyScore,
    AnomalyDetectionResult,
    CriticalityLevel,
    TrainingRequest,
    TrainingResult
)


def test_anomaly_detection_request():
    """Test du modèle AnomalyDetectionRequest"""
    request = AnomalyDetectionRequest(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        features={"rms": 25.5, "kurtosis": 3.2, "crest_factor": 2.1}
    )
    assert request.asset_id == "ASSET001"
    assert request.sensor_id == "SENSOR001"
    assert len(request.features) == 3
    assert request.features["rms"] == 25.5
    assert isinstance(request.timestamp, datetime)


def test_anomaly_score():
    """Test du modèle AnomalyScore"""
    score = AnomalyScore(
        score=0.75,
        model_name="isolation_forest",
        threshold=0.5,
        is_anomaly=True
    )
    assert score.score == 0.75
    assert score.model_name == "isolation_forest"
    assert score.threshold == 0.5
    assert score.is_anomaly is True


def test_anomaly_detection_result():
    """Test du modèle AnomalyDetectionResult"""
    scores = [
        AnomalyScore(
            score=0.7,
            model_name="isolation_forest",
            threshold=0.5,
            is_anomaly=True
        ),
        AnomalyScore(
            score=0.6,
            model_name="one_class_svm",
            threshold=0.5,
            is_anomaly=True
        )
    ]
    
    result = AnomalyDetectionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp=datetime.now(timezone.utc),
        scores=scores,
        final_score=0.65,
        is_anomaly=True,
        criticality=CriticalityLevel.HIGH,
        features={"rms": 25.5, "kurtosis": 3.2}
    )
    
    assert result.asset_id == "ASSET001"
    assert len(result.scores) == 2
    assert result.final_score == 0.65
    assert result.is_anomaly is True
    assert result.criticality == CriticalityLevel.HIGH


def test_training_request():
    """Test du modèle TrainingRequest"""
    request = TrainingRequest(
        model_name="isolation_forest",
        parameters={"contamination": 0.1, "n_estimators": 100}
    )
    assert request.model_name == "isolation_forest"
    assert request.parameters["contamination"] == 0.1


def test_training_result():
    """Test du modèle TrainingResult"""
    result = TrainingResult(
        model_name="isolation_forest",
        status="success",
        message="Model trained successfully",
        metrics={"accuracy": 0.95, "precision": 0.92},
        model_version="v1.0"
    )
    assert result.status == "success"
    assert "accuracy" in result.metrics

