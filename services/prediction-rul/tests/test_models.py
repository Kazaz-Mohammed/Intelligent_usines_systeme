"""
Tests pour les modèles de données
"""
import pytest
from datetime import datetime, timezone
from app.models.rul_data import (
    RULPredictionRequest,
    RULPredictionResult,
    TrainingRequest,
    TrainingResult
)


def test_rul_prediction_request():
    """Test RULPredictionRequest"""
    request = RULPredictionRequest(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        features={"rms": 10.5, "kurtosis": 2.3}
    )
    assert request.asset_id == "ASSET001"
    assert request.sensor_id == "SENSOR001"
    assert request.features["rms"] == 10.5
    assert isinstance(request.timestamp, datetime)


def test_rul_prediction_result():
    """Test RULPredictionResult"""
    result = RULPredictionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp=datetime.now(timezone.utc),
        rul_prediction=150.5,
        confidence_interval_lower=140.0,
        confidence_interval_upper=160.0,
        confidence_level=0.95,
        uncertainty=10.0,
        model_used="lstm",
        features={"rms": 10.5}
    )
    assert result.rul_prediction == 150.5
    assert result.confidence_interval_lower == 140.0
    assert result.confidence_interval_upper == 160.0
    assert result.model_used == "lstm"


def test_training_request():
    """Test TrainingRequest"""
    request = TrainingRequest(
        model_name="lstm",
        training_data=[[1.0, 2.0], [3.0, 4.0]],
        target_data=[10.0, 20.0],
        feature_names=["rms", "kurtosis"]
    )
    assert request.model_name == "lstm"
    assert len(request.training_data) == 2
    assert request.feature_names == ["rms", "kurtosis"]


def test_training_result():
    """Test TrainingResult"""
    result = TrainingResult(
        model_name="lstm",
        status="success",
        message="Modèle entraîné avec succès",
        metrics={"mae": 5.2, "rmse": 7.8},
        model_version="v1.0",
        training_time_seconds=120.5
    )
    assert result.status == "success"
    assert result.metrics["mae"] == 5.2
    assert result.training_time_seconds == 120.5

