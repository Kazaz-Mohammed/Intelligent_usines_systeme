"""
Tests pour l'API REST de détection d'anomalies
"""
import pytest
import httpx
from httpx import ASGITransport
import numpy as np
from datetime import datetime, timezone
from app.main import app
from app.models.anomaly_data import AnomalyDetectionRequest


@pytest.fixture
async def client():
    """Client de test FastAPI"""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_training_data():
    """Données d'entraînement de test"""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features).tolist()
    return X


@pytest.fixture
def sample_detection_request():
    """Requête de détection de test"""
    return {
        "asset_id": "ASSET001",
        "sensor_id": "SENSOR001",
        "features": {
            "rms": 0.5,
            "kurtosis": 0.3,
            "crest_factor": 1.2,
            "variance": 0.1,
            "mean": 0.0
        }
    }


@pytest.mark.asyncio
async def test_detect_anomaly_not_ready(client):
    """Test que la détection échoue si aucun modèle n'est entraîné"""
    request = {
        "asset_id": "ASSET001",
        "features": {"rms": 0.5}
    }
    
    response = await client.post("/api/v1/anomalies/detect", json=request)
    
    assert response.status_code == 503
    data = response.json()
    assert "n'est entraîné" in data["detail"]


@pytest.mark.asyncio
async def test_train_models(client, sample_training_data):
    """Test de l'entraînement des modèles"""
    training_data = {
        "data": sample_training_data,
        "feature_names": ["rms", "kurtosis", "crest_factor", "variance", "mean"]
    }
    
    response = await client.post("/api/v1/anomalies/train", json=training_data)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # Au moins un modèle devrait être entraîné
    assert len(data) > 0


@pytest.mark.asyncio
async def test_detect_anomaly(client, sample_training_data, sample_detection_request):
    """Test de la détection d'anomalie"""
    # Entraîner d'abord
    training_data = {
        "data": sample_training_data,
        "feature_names": ["rms", "kurtosis", "crest_factor", "variance", "mean"]
    }
    await client.post("/api/v1/anomalies/train", json=training_data)
    
    # Détecter
    response = await client.post("/api/v1/anomalies/detect", json=sample_detection_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "ASSET001"
    assert "final_score" in data
    assert "is_anomaly" in data
    assert "criticality" in data
    assert "scores" in data
    assert 0 <= data["final_score"] <= 1


@pytest.mark.asyncio
async def test_detect_anomalies_batch(client, sample_training_data):
    """Test de la détection batch"""
    # Entraîner d'abord
    training_data = {
        "data": sample_training_data,
        "feature_names": ["rms", "kurtosis", "crest_factor", "variance", "mean"]
    }
    await client.post("/api/v1/anomalies/train", json=training_data)
    
    # Détecter batch
    requests = [
        {
            "asset_id": f"ASSET{i:03d}",
            "features": {
                "rms": 0.5 + i * 0.1,
                "kurtosis": 0.3,
                "crest_factor": 1.2,
                "variance": 0.1,
                "mean": 0.0
            }
        }
        for i in range(3)
    ]
    
    response = await client.post("/api/v1/anomalies/detect/batch", json=requests)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for result in data:
        assert "final_score" in result
        assert "is_anomaly" in result


@pytest.mark.asyncio
async def test_get_status(client, sample_training_data):
    """Test de récupération du statut"""
    # Avant entraînement
    response = await client.get("/api/v1/anomalies/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "models" in data
    
    # Entraîner
    training_data = {
        "data": sample_training_data,
        "feature_names": ["rms", "kurtosis", "crest_factor", "variance", "mean"]
    }
    await client.post("/api/v1/anomalies/train", json=training_data)
    
    # Après entraînement
    response = await client.get("/api/v1/anomalies/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


@pytest.mark.asyncio
async def test_get_anomalies(client):
    """Test de récupération de l'historique (placeholder)"""
    response = await client.get("/api/v1/anomalies/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "filters" in data


@pytest.mark.asyncio
async def test_get_metrics(client):
    """Test de récupération des métriques (placeholder)"""
    response = await client.get("/api/v1/anomalies/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_train_models_invalid_data(client):
    """Test que l'entraînement échoue avec des données invalides"""
    # Données 1D au lieu de 2D
    training_data = {
        "data": [1, 2, 3, 4, 5]
    }
    
    response = await client.post("/api/v1/anomalies/train", json=training_data)
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_train_models_missing_data(client):
    """Test que l'entraînement échoue sans données"""
    training_data = {}
    
    response = await client.post("/api/v1/anomalies/train", json=training_data)
    
    assert response.status_code == 400
    data = response.json()
    assert "data" in data["detail"]



