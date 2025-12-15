"""
Tests pour l'API RUL
"""
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.main import app
from app.api.rul import get_rul_prediction_service
from app.models.rul_data import RULPredictionRequest, RULPredictionResult
from app.services.rul_prediction_service import RULPredictionService


@pytest.fixture
def mock_rul_service():
    """Fixture pour créer un service mock"""
    service = Mock(spec=RULPredictionService)
    service.is_ready.return_value = True
    service.get_model_status.return_value = {
        "lstm": {"available": True, "trained": True, "model_type": "lstm"}
    }
    # Ajouter l'attribut models pour les tests d'entraînement
    mock_model = Mock()
    mock_model.train = Mock()
    service.models = {"lstm": mock_model}
    return service


@pytest.fixture
def client(mock_rul_service):
    """Fixture pour créer un client HTTP avec service mocké"""
    # Override la dépendance FastAPI
    app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
    
    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    
    # Nettoyer après le test
    app.dependency_overrides.clear()


@pytest.fixture
def sample_prediction_result():
    """Fixture pour créer un résultat de prédiction"""
    return RULPredictionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp="2024-01-01T12:00:00Z",
        rul_prediction=150.5,
        confidence_interval_lower=140.0,
        confidence_interval_upper=160.0,
        confidence_level=0.95,
        uncertainty=10.0,
        model_used="ensemble",
        model_scores={"lstm": 150.0, "gru": 151.0},
        features={"rms": 10.5},
        metadata={}
    )


class TestAPIRUL:
    """Tests pour l'API RUL"""
    
    @pytest.mark.asyncio
    async def test_predict_rul_not_ready(self, mock_rul_service):
        """Test prédiction quand aucun modèle n'est entraîné"""
        mock_rul_service.is_ready.return_value = False
        
        app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                request_data = {
                    "asset_id": "ASSET001",
                    "features": {"rms": 10.5, "kurtosis": 2.3}
                }
                
                response = await client.post("/api/v1/rul/predict", json=request_data)
                
                assert response.status_code == 503
                assert "entraîné" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_predict_rul_success(self, mock_rul_service, sample_prediction_result):
        """Test prédiction réussie"""
        mock_rul_service.predict_rul.return_value = sample_prediction_result
        
        app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                request_data = {
                    "asset_id": "ASSET001",
                    "sensor_id": "SENSOR001",
                    "features": {"rms": 10.5, "kurtosis": 2.3, "crest_factor": 3.1}
                }
                
                response = await client.post("/api/v1/rul/predict", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["asset_id"] == "ASSET001"
                assert data["rul_prediction"] >= 0
                assert "confidence_interval_lower" in data
                assert "confidence_interval_upper" in data
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_predict_rul_batch(self, mock_rul_service):
        """Test prédiction batch"""
        from datetime import datetime, timezone
        
        mock_results = [
            RULPredictionResult(
                asset_id=f"ASSET{i:03d}",
                sensor_id="SENSOR001",
                timestamp=datetime.now(timezone.utc),
                rul_prediction=150.0 + i,
                confidence_interval_lower=140.0,
                confidence_interval_upper=160.0,
                confidence_level=0.95,
                uncertainty=10.0,
                model_used="ensemble",
                model_scores={},
                features={"rms": 10.5},
                metadata={}
            )
            for i in range(3)
        ]
        mock_rul_service.predict_rul_batch.return_value = mock_results
        
        app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                requests_data = [
                    {
                        "asset_id": f"ASSET{i:03d}",
                        "features": {"rms": 10.5 + i, "kurtosis": 2.3}
                    }
                    for i in range(3)
                ]
                
                response = await client.post("/api/v1/rul/predict/batch", json=requests_data)
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 3
                assert all("rul_prediction" in item for item in data)
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_train_models(self, mock_rul_service):
        """Test entraînement des modèles"""
        mock_rul_service.train_all_models.return_value = {
            "lstm": {
                "status": "success",
                "metrics": {"train_mae": 5.0, "train_rmse": 7.0}
            }
        }
        
        app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # Ne pas fournir model_name pour entraîner tous les modèles (plus simple)
                request_data = {
                    "training_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    "target_data": [10.0, 20.0],
                    "feature_names": ["rms", "kurtosis", "crest_factor"]
                }
                
                response = await client.post("/api/v1/rul/train", json=request_data)
                
                # Vérifier le statut et les données
                if response.status_code != 200:
                    print(f"Response status: {response.status_code}")
                    print(f"Response body: {response.text}")
                
                assert response.status_code == 200
                data = response.json()
                # Retourne tous les modèles entraînés
                assert len(data) > 0
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_status(self, mock_rul_service):
        """Test récupération du statut"""
        app.dependency_overrides[get_rul_prediction_service] = lambda: mock_rul_service
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/v1/rul/status")
                
                assert response.status_code == 200
                data = response.json()
                assert "ready" in data
                assert "models" in data
                assert data["ready"] is True
        finally:
            app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_get_rul_predictions(self):
        """Test récupération de l'historique"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/rul/")
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "total" in data

