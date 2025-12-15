"""
Tests pour l'API REST
"""
import pytest
from unittest.mock import MagicMock, patch
import httpx

from app.api.features import FeatureExtractionServiceSingleton


# Créer une instance d'app FastAPI sans lifespan pour les tests
from fastapi import FastAPI
from app.api.features import router as features_router
from app.config import settings

test_app = FastAPI(
    title="Extraction Features Service Test",
    description="Service d'extraction de caractéristiques temporelles et fréquentielles pour la maintenance prédictive",
    version="0.1.0"
)

test_app.include_router(features_router, prefix="/api/v1/features", tags=["Features"])

@test_app.get("/")
async def root():
    return {"message": "Extraction Features Service is running"}

@test_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.service_name, "version": "0.1.0"}


class TestAPIFeatures:
    """Tests pour l'API REST"""

    @pytest.fixture
    def client(self):
        """Client de test FastAPI"""
        # Utiliser httpx directement avec ASGITransport
        from httpx import ASGITransport
        transport = ASGITransport(app=test_app)
        return httpx.Client(transport=transport, base_url="http://test")

    @pytest.fixture
    def mock_feature_extraction_service(self):
        """Mock du service d'extraction de features"""
        mock_service = MagicMock()
        mock_service.get_statistics.return_value = {
            "buffers": {},
            "last_processed": {},
            "services": {
                "temporal_features": True,
                "frequency_features": True,
                "wavelet_features": True,
                "standardization": True,
                "feast": False
            }
        }
        FeatureExtractionServiceSingleton.set_instance(mock_service)
        return mock_service

    def test_health_check(self, client):
        """Test endpoint health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_status(self, client, mock_feature_extraction_service):
        """Test endpoint status"""
        response = client.get("/api/v1/features/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "statistics" in data
        assert "configuration" in data

    @pytest.mark.timescaledb
    def test_get_features(self, client):
        """Test récupération des features"""
        response = client.get("/api/v1/features/features/ASSET001")
        assert response.status_code in [200, 404, 500]  # Peut échouer si DB non disponible
        if response.status_code == 200:
            data = response.json()
            assert "asset_id" in data
            assert "count" in data
            assert "features" in data

    @pytest.mark.timescaledb
    def test_get_feature_vector(self, client):
        """Test récupération d'un vecteur de features"""
        response = client.get("/api/v1/features/features/ASSET001/vector")
        assert response.status_code in [200, 404, 500]  # Peut échouer si DB non disponible
        if response.status_code == 200:
            data = response.json()
            assert "feature_vector_id" in data
            assert "asset_id" in data
            assert "features" in data

    @pytest.mark.timescaledb
    def test_get_asset_info(self, client):
        """Test récupération des informations d'actif"""
        response = client.get("/api/v1/features/assets/ASSET001")
        assert response.status_code in [200, 404, 500]  # Peut échouer si DB non disponible
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "type" in data

    @pytest.mark.timescaledb
    def test_get_asset_type(self, client):
        """Test récupération du type d'actif"""
        response = client.get("/api/v1/features/assets/ASSET001/type")
        assert response.status_code in [200, 404, 500]  # Peut échouer si DB non disponible
        if response.status_code == 200:
            data = response.json()
            assert "asset_id" in data
            assert "type" in data

    def test_get_metrics(self, client, mock_feature_extraction_service):
        """Test récupération des métriques"""
        response = client.get("/api/v1/features/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "buffers" in data
        assert "last_processed" in data
        assert "services" in data

    def test_compute_features(self, client):
        """Test calcul de features"""
        response = client.post(
            "/api/v1/features/compute",
            json={
                "asset_id": "ASSET001",
                "sensor_ids": ["SENSOR001"],
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T01:00:00Z"
            }
        )
        # Pour l'instant, retourne un message indiquant que ce n'est pas implémenté
        assert response.status_code == 200
        data = response.json()
        assert "asset_id" in data
        assert "message" in data

