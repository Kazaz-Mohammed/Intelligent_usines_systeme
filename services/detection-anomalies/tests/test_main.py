"""
Tests pour le point d'entrée FastAPI
"""
import pytest
import httpx
from httpx import ASGITransport
from app.main import app


@pytest.fixture
async def client():
    """Client de test FastAPI"""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test du endpoint root"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "service" in data
    assert "version" in data
    assert data["service"] == "detection-anomalies-service"


@pytest.mark.asyncio
async def test_health_check(client):
    """Test du health check"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "detection-anomalies-service"
    assert data["version"] == "0.1.0"

