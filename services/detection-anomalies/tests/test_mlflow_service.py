"""
Tests pour le service MLflow
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.mlflow_service import MLflowService
from app.config import settings


@pytest.fixture
def mlflow_service():
    """Fixture pour créer un service MLflow"""
    with patch('app.services.mlflow_service.mlflow'):
        with patch('app.services.mlflow_service.MlflowClient'):
            service = MLflowService()
            yield service


def test_mlflow_service_init_disabled():
    """Test de l'initialisation avec MLflow désactivé"""
    with patch('app.config.settings.mlflow_enabled', False):
        service = MLflowService()
        assert not service.enabled


def test_start_run_disabled(mlflow_service):
    """Test que start_run retourne None si MLflow est désactivé"""
    mlflow_service.enabled = False
    assert mlflow_service.start_run() is None


def test_log_params_disabled(mlflow_service):
    """Test que log_params ne fait rien si MLflow est désactivé"""
    mlflow_service.enabled = False
    # Ne devrait pas lever d'exception
    mlflow_service.log_params({"param1": "value1"})


def test_log_metrics_disabled(mlflow_service):
    """Test que log_metrics ne fait rien si MLflow est désactivé"""
    mlflow_service.enabled = False
    # Ne devrait pas lever d'exception
    mlflow_service.log_metrics({"metric1": 0.5})


def test_load_model_disabled(mlflow_service):
    """Test que load_model retourne None si MLflow est désactivé"""
    mlflow_service.enabled = False
    assert mlflow_service.load_model("test_model") is None

