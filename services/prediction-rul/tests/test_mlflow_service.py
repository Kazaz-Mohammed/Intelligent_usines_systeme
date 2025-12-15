"""
Tests pour le service MLflow
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.mlflow_service import MLflowService
from app.config import settings


@pytest.fixture
def mlflow_service():
    """Fixture pour créer une instance du service"""
    with patch('app.services.mlflow_service.mlflow') as mock_mlflow:
        with patch('app.services.mlflow_service.settings') as mock_settings:
            mock_settings.mlflow_enabled = True
            mock_settings.mlflow_tracking_uri = "http://localhost:5000"
            mock_settings.mlflow_experiment_name = "rul-prediction"
            
            service = MLflowService()
            service.enabled = True
            return service


@pytest.fixture
def mlflow_service_disabled():
    """Fixture pour un service MLflow désactivé"""
    with patch('app.services.mlflow_service.settings') as mock_settings:
        mock_settings.mlflow_enabled = False
        service = MLflowService()
        return service


class TestMLflowService:
    """Tests pour MLflowService"""
    
    def test_init_enabled(self, mlflow_service):
        """Test initialisation avec MLflow activé"""
        assert mlflow_service.enabled is True
        assert mlflow_service.tracking_uri == "http://localhost:5000"
        assert mlflow_service.experiment_name == "rul-prediction"
    
    def test_init_disabled(self, mlflow_service_disabled):
        """Test initialisation avec MLflow désactivé"""
        assert mlflow_service_disabled.enabled is False
    
    @patch('app.services.mlflow_service.mlflow')
    def test_start_run(self, mock_mlflow, mlflow_service):
        """Test démarrage d'une run"""
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run
        
        run = mlflow_service.start_run(run_name="test_run", tags={"tag1": "value1"})
        
        assert run is not None
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run", tags={"tag1": "value1"})
    
    def test_start_run_disabled(self, mlflow_service_disabled):
        """Test démarrage d'une run avec MLflow désactivé"""
        run = mlflow_service_disabled.start_run()
        assert run is None
    
    @patch('app.services.mlflow_service.mlflow')
    def test_end_run(self, mock_mlflow, mlflow_service):
        """Test fin d'une run"""
        mlflow_service.end_run(status="FINISHED")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
    
    @patch('app.services.mlflow_service.mlflow')
    def test_log_params(self, mock_mlflow, mlflow_service):
        """Test logging de paramètres"""
        params = {"epochs": 100, "learning_rate": 0.001}
        mlflow_service.log_params(params)
        mock_mlflow.log_params.assert_called_once_with(params)
    
    @patch('app.services.mlflow_service.mlflow')
    def test_log_metrics(self, mock_mlflow, mlflow_service):
        """Test logging de métriques"""
        metrics = {"mae": 5.0, "rmse": 7.0}
        mlflow_service.log_metrics(metrics)
        mock_mlflow.log_metrics.assert_called_once_with(metrics)
    
    @patch('app.services.mlflow_service.mlflow')
    def test_log_metrics_with_step(self, mock_mlflow, mlflow_service):
        """Test logging de métriques avec step"""
        metrics = {"loss": 0.5}
        mlflow_service.log_metrics(metrics, step=10)
        assert mock_mlflow.log_metric.call_count == 1
    
    @patch('app.services.mlflow_service.mlflow')
    def test_log_model_pytorch(self, mock_mlflow, mlflow_service):
        """Test logging d'un modèle PyTorch"""
        mock_model = MagicMock()
        mock_model.state_dict = MagicMock()
        
        mlflow_service.log_model(mock_model, "model_path", "LSTM_Model")
        
        mock_mlflow.pytorch.log_model.assert_called_once()
        mock_mlflow.get_artifact_uri.assert_called_once()
    
    @patch('app.services.mlflow_service.mlflow')
    def test_load_model_pytorch(self, mock_mlflow, mlflow_service):
        """Test chargement d'un modèle PyTorch"""
        mock_model = MagicMock()
        mock_mlflow.pytorch.load_model.return_value = mock_model
        
        model = mlflow_service.load_model("models:/LSTM_Model/1", model_type="pytorch")
        
        assert model is not None
        mock_mlflow.pytorch.load_model.assert_called_once_with("models:/LSTM_Model/1")
    
    @patch('app.services.mlflow_service.mlflow')
    def test_get_run_metrics(self, mock_mlflow, mlflow_service):
        """Test récupération des métriques d'une run"""
        mock_run = MagicMock()
        mock_run.data.metrics = {"mae": 5.0, "rmse": 7.0}
        mock_mlflow.get_run.return_value = mock_run
        
        metrics = mlflow_service.get_run_metrics("run-id")
        
        assert "mae" in metrics
        assert metrics["mae"] == 5.0
    
    @patch('app.services.mlflow_service.mlflow')
    def test_get_run_params(self, mock_mlflow, mlflow_service):
        """Test récupération des paramètres d'une run"""
        mock_run = MagicMock()
        mock_run.data.params = {"epochs": "100", "lr": "0.001"}
        mock_mlflow.get_run.return_value = mock_run
        
        params = mlflow_service.get_run_params("run-id")
        
        assert "epochs" in params
        assert params["epochs"] == "100"
    
    @patch('mlflow.tracking.MlflowClient')
    def test_get_latest_model_version(self, MockClient, mlflow_service):
        """Test récupération de la dernière version"""
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_client.get_latest_versions.return_value = [mock_version]
        MockClient.return_value = mock_client
        
        version = mlflow_service.get_latest_model_version("LSTM_Model")
        
        assert version == 3
    
    def test_get_model_uri_with_version(self, mlflow_service):
        """Test récupération de l'URI avec version"""
        uri = mlflow_service.get_model_uri("LSTM_Model", version=2)
        assert uri == "models:/LSTM_Model/2"
    
    def test_get_model_uri_with_stage(self, mlflow_service):
        """Test récupération de l'URI avec stage"""
        uri = mlflow_service.get_model_uri("LSTM_Model", stage="Production")
        assert uri == "models:/LSTM_Model/Production"
    
    @patch('app.services.mlflow_service.mlflow')
    def test_search_runs(self, mock_mlflow, mlflow_service):
        """Test recherche de runs"""
        import pandas as pd
        
        mock_runs = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "status": ["FINISHED", "FINISHED"],
            "metrics.mae": [5.0, 6.0],
            "params.epochs": ["100", "200"],
            "start_time": [1000, 2000],
            "end_time": [1500, 2500]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        runs = mlflow_service.search_runs(filter_string="metrics.mae < 10.0")
        
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run1"
    
    def test_get_service_info(self, mlflow_service):
        """Test récupération des informations du service"""
        info = mlflow_service.get_service_info()
        
        assert "enabled" in info
        assert "tracking_uri" in info
        assert "experiment_name" in info

