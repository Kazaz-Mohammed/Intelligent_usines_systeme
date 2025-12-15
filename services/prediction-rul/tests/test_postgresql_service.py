"""
Tests pour le service PostgreSQL
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import json

from app.database.postgresql import PostgreSQLService
from app.models.rul_data import RULPredictionResult
from app.config import settings


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock les paramètres de la base de données pour les tests"""
    with patch('app.config.settings') as mock_settings:
        mock_settings.database_host = "test_host"
        mock_settings.database_port = 5432
        mock_settings.database_name = "test_db"
        mock_settings.database_user = "test_user"
        mock_settings.database_password = "test_password"
        yield mock_settings


@pytest.fixture
def postgresql_service():
    """Fixture pour le service PostgreSQL avec un pool mocké"""
    with patch('psycopg2.pool.ThreadedConnectionPool') as MockPool:
        service = PostgreSQLService()
        service.pool = MockPool.return_value
        yield service


@pytest.fixture
def mock_connection():
    """Fixture pour une connexion et un curseur mockés"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None
    yield mock_conn, mock_cursor


@pytest.fixture
def sample_rul_result():
    """Exemple d'objet RULPredictionResult"""
    return RULPredictionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp=datetime.now(timezone.utc),
        rul_prediction=150.5,
        confidence_interval_lower=140.0,
        confidence_interval_upper=160.0,
        confidence_level=0.95,
        uncertainty=10.0,
        model_used="ensemble",
        model_scores={"lstm": 150.0, "gru": 151.0},
        features={"rms": 10.5, "kurtosis": 2.3},
        metadata={"source": "test"}
    )


class TestPostgreSQLService:
    """Tests pour PostgreSQLService"""
    
    def test_init(self, postgresql_service):
        """Test initialisation du service"""
        postgresql_service.pool.getconn.assert_called_once()  # Appelé par _create_tables
        postgresql_service.pool.putconn.assert_called_once()
        assert postgresql_service.pool is not None
    
    def test_get_connection(self, postgresql_service, mock_connection):
        """Test obtention d'une connexion"""
        conn, _ = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        
        retrieved_conn = postgresql_service.get_connection()
        
        assert retrieved_conn == conn
        postgresql_service.pool.getconn.assert_called()
    
    def test_create_tables(self, postgresql_service, mock_connection):
        """Test création des tables"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        
        postgresql_service._create_tables()
        
        # Vérifier que execute a été appelé plusieurs fois
        assert cursor.execute.call_count >= 7  # CREATE TABLE + 6 CREATE INDEX
        conn.commit.assert_called_once()
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_insert_rul_prediction_success(self, postgresql_service, mock_connection, sample_rul_result):
        """Test insertion réussie d'une prédiction RUL"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        cursor.fetchone.return_value = (123,)
        
        prediction_id = postgresql_service.insert_rul_prediction(sample_rul_result)
        
        assert prediction_id == 123
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "INSERT INTO rul_predictions" in args[0]
        assert args[1][0] == "ASSET001"  # asset_id
        assert args[1][3] == 150.5  # predicted_rul (rul_prediction)
        assert args[1][8] == "ensemble"  # model_used
        conn.commit.assert_called_once()
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_insert_rul_prediction_error(self, postgresql_service, mock_connection, sample_rul_result):
        """Test erreur lors de l'insertion"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        cursor.execute.side_effect = Exception("Insert error")
        
        prediction_id = postgresql_service.insert_rul_prediction(sample_rul_result)
        
        assert prediction_id is None
        conn.rollback.assert_called_once()
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_get_rul_predictions_no_filters(self, postgresql_service, mock_connection, sample_rul_result):
        """Test récupération de prédictions sans filtres"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        
        row_dict = sample_rul_result.model_dump()
        row_dict['id'] = 1
        row_dict['created_at'] = datetime.now(timezone.utc)
        row_dict['model_scores'] = json.dumps(row_dict['model_scores'])
        row_dict['features'] = json.dumps(row_dict['features'])
        row_dict['metadata'] = json.dumps(row_dict['metadata'])
        
        cursor.fetchone.side_effect = [{'count': 1}, row_dict]
        cursor.fetchall.return_value = [row_dict]
        
        result = postgresql_service.get_rul_predictions()
        
        assert result['total'] == 1
        assert len(result['predictions']) == 1
        assert isinstance(result['predictions'][0], RULPredictionResult)
        assert result['predictions'][0].asset_id == "ASSET001"
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_get_rul_predictions_with_filters(self, postgresql_service, mock_connection, sample_rul_result):
        """Test récupération avec filtres"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        
        row_dict = sample_rul_result.model_dump()
        row_dict['id'] = 1
        row_dict['created_at'] = datetime.now(timezone.utc)
        row_dict['model_scores'] = json.dumps(row_dict['model_scores'])
        row_dict['features'] = json.dumps(row_dict['features'])
        row_dict['metadata'] = json.dumps(row_dict['metadata'])
        
        cursor.fetchone.side_effect = [{'count': 1}, row_dict]
        cursor.fetchall.return_value = [row_dict]
        
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        
        result = postgresql_service.get_rul_predictions(
            asset_id="ASSET001",
            model_used="ensemble",
            start_date=start_date,
            end_date=end_date,
            limit=10,
            offset=0
        )
        
        assert result['total'] == 1
        assert len(result['predictions']) == 1
        assert result['predictions'][0].asset_id == "ASSET001"
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_get_rul_prediction_count(self, postgresql_service, mock_connection):
        """Test comptage de prédictions"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        cursor.fetchone.return_value = (100,)
        
        count = postgresql_service.get_rul_prediction_count()
        
        assert count == 100
        cursor.execute.assert_called_once()
        assert "SELECT COUNT(*) FROM rul_predictions" in cursor.execute.call_args[0][0]
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_get_latest_rul_prediction(self, postgresql_service, mock_connection, sample_rul_result):
        """Test récupération de la dernière prédiction"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        
        row_dict = sample_rul_result.model_dump()
        row_dict['id'] = 1
        row_dict['created_at'] = datetime.now(timezone.utc)
        row_dict['model_scores'] = json.dumps(row_dict['model_scores'])
        row_dict['features'] = json.dumps(row_dict['features'])
        row_dict['metadata'] = json.dumps(row_dict['metadata'])
        
        cursor.fetchone.return_value = row_dict
        
        result = postgresql_service.get_latest_rul_prediction("ASSET001")
        
        assert result is not None
        assert isinstance(result, RULPredictionResult)
        assert result.asset_id == "ASSET001"
        assert result.rul_prediction == 150.5
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_get_latest_rul_prediction_not_found(self, postgresql_service, mock_connection):
        """Test récupération quand aucune prédiction n'existe"""
        conn, cursor = mock_connection
        postgresql_service.pool.getconn.return_value = conn
        postgresql_service.pool.putconn = MagicMock()
        cursor.fetchone.return_value = None
        
        result = postgresql_service.get_latest_rul_prediction("ASSET999")
        
        assert result is None
        postgresql_service.pool.putconn.assert_called_once_with(conn)
    
    def test_close(self, postgresql_service):
        """Test fermeture du service"""
        postgresql_service.close()
        postgresql_service.pool.closeall.assert_called_once()
        assert postgresql_service.pool is None
    
    def test_close_no_pool(self, postgresql_service):
        """Test fermeture quand le pool n'est pas initialisé"""
        postgresql_service.pool = None
        postgresql_service.close()
        assert postgresql_service.pool is None

