"""
Tests pour le service PostgreSQL
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import psycopg2
from psycopg2.extras import RealDictCursor

from app.database.postgresql import PostgreSQLService
from app.models.anomaly_data import AnomalyDetectionResult, AnomalyScore, CriticalityLevel


@pytest.fixture
def postgresql_service():
    """Fixture pour créer une instance du service PostgreSQL"""
    with patch('app.database.postgresql.ThreadedConnectionPool') as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        service = PostgreSQLService()
        service.connection_pool = mock_pool
        return service


@pytest.fixture
def mock_connection():
    """Fixture pour créer une connexion mock"""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn, cursor


@pytest.fixture
def sample_anomaly_result():
    """Fixture pour créer un résultat d'anomalie de test"""
    return AnomalyDetectionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp=datetime.now(timezone.utc),
        final_score=0.75,
        is_anomaly=True,
        criticality=CriticalityLevel.HIGH,
        scores=[
            AnomalyScore(model_name="isolation_forest", score=0.8, threshold=0.5, is_anomaly=True),
            AnomalyScore(model_name="one_class_svm", score=0.7, threshold=0.5, is_anomaly=True),
            AnomalyScore(model_name="lstm_autoencoder", score=0.75, threshold=0.5, is_anomaly=True)
        ],
        features={"rms": 10.5, "kurtosis": 2.3},
        metadata={"source": "test"}
    )


class TestPostgreSQLService:
    """Tests pour PostgreSQLService"""
    
    def test_init(self):
        """Test initialisation du service"""
        with patch('app.database.postgresql.ThreadedConnectionPool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool
            
            with patch.object(PostgreSQLService, '_create_tables') as mock_create:
                service = PostgreSQLService()
                assert service.connection_pool == mock_pool
                mock_create.assert_called_once()
    
    def test_get_connection(self, postgresql_service, mock_connection):
        """Test obtention d'une connexion du pool"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        
        # Reset le mock car _create_tables a déjà été appelé
        postgresql_service.connection_pool.getconn.reset_mock()
        
        with postgresql_service.get_connection() as c:
            assert c == conn
        
        postgresql_service.connection_pool.getconn.assert_called_once()
        postgresql_service.connection_pool.putconn.assert_called_once_with(conn)
    
    def test_get_connection_error(self, postgresql_service):
        """Test gestion d'erreur lors de l'obtention d'une connexion"""
        postgresql_service.connection_pool = None
        
        with pytest.raises(RuntimeError, match="Pool de connexions PostgreSQL non initialisé"):
            with postgresql_service.get_connection():
                pass
    
    def test_get_connection_rollback_on_error(self, postgresql_service, mock_connection):
        """Test rollback en cas d'erreur"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        
        with pytest.raises(ValueError):
            with postgresql_service.get_connection():
                raise ValueError("Test error")
        
        conn.rollback.assert_called_once()
        postgresql_service.connection_pool.putconn.assert_called_once_with(conn)
    
    def test_create_tables(self, postgresql_service, mock_connection):
        """Test création des tables"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        
        postgresql_service._create_tables()
        
        # Vérifier que execute a été appelé plusieurs fois (pour chaque statement)
        assert cursor.execute.call_count >= 7  # Au moins 7 statements (CREATE TABLE + 6 CREATE INDEX)
    
    def test_insert_anomaly_success(self, postgresql_service, mock_connection, sample_anomaly_result):
        """Test insertion réussie d'une anomalie"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.fetchone.return_value = (123,)
        
        anomaly_id = postgresql_service.insert_anomaly(sample_anomaly_result)
        
        assert anomaly_id == 123
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "INSERT INTO anomaly_detections" in args[0]
        assert args[1][0] == "ASSET001"  # asset_id
        assert args[1][1] == "SENSOR001"  # sensor_id
        assert args[1][4] is True  # is_anomaly
        assert args[1][5] == "high"  # criticality
    
    def test_insert_anomaly_with_dict(self, postgresql_service, mock_connection):
        """Test insertion avec un dictionnaire"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.fetchone.return_value = (456,)
        
        anomaly_dict = {
            "asset_id": "ASSET002",
            "sensor_id": "SENSOR002",
            "timestamp": datetime.now(timezone.utc),
            "final_score": 0.65,
            "is_anomaly": True,
            "criticality": "medium",
            "scores": [{"model_name": "isolation_forest", "score": 0.7, "threshold": 0.5, "is_anomaly": True}],
            "features": {"rms": 8.5},
            "metadata": {}
        }
        
        anomaly_id = postgresql_service.insert_anomaly(anomaly_dict)
        
        assert anomaly_id == 456
        cursor.execute.assert_called_once()
    
    def test_insert_anomaly_with_string_timestamp(self, postgresql_service, mock_connection):
        """Test insertion avec timestamp en string"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.fetchone.return_value = (789,)
        
        anomaly_dict = {
            "asset_id": "ASSET003",
            "timestamp": "2024-01-01T12:00:00Z",
            "final_score": 0.5,
            "is_anomaly": False,
            "criticality": "low",
            "scores": [],
            "features": {}
        }
        
        anomaly_id = postgresql_service.insert_anomaly(anomaly_dict)
        
        assert anomaly_id == 789
        # Vérifier que le timestamp a été converti
        args = cursor.execute.call_args[0]
        assert isinstance(args[1][2], datetime)  # timestamp
    
    def test_insert_anomaly_error(self, postgresql_service, mock_connection):
        """Test gestion d'erreur lors de l'insertion"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.execute.side_effect = psycopg2.Error("Database error")
        
        anomaly_dict = {
            "asset_id": "ASSET004",
            "timestamp": datetime.now(timezone.utc),
            "final_score": 0.5,
            "is_anomaly": False,
            "criticality": "low",
            "scores": [],
            "features": {}
        }
        
        anomaly_id = postgresql_service.insert_anomaly(anomaly_dict)
        
        assert anomaly_id is None
    
    def test_get_anomalies_no_filters(self, postgresql_service, mock_connection):
        """Test récupération des anomalies sans filtres"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        
        # Mock RealDictCursor - créer un dict simple
        mock_row = {
            'id': 1,
            'asset_id': 'ASSET001',
            'sensor_id': 'SENSOR001',
            'timestamp': datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            'final_score': 0.75,
            'is_anomaly': True,
            'criticality': 'high',
            'scores': json.dumps([{"model_name": "isolation_forest", "score": 0.8}]),
            'features': json.dumps({"rms": 10.5}),
            'metadata': json.dumps({}),
            'created_at': datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        }
        
        # Simuler RealDictCursor qui retourne des dicts
        cursor.__class__ = RealDictCursor
        cursor.fetchall.return_value = [mock_row]
        
        anomalies = postgresql_service.get_anomalies()
        
        assert len(anomalies) == 1
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "SELECT" in args[0]
        assert "WHERE 1=1" in args[0]
    
    def test_get_anomalies_with_filters(self, postgresql_service, mock_connection):
        """Test récupération avec filtres"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        
        cursor.__class__ = RealDictCursor
        cursor.fetchall.return_value = []
        
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        
        anomalies = postgresql_service.get_anomalies(
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            start_date=start_date,
            end_date=end_date,
            is_anomaly=True,
            criticality="high",
            limit=50,
            offset=10
        )
        
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "asset_id = %s" in args[0]
        assert "sensor_id = %s" in args[0]
        assert "timestamp >= %s" in args[0]
        assert "timestamp <= %s" in args[0]
        assert "is_anomaly = %s" in args[0]
        assert "criticality = %s" in args[0]
        assert len(args[1]) == 8  # 6 filtres + limit + offset
    
    def test_get_anomalies_error(self, postgresql_service, mock_connection):
        """Test gestion d'erreur lors de la récupération"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.execute.side_effect = psycopg2.Error("Database error")
        
        anomalies = postgresql_service.get_anomalies()
        
        assert anomalies == []
    
    def test_get_anomaly_count_no_filters(self, postgresql_service, mock_connection):
        """Test comptage sans filtres"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.fetchone.return_value = (42,)
        
        count = postgresql_service.get_anomaly_count()
        
        assert count == 42
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "COUNT(*)" in args[0]
        assert "WHERE 1=1" in args[0]
    
    def test_get_anomaly_count_with_filters(self, postgresql_service, mock_connection):
        """Test comptage avec filtres"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.fetchone.return_value = (10,)
        
        count = postgresql_service.get_anomaly_count(
            asset_id="ASSET001",
            is_anomaly=True,
            criticality="high"
        )
        
        assert count == 10
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0]
        assert "asset_id = %s" in args[0]
        assert "is_anomaly = %s" in args[0]
        assert "criticality = %s" in args[0]
        assert len(args[1]) == 3
    
    def test_get_anomaly_count_error(self, postgresql_service, mock_connection):
        """Test gestion d'erreur lors du comptage"""
        conn, cursor = mock_connection
        postgresql_service.connection_pool.getconn.return_value = conn
        postgresql_service.connection_pool.putconn = MagicMock()
        cursor.execute.side_effect = psycopg2.Error("Database error")
        
        count = postgresql_service.get_anomaly_count()
        
        assert count == 0
    
    def test_close(self, postgresql_service):
        """Test fermeture du pool"""
        mock_pool = MagicMock()
        postgresql_service.connection_pool = mock_pool
        
        postgresql_service.close()
        
        mock_pool.closeall.assert_called_once()
        assert postgresql_service.connection_pool is None
    
    def test_close_no_pool(self, postgresql_service):
        """Test fermeture quand le pool n'existe pas"""
        postgresql_service.connection_pool = None
        
        # Ne doit pas lever d'exception
        postgresql_service.close()
        
        assert postgresql_service.connection_pool is None

