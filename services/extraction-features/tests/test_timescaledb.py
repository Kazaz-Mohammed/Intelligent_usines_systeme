"""
Tests pour le service TimescaleDB
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.database.timescaledb import TimescaleDBService
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector


class TestTimescaleDBService:
    """Tests pour TimescaleDBService"""

    @pytest.fixture
    def timescaledb_service(self):
        return TimescaleDBService()

    @pytest.fixture
    def sample_feature(self):
        """Feature d'exemple"""
        return ExtractedFeature(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            feature_name="rms",
            feature_value=25.5,
            feature_type="temporal",
            metadata={"source": "test"}
        )

    @pytest.fixture
    def sample_feature_vector(self):
        """Vecteur de features d'exemple"""
        base_time = datetime.utcnow()
        return ExtractedFeaturesVector(
            feature_vector_id="fv_001",
            timestamp=base_time,
            asset_id="ASSET001",
            start_time=base_time,
            end_time=base_time,
            features={
                "rms": 25.5,
                "kurtosis": 2.3,
                "skewness": 0.5
            },
            feature_metadata={"window_size": 100}
        )

    @pytest.mark.timescaledb
    def test_insert_extracted_feature(self, timescaledb_service, sample_feature):
        """Test insertion d'une feature"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            timescaledb_service.insert_extracted_feature(sample_feature)
            # Si l'insertion réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Insertion échouée: {e}")

    @pytest.mark.timescaledb
    def test_insert_extracted_features_batch(self, timescaledb_service, sample_feature):
        """Test insertion d'un lot de features"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        features = [sample_feature] * 10

        try:
            timescaledb_service.insert_extracted_features_batch(features)
            # Si l'insertion réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Insertion batch échouée: {e}")

    @pytest.mark.timescaledb
    def test_insert_feature_vector(self, timescaledb_service, sample_feature_vector):
        """Test insertion d'un vecteur de features"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            timescaledb_service.insert_feature_vector(sample_feature_vector)
            # Si l'insertion réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Insertion vecteur échouée: {e}")

    @pytest.mark.timescaledb
    def test_get_features_by_asset(self, timescaledb_service):
        """Test récupération des features par asset"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            features = timescaledb_service.get_features_by_asset("ASSET001", limit=10)
            assert isinstance(features, list)
            # Si des features existent, vérifier leur structure
            if features:
                assert all(isinstance(f, ExtractedFeature) for f in features)
        except Exception as e:
            pytest.fail(f"Récupération échouée: {e}")

    @patch('app.database.timescaledb.SimpleConnectionPool')
    def test_insert_extracted_feature_with_mock(self, mock_pool, sample_feature):
        """Test insertion avec mock"""
        # Créer un mock de la connexion
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None

        # Créer un mock du pool
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool_instance.putconn = MagicMock()
        mock_pool.return_value = mock_pool_instance

        # Créer le service
        service = TimescaleDBService()
        service.pool = mock_pool_instance

        # Tester
        service.insert_extracted_feature(sample_feature)

        # Vérifier que execute a été appelé
        assert mock_cursor.execute.called
        assert mock_conn.commit.called

    def test_get_connection_context_manager(self, timescaledb_service):
        """Test context manager pour obtenir une connexion"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            with timescaledb_service.get_connection() as conn:
                assert conn is not None
        except Exception as e:
            pytest.fail(f"Connexion échouée: {e}")

    def test_close(self, timescaledb_service):
        """Test fermeture du service"""
        # Ne pas fermer si le pool n'est pas initialisé
        if timescaledb_service.pool:
            timescaledb_service.close()
            # Vérifier que le pool est fermé
            assert timescaledb_service.pool is None
        else:
            # Si le pool n'est pas initialisé, close() devrait gérer gracieusement
            timescaledb_service.close()
            assert timescaledb_service.pool is None

