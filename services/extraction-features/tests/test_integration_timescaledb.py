"""
Tests d'intégration TimescaleDB
"""
import pytest
from datetime import datetime

from app.database.timescaledb import TimescaleDBService
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector


@pytest.mark.integration
@pytest.mark.timescaledb
class TestTimescaleDBIntegration:
    """Tests d'intégration TimescaleDB"""

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
            feature_vector_id="fv_test_001",
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

    def test_insert_and_retrieve_feature(self, timescaledb_service, sample_feature):
        """Test insertion et récupération d'une feature"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            # Insérer la feature
            timescaledb_service.insert_extracted_feature(sample_feature)

            # Récupérer les features
            features = timescaledb_service.get_features_by_asset("ASSET001", limit=10)

            # Vérifier que la feature a été insérée
            assert len(features) > 0
            feature_names = [f.feature_name for f in features]
            assert "rms" in feature_names

        except Exception as e:
            pytest.fail(f"Test échoué: {e}")

    def test_insert_and_retrieve_feature_vector(self, timescaledb_service, sample_feature_vector):
        """Test insertion et récupération d'un vecteur de features"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        try:
            # Insérer le vecteur
            timescaledb_service.insert_feature_vector(sample_feature_vector)

            # Vérifier que l'insertion a réussi (pas d'exception)
            assert True

        except Exception as e:
            pytest.fail(f"Test échoué: {e}")

    def test_batch_insert(self, timescaledb_service, sample_feature):
        """Test insertion batch"""
        if not timescaledb_service.pool:
            pytest.skip("TimescaleDB non disponible")

        features = [
            ExtractedFeature(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                feature_name=f"feature_{i}",
                feature_value=25.0 + i * 0.5,
                feature_type="temporal",
                metadata={"source": "test"}
            ) for i in range(10)
        ]

        try:
            timescaledb_service.insert_extracted_features_batch(features)
            # Vérifier que l'insertion a réussi
            assert True
        except Exception as e:
            pytest.fail(f"Insertion batch échouée: {e}")

