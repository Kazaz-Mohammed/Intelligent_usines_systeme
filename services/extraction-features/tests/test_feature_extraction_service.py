"""
Tests pour le service principal d'extraction de features
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.services.feature_extraction_service import FeatureExtractionService
from app.models.feature_data import (
    PreprocessedDataReference,
    WindowedDataReference,
    ExtractedFeature
)


class TestFeatureExtractionService:
    """Tests pour FeatureExtractionService"""

    @pytest.fixture
    def feature_extraction_service(self):
        return FeatureExtractionService()

    @pytest.fixture
    def sample_preprocessed_data(self):
        """Données prétraitées d'exemple"""
        base_time = datetime.utcnow()
        return [
            PreprocessedDataReference(
                timestamp=base_time,
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i * 0.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={},
                frequency_analysis=None
            ) for i in range(50)
        ]

    @pytest.fixture
    def sample_windowed_data(self):
        """Fenêtre de données d'exemple"""
        base_time = datetime.utcnow()
        return WindowedDataReference(
            window_id="window_001",
            asset_id="ASSET001",
            start_time=base_time,
            end_time=base_time,
            sensor_data={
                "SENSOR001": [
                    PreprocessedDataReference(
                        timestamp=base_time,
                        asset_id="ASSET001",
                        sensor_id="SENSOR001",
                        value=25.0 + i * 0.5,
                        unit="°C",
                        quality=2,
                        source_type="TEST",
                        preprocessing_metadata={},
                        frequency_analysis=None
                    ) for i in range(50)
                ]
            },
            metadata={}
        )

    @pytest.mark.asyncio
    async def test_process_preprocessed_data_streaming(self, feature_extraction_service, sample_preprocessed_data):
        """Test traitement de données prétraitées en mode streaming"""
        # Mock des services dépendants
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_extracted_features_batch') as mock_insert:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_extracted_features_batch') as mock_publish:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    await feature_extraction_service.process_preprocessed_data(
                        sample_preprocessed_data,
                        mode="streaming"
                    )

                    # Vérifier que les méthodes ont été appelées
                    assert mock_insert.called
                    assert mock_publish.called

    @pytest.mark.asyncio
    async def test_process_windowed_data(self, feature_extraction_service, sample_windowed_data):
        """Test traitement de fenêtres de données"""
        # Mock des services dépendants
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_feature_vector') as mock_insert:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_feature_vector') as mock_publish:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    await feature_extraction_service.process_windowed_data(
                        sample_windowed_data
                    )

                    # Vérifier que les méthodes ont été appelées
                    assert mock_insert.called
                    assert mock_publish.called

    def test_get_buffer_size(self, feature_extraction_service):
        """Test récupération de la taille du buffer"""
        size = feature_extraction_service.get_buffer_size("ASSET001")
        assert isinstance(size, int)
        assert size >= 0

    def test_get_statistics(self, feature_extraction_service):
        """Test récupération des statistiques"""
        statistics = feature_extraction_service.get_statistics()
        assert isinstance(statistics, dict)
        assert "buffers" in statistics
        assert "last_processed" in statistics
        assert "services" in statistics

    @pytest.mark.asyncio
    async def test_process_preprocessed_data_empty(self, feature_extraction_service):
        """Test avec données vides"""
        await feature_extraction_service.process_preprocessed_data([], mode="streaming")
        # Ne devrait pas lever d'exception

    @pytest.mark.asyncio
    async def test_process_windowed_data_empty_sensors(self, feature_extraction_service):
        """Test avec fenêtre vide"""
        empty_window = WindowedDataReference(
            window_id="window_empty",
            asset_id="ASSET001",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            sensor_data={},
            metadata={}
        )

        # Ne devrait pas lever d'exception
        await feature_extraction_service.process_windowed_data(empty_window)

    def test_create_feature_vector(self, feature_extraction_service, sample_windowed_data):
        """Test création d'un vecteur de features"""
        features = [
            ExtractedFeature(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                feature_name="rms",
                feature_value=25.5,
                feature_type="temporal",
                metadata={}
            )
        ]

        vector = feature_extraction_service._create_feature_vector(
            sample_windowed_data,
            features,
            asset_type="pump"
        )

        assert vector.feature_vector_id == f"fv_{sample_windowed_data.window_id}"
        assert vector.asset_id == "ASSET001"
        assert "rms" in vector.features
        assert vector.features["rms"] == 25.5

