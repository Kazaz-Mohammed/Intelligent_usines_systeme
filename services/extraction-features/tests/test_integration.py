"""
Tests d'intégration pour le service Extraction Features
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.services.feature_extraction_service import FeatureExtractionService
from app.models.feature_data import PreprocessedDataReference, WindowedDataReference


@pytest.mark.integration
class TestIntegration:
    """Tests d'intégration"""

    @pytest.fixture
    def feature_extraction_service(self):
        return FeatureExtractionService()

    @pytest.mark.asyncio
    async def test_full_pipeline_preprocessed_data(self, feature_extraction_service):
        """Test du pipeline complet avec données prétraitées"""
        # Créer des données prétraitées
        base_time = datetime.utcnow()
        preprocessed_data = [
            PreprocessedDataReference(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + (i % 10) * 0.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={"cleaned": True},
                frequency_analysis=None
            ) for i in range(50)
        ]

        # Mock des services externes
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_extracted_features_batch') as mock_db:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_extracted_features_batch') as mock_kafka:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    # Traiter les données
                    await feature_extraction_service.process_preprocessed_data(
                        preprocessed_data,
                        mode="streaming"
                    )

                    # Vérifier que les services ont été appelés
                    assert mock_db.called
                    assert mock_kafka.called

    @pytest.mark.asyncio
    async def test_full_pipeline_windowed_data(self, feature_extraction_service):
        """Test du pipeline complet avec fenêtres de données"""
        # Créer une fenêtre de données
        base_time = datetime.utcnow()
        windowed_data = WindowedDataReference(
            window_id="window_001",
            asset_id="ASSET001",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=50),
            sensor_data={
                "SENSOR001": [
                    PreprocessedDataReference(
                        timestamp=base_time + timedelta(seconds=i),
                        asset_id="ASSET001",
                        sensor_id="SENSOR001",
                        value=25.0 + (i % 10) * 0.5,
                        unit="°C",
                        quality=2,
                        source_type="TEST",
                        preprocessing_metadata={"cleaned": True},
                        frequency_analysis=None
                    ) for i in range(50)
                ]
            },
            metadata={"window_size": 50}
        )

        # Mock des services externes
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_feature_vector') as mock_db:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_feature_vector') as mock_kafka:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    # Traiter la fenêtre
                    await feature_extraction_service.process_windowed_data(windowed_data)

                    # Vérifier que les services ont été appelés
                    assert mock_db.called
                    assert mock_kafka.called

    @pytest.mark.asyncio
    async def test_feature_extraction_with_standardization(self, feature_extraction_service):
        """Test extraction de features avec standardisation"""
        # Créer des données prétraitées
        base_time = datetime.utcnow()
        preprocessed_data = [
            PreprocessedDataReference(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + (i % 10) * 0.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={"cleaned": True},
                frequency_analysis=None
            ) for i in range(50)
        ]

        # Mock des services externes
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_extracted_features_batch') as mock_db:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_extracted_features_batch') as mock_kafka:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    # Traiter les données
                    await feature_extraction_service.process_preprocessed_data(
                        preprocessed_data,
                        mode="streaming"
                    )

                    # Vérifier que les services ont été appelés
                    assert mock_db.called
                    # Vérifier que les features standardisées ont été ajoutées
                    call_args = mock_db.call_args
                    features = call_args[0][0] if call_args[0] else []
                    # Il devrait y avoir des features standardisées si la standardisation est activée
                    assert len(features) > 0

