"""
Tests d'intégration end-to-end
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.services.feature_extraction_service import FeatureExtractionService
from app.models.feature_data import PreprocessedDataReference, WindowedDataReference


@pytest.mark.integration
class TestEndToEnd:
    """Tests d'intégration end-to-end"""

    @pytest.fixture
    def feature_extraction_service(self):
        return FeatureExtractionService()

    @pytest.mark.asyncio
    async def test_end_to_end_preprocessed_to_features(self, feature_extraction_service):
        """Test end-to-end: données prétraitées -> features"""
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
            ) for i in range(100)
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

                    # Vérifier que les features ont été calculées et stockées
                    assert mock_db.called
                    assert mock_kafka.called

                    # Vérifier que des features ont été extraites
                    call_args = mock_db.call_args
                    features = call_args[0][0] if call_args[0] else []
                    assert len(features) > 0

                    # Vérifier que les features ont les bons types
                    feature_types = set(f.feature_type for f in features)
                    assert "temporal" in feature_types

    @pytest.mark.asyncio
    async def test_end_to_end_windowed_to_feature_vector(self, feature_extraction_service):
        """Test end-to-end: fenêtre -> vecteur de features"""
        # Créer une fenêtre de données
        base_time = datetime.utcnow()
        windowed_data = WindowedDataReference(
            window_id="window_001",
            asset_id="ASSET001",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=100),
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
                    ) for i in range(100)
                ]
            },
            metadata={"window_size": 100}
        )

        # Mock des services externes
        with patch.object(feature_extraction_service.timescale_db_service, 'insert_feature_vector') as mock_db:
            with patch.object(feature_extraction_service.kafka_producer, 'publish_feature_vector') as mock_kafka:
                with patch.object(feature_extraction_service.asset_service, 'get_asset_type', return_value="pump"):
                    # Traiter la fenêtre
                    await feature_extraction_service.process_windowed_data(windowed_data)

                    # Vérifier que le vecteur de features a été créé et stocké
                    assert mock_db.called
                    assert mock_kafka.called

                    # Vérifier que le vecteur de features a les bonnes propriétés
                    call_args = mock_db.call_args
                    vector = call_args[0][0] if call_args[0] else None
                    assert vector is not None
                    assert vector.asset_id == "ASSET001"
                    assert len(vector.features) > 0

