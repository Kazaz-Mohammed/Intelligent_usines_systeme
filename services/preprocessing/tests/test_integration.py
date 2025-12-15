"""
Tests d'intégration pour le service Prétraitement
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services.preprocessing_service import PreprocessingService
from app.models.sensor_data import SensorData


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Tests d'intégration"""
    
    @pytest.fixture
    def preprocessing_service(self):
        return PreprocessingService()
    
    def test_full_pipeline_streaming(self, preprocessing_service):
        """Test pipeline complet en mode streaming"""
        # Mock Kafka producer pour éviter besoin de Kafka réel
        with patch.object(preprocessing_service.kafka_producer, 'publish_preprocessed_data') as mock_publish:
            sensor_data = SensorData(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.5,
                unit="°C",
                quality=2,
                source_type="TEST"
            )
            
            preprocessing_service.process_and_publish(sensor_data)
            
            # Vérifier que publish a été appelé
            assert mock_publish.called
    
    def test_full_pipeline_batch(self, preprocessing_service):
        """Test pipeline complet en mode batch"""
        # Mock Kafka producer
        with patch.object(preprocessing_service.kafka_producer, 'publish_windowed_data') as mock_publish:
            # Créer assez de données
            for i in range(150):
                sensor_data = SensorData(
                    timestamp=datetime.utcnow() + timedelta(seconds=i),
                    asset_id="ASSET001",
                    sensor_id="SENSOR001",
                    value=25.0 + i * 0.1,
                    unit="°C",
                    quality=2,
                    source_type="TEST"
                )
                
                windows = preprocessing_service.accumulate_and_process_batch(
                    sensor_data,
                    min_batch_size=100
                )
                
                if windows:
                    # Vérifier que publish a été appelé
                    assert mock_publish.called
                    break

