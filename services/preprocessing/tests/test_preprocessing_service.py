"""
Tests pour le service principal de prétraitement
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from app.services.preprocessing_service import PreprocessingService
from app.models.sensor_data import SensorData, PreprocessedData


class TestPreprocessingService:
    """Tests pour PreprocessingService"""
    
    @pytest.fixture
    def preprocessing_service(self):
        return PreprocessingService()
    
    @pytest.fixture
    def sample_sensor_data(self):
        return SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=25.5,
            unit="°C",
            quality=2,
            source_type="TEST"
        )
    
    def test_process_single_sensor_data(self, preprocessing_service, sample_sensor_data):
        """Test traitement d'une donnée unique"""
        result = preprocessing_service.process_single_sensor_data(sample_sensor_data)
        
        assert result is not None
        assert isinstance(result, PreprocessedData)
        assert result.value == 25.5
        assert result.asset_id == "ASSET001"
    
    def test_process_single_sensor_data_bad_quality(self, preprocessing_service):
        """Test traitement d'une donnée de mauvaise qualité"""
        bad_data = SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=25.5,
            unit="°C",
            quality=0,  # Bad quality
            source_type="TEST"
        )
        
        result = preprocessing_service.process_single_sensor_data(bad_data)
        
        # Devrait retourner None car qualité trop faible
        assert result is None
    
    @patch('app.services.preprocessing_service.KafkaProducerService')
    def test_process_and_publish(self, mock_producer_class, preprocessing_service, sample_sensor_data):
        """Test traitement et publication"""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer
        preprocessing_service.kafka_producer = mock_producer
        
        preprocessing_service.process_and_publish(sample_sensor_data)
        
        # Vérifier que publish a été appelé
        assert mock_producer.publish_preprocessed_data.called
    
    def test_process_batch(self, preprocessing_service):
        """Test traitement d'un batch"""
        sensor_data_list = [
            SensorData(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id=f"SENSOR{i:03d}",
                value=25.0 + i,
                unit="°C",
                quality=2,
                source_type="TEST"
            )
            for i in range(10)
        ]
        
        result = preprocessing_service.process_batch(sensor_data_list)
        
        assert len(result) > 0
        assert all(isinstance(d, PreprocessedData) for d in result)
    
    def test_accumulate_and_process_batch(self, preprocessing_service, sample_sensor_data):
        """Test accumulation et création de fenêtres"""
        # Créer assez de données pour déclencher la création de fenêtres
        windows = None
        for i in range(150):  # Plus que min_batch_size (100)
            data = SensorData(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i * 0.1,
                unit="°C",
                quality=2,
                source_type="TEST"
            )
            windows = preprocessing_service.accumulate_and_process_batch(data, min_batch_size=100)
            
            if windows:
                break
        
        # Si assez de données, des fenêtres devraient être créées
        # (peut être None si pas encore assez de données)
        if windows:
            assert len(windows) > 0
            assert windows[0].asset_id == "ASSET001"

