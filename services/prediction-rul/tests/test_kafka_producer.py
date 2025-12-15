"""
Tests pour le service Kafka Producer
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.kafka_producer import KafkaProducerService
from app.models.rul_data import RULPredictionResult
from datetime import datetime, timezone


@pytest.fixture
def kafka_producer():
    """Fixture pour créer une instance du producer"""
    with patch('app.services.kafka_producer.Producer') as MockProducer:
        mock_producer = MagicMock()
        MockProducer.return_value = mock_producer
        
        service = KafkaProducerService()
        service.producer = mock_producer
        return service


@pytest.fixture
def sample_rul_result():
    """Fixture pour créer un résultat RUL"""
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
        features={"rms": 10.5},
        metadata={}
    )


class TestKafkaProducerService:
    """Tests pour KafkaProducerService"""
    
    def test_init(self, kafka_producer):
        """Test initialisation"""
        assert kafka_producer.producer is not None
    
    def test_publish_rul_prediction(self, kafka_producer, sample_rul_result):
        """Test publication d'une prédiction RUL"""
        result = kafka_producer.publish_rul_prediction(sample_rul_result)
        
        assert result is True
        kafka_producer.producer.produce.assert_called_once()
        kafka_producer.producer.poll.assert_called_once()
    
    def test_publish_rul_predictions_batch(self, kafka_producer, sample_rul_result):
        """Test publication batch"""
        results = [
            sample_rul_result,
            RULPredictionResult(
                asset_id="ASSET002",
                sensor_id="SENSOR002",
                timestamp=datetime.now(timezone.utc),
                rul_prediction=200.0,
                confidence_interval_lower=190.0,
                confidence_interval_upper=210.0,
                confidence_level=0.95,
                uncertainty=10.0,
                model_used="ensemble",
                model_scores={},
                features={"rms": 12.0},
                metadata={}
            )
        ]
        
        count = kafka_producer.publish_rul_predictions_batch(results)
        
        assert count == 2
        assert kafka_producer.producer.produce.call_count == 2
        kafka_producer.producer.flush.assert_called_once()
    
    def test_close(self, kafka_producer):
        """Test fermeture du producer"""
        mock_producer = kafka_producer.producer
        kafka_producer.close()
        
        mock_producer.flush.assert_called_once()
        assert kafka_producer.producer is None

