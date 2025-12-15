"""
Tests pour le service Kafka Consumer
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from app.services.kafka_consumer import KafkaConsumerService
from app.config import settings


@pytest.fixture
def kafka_consumer():
    """Fixture pour créer une instance du consumer"""
    with patch('app.services.kafka_consumer.Consumer') as MockConsumer:
        mock_consumer = MagicMock()
        MockConsumer.return_value = mock_consumer
        
        service = KafkaConsumerService()
        service.consumer = mock_consumer
        return service


@pytest.fixture
def sample_message():
    """Fixture pour créer un message Kafka factice"""
    message_data = {
        "asset_id": "ASSET001",
        "sensor_id": "SENSOR001",
        "features": {"rms": 10.5, "kurtosis": 2.3},
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    msg = MagicMock()
    msg.error.return_value = None
    msg.value.return_value = json.dumps(message_data).encode('utf-8')
    msg.topic.return_value = "extracted-features"
    msg.partition.return_value = 0
    msg.offset.return_value = 123
    
    return msg


class TestKafkaConsumerService:
    """Tests pour KafkaConsumerService"""
    
    def test_init(self, kafka_consumer):
        """Test initialisation"""
        assert kafka_consumer.consumer is not None
    
    def test_subscribe(self, kafka_consumer):
        """Test abonnement aux topics"""
        topics = ["extracted-features"]
        kafka_consumer.subscribe(topics)
        
        kafka_consumer.consumer.subscribe.assert_called_once_with(topics)
    
    def test_consume_features(self, kafka_consumer, sample_message):
        """Test consommation de messages"""
        callback_calls = []
        
        def callback(data):
            callback_calls.append(data)
        
        # Mock poll pour retourner des messages
        kafka_consumer.consumer.poll.side_effect = [
            sample_message,
            sample_message,
            None  # Fin de la consommation
        ]
        
        count = kafka_consumer.consume_features(callback, max_messages=2)
        
        assert count == 2
        assert len(callback_calls) == 2
        assert callback_calls[0]["asset_id"] == "ASSET001"
    
    def test_consume_features_invalid_json(self, kafka_consumer):
        """Test avec message JSON invalide"""
        callback_calls = []
        
        def callback(data):
            callback_calls.append(data)
        
        # Message avec JSON invalide
        invalid_msg = MagicMock()
        invalid_msg.error.return_value = None
        invalid_msg.value.return_value = b"invalid json"
        
        kafka_consumer.consumer.poll.side_effect = [
            invalid_msg,
            None
        ]
        
        count = kafka_consumer.consume_features(callback, max_messages=1)
        
        # Le message invalide est ignoré mais le compteur est incrémenté
        assert count == 1
        assert len(callback_calls) == 0
    
    def test_close(self, kafka_consumer):
        """Test fermeture du consumer"""
        mock_consumer = kafka_consumer.consumer
        kafka_consumer.close()
        
        mock_consumer.close.assert_called_once()
        assert kafka_consumer.consumer is None

