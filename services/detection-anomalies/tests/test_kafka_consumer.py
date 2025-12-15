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
    """Fixture pour créer un consumer Kafka"""
    with patch('app.services.kafka_consumer.Consumer'):
        consumer = KafkaConsumerService()
        yield consumer


def test_kafka_consumer_init(kafka_consumer):
    """Test de l'initialisation du consumer"""
    assert kafka_consumer.consumer is not None


def test_consume_features_callback(kafka_consumer):
    """Test que le callback est appelé avec les bonnes données"""
    callback_called = []
    
    def callback(data):
        callback_called.append(data)
    
    # Mock du message Kafka
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = json.dumps({
        "asset_id": "ASSET001",
        "features": {"rms": 0.5}
    }).encode('utf-8')
    mock_msg.topic.return_value = "test-topic"
    mock_msg.partition.return_value = 0
    
    # Mock du consumer - retourner le message puis None immédiatement
    call_count = [0]
    def poll_side_effect(timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_msg
        return None  # Terminer immédiatement
    
    kafka_consumer.consumer.poll = Mock(side_effect=poll_side_effect)
    kafka_consumer.consumer.subscribe = Mock()
    
    # Consommer avec timeout très court
    kafka_consumer.consume_features(callback, timeout=0.001, max_messages=1)
    
    # Vérifier que le callback a été appelé
    assert len(callback_called) == 1
    assert callback_called[0]["asset_id"] == "ASSET001"


def test_consume_features_invalid_json(kafka_consumer):
    """Test que les messages JSON invalides sont ignorés"""
    callback_called = []
    
    def callback(data):
        callback_called.append(data)
    
    # Mock du message Kafka avec JSON invalide
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b"invalid json"
    mock_msg.topic.return_value = "test-topic"
    mock_msg.partition.return_value = 0
    
    # Mock du consumer - retourner le message invalide puis None pour terminer rapidement
    call_count = [0]  # Utiliser une liste pour la mutabilité dans la closure
    def poll_side_effect(timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_msg
        return None  # Terminer immédiatement
    
    kafka_consumer.consumer.poll = Mock(side_effect=poll_side_effect)
    kafka_consumer.consumer.subscribe = Mock()
    
    # Consommer avec timeout très court
    kafka_consumer.consume_features(callback, timeout=0.001, max_messages=1)
    
    # Vérifier que le callback n'a pas été appelé
    assert len(callback_called) == 0


def test_close(kafka_consumer):
    """Test de la fermeture du consumer"""
    # S'assurer que le consumer existe et est un mock
    if kafka_consumer.consumer is None:
        kafka_consumer._create_consumer()
    
    # Capturer la référence au mock avant close()
    consumer_mock = kafka_consumer.consumer
    consumer_mock.close = Mock()
    
    # Appeler close() qui va mettre consumer à None
    kafka_consumer.close()
    
    # Vérifier que close() a été appelé sur le mock (avant qu'il ne soit mis à None)
    consumer_mock.close.assert_called_once()
    
    # Vérifier que consumer est maintenant None
    assert kafka_consumer.consumer is None

