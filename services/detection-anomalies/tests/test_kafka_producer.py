"""
Tests pour le service Kafka Producer
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from app.services.kafka_producer import KafkaProducerService
from app.config import settings


@pytest.fixture
def kafka_producer():
    """Fixture pour créer un producer Kafka"""
    with patch('app.services.kafka_producer.Producer'):
        producer = KafkaProducerService()
        yield producer


def test_kafka_producer_init(kafka_producer):
    """Test de l'initialisation du producer"""
    assert kafka_producer.producer is not None


def test_publish_anomaly_dict(kafka_producer):
    """Test de la publication d'une anomalie (dict)"""
    anomaly_data = {
        "asset_id": "ASSET001",
        "is_anomaly": True,
        "score": 0.8
    }
    
    kafka_producer.producer.produce = Mock()
    kafka_producer.producer.poll = Mock()
    
    kafka_producer.publish_anomaly(anomaly_data)
    
    # Vérifier que produce a été appelé
    kafka_producer.producer.produce.assert_called_once()
    call_args = kafka_producer.producer.produce.call_args
    assert call_args[0][0] == settings.kafka_topic_output_anomalies


def test_publish_anomaly_pydantic(kafka_producer):
    """Test de la publication d'une anomalie (Pydantic model)"""
    from app.models.anomaly_data import AnomalyDetectionResult, AnomalyScore, CriticalityLevel
    from datetime import datetime, timezone
    
    anomaly_result = AnomalyDetectionResult(
        asset_id="ASSET001",
        sensor_id="SENSOR001",
        timestamp=datetime.now(timezone.utc),
        scores=[AnomalyScore(score=0.8, model_name="test", threshold=0.5, is_anomaly=True)],
        final_score=0.8,
        is_anomaly=True,
        criticality=CriticalityLevel.HIGH,
        features={"rms": 0.5}
    )
    
    kafka_producer.producer.produce = Mock()
    kafka_producer.producer.poll = Mock()
    
    kafka_producer.publish_anomaly(anomaly_result)
    
    # Vérifier que produce a été appelé
    kafka_producer.producer.produce.assert_called_once()


def test_publish_anomalies_batch(kafka_producer):
    """Test de la publication d'un batch d'anomalies"""
    anomalies = [
        {"asset_id": f"ASSET{i:03d}", "score": 0.5 + i * 0.1}
        for i in range(3)
    ]
    
    kafka_producer.producer.produce = Mock()
    kafka_producer.producer.flush = Mock()
    
    kafka_producer.publish_anomalies_batch(anomalies)
    
    # Vérifier que produce a été appelé 3 fois
    assert kafka_producer.producer.produce.call_count == 3
    kafka_producer.producer.flush.assert_called_once()


def test_close(kafka_producer):
    """Test de la fermeture du producer"""
    # S'assurer que le producer existe et est un mock
    if kafka_producer.producer is None:
        kafka_producer._create_producer()
    
    # Capturer la référence au mock avant close()
    producer_mock = kafka_producer.producer
    producer_mock.flush = Mock()
    
    # Appeler close() qui va mettre producer à None
    kafka_producer.close()
    
    # Vérifier que flush() a été appelé sur le mock (avant qu'il ne soit mis à None)
    producer_mock.flush.assert_called_once()
    
    # Vérifier que producer est maintenant None
    assert kafka_producer.producer is None

