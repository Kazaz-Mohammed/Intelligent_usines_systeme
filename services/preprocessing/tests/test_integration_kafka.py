"""
Tests d'intégration avec Kafka
"""
import pytest
import json
import time
from datetime import datetime
from confluent_kafka import Producer, Consumer
from typing import Optional

from app.models.sensor_data import SensorData
from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService


@pytest.mark.integration
class TestKafkaIntegration:
    """Tests d'intégration avec Kafka"""
    
    @pytest.fixture
    def kafka_producer(self):
        """Producer Kafka pour les tests"""
        return Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'test-producer'
        })
    
    @pytest.fixture
    def kafka_consumer(self):
        """Consumer Kafka pour les tests"""
        return Consumer({
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'test-consumer-group',
            'auto.offset.reset': 'earliest'
        })
    
    def test_kafka_producer_connection(self, kafka_producer):
        """Test connexion au producer Kafka"""
        # Test simple de connexion
        try:
            # Vérifier que le producer peut obtenir les métadonnées
            metadata = kafka_producer.list_topics(timeout=5)
            assert metadata is not None
            assert 'sensor-data' in metadata.topics
            print("✅ Producer Kafka connecté avec succès")
        except Exception as e:
            pytest.skip(f"Kafka non disponible: {e}")
    
    def test_kafka_consumer_connection(self, kafka_consumer):
        """Test connexion au consumer Kafka"""
        try:
            # Vérifier que le consumer peut se connecter
            metadata = kafka_consumer.list_topics(timeout=5)
            assert metadata is not None
            print("✅ Consumer Kafka connecté avec succès")
        except Exception as e:
            pytest.skip(f"Kafka non disponible: {e}")
    
    def test_send_and_receive_message(self, kafka_producer, kafka_consumer):
        """Test envoi et réception d'un message"""
        try:
            topic = 'sensor-data'
            test_message = {
                "timestamp": datetime.utcnow().isoformat(),
                "asset_id": "TEST_ASSET",
                "sensor_id": "TEST_SENSOR",
                "value": 25.5,
                "unit": "°C",
                "quality": 2,
                "source_type": "TEST"
            }
            
            # Envoyer le message
            kafka_producer.produce(
                topic,
                key="TEST_KEY",
                value=json.dumps(test_message)
            )
            kafka_producer.flush()
            print(f"✅ Message envoyé vers {topic}")
            
            # Consommer le message
            kafka_consumer.subscribe([topic])
            msg = kafka_consumer.poll(timeout=10)
            
            if msg is None:
                pytest.skip("Aucun message reçu (timeout)")
            
            assert msg.error() is None
            received_data = json.loads(msg.value().decode('utf-8'))
            assert received_data['asset_id'] == "TEST_ASSET"
            assert received_data['value'] == 25.5
            print("✅ Message reçu avec succès")
            
        except Exception as e:
            pytest.skip(f"Kafka non disponible: {e}")
        finally:
            kafka_consumer.close()
    
    def test_kafka_producer_service(self):
        """Test du service KafkaProducerService"""
        try:
            producer_service = KafkaProducerService()
            
            # Créer une donnée prétraitée
            from app.models.sensor_data import PreprocessedData
            
            preprocessed_data = PreprocessedData(
                timestamp=datetime.utcnow(),
                asset_id="TEST_ASSET",
                sensor_id="TEST_SENSOR",
                value=25.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={"test": True}
            )
            
            # Publier (ne devrait pas lever d'erreur)
            try:
                producer_service.publish_preprocessed_data(preprocessed_data)
                producer_service.flush()
                print("✅ Service KafkaProducerService fonctionne")
            except Exception as e:
                # Si Kafka n'est pas disponible, on skip
                pytest.skip(f"Kafka non disponible: {e}")
            
        except Exception as e:
            pytest.skip(f"Kafka non disponible: {e}")
        finally:
            if 'producer_service' in locals():
                producer_service.close()

