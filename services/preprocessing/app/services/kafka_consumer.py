"""
Service de consommation Kafka
"""
import json
import logging
from typing import Optional, Callable
from confluent_kafka import Consumer, KafkaError, KafkaException

# Schema registry imports are optional - we use plain JSON deserialization
try:
    from confluent_kafka.serialization import SerializationContext, MessageField
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.json_schema import JSONDeserializer
    SCHEMA_REGISTRY_AVAILABLE = True
except ImportError:
    SCHEMA_REGISTRY_AVAILABLE = False
    # Dummy classes to avoid import errors
    class SerializationContext:
        pass
    class MessageField:
        pass
    class SchemaRegistryClient:
        pass
    class JSONDeserializer:
        pass

from app.config import settings
from app.models.sensor_data import SensorData

logger = logging.getLogger(__name__)


class KafkaConsumerService:
    """Service pour consommer des messages depuis Kafka"""
    
    def __init__(self):
        self.consumer: Optional[Consumer] = None
        self.running = False
        
    def create_consumer(self) -> Consumer:
        """Crée et configure le consumer Kafka"""
        config = {
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'group.id': settings.kafka_consumer_group,
            'auto.offset.reset': settings.kafka_auto_offset_reset,
            'enable.auto.commit': settings.kafka_enable_auto_commit,
            'session.timeout.ms': 30000,
            'max.poll.interval.ms': 300000,
        }
        
        consumer = Consumer(config)
        consumer.subscribe([settings.kafka_topic_input])
        
        logger.info(f"Kafka consumer créé pour topic: {settings.kafka_topic_input}")
        return consumer
    
    def start(self, message_handler: Callable[[SensorData], None]):
        """
        Démarre la consommation de messages
        
        Args:
            message_handler: Fonction appelée pour chaque message reçu
        """
        if self.running:
            logger.warning("Consumer déjà en cours d'exécution")
            return
        
        self.consumer = self.create_consumer()
        self.running = True
        
        logger.info("Démarrage de la consommation Kafka...")
        logger.info(f"En attente de messages depuis le topic: {settings.kafka_topic_input}")
        
        message_count = 0
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    # Log every 10 seconds that we're waiting
                    if message_count == 0:
                        logger.debug("En attente de messages...")
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Fin de partition atteinte: {msg.topic()}[{msg.partition()}]")
                    else:
                        logger.error(f"Erreur Kafka: {msg.error()}")
                    continue
                
                try:
                    # Désérialiser le message JSON
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    # Normalize field names: convert camelCase to snake_case for backward compatibility
                    if 'assetId' in data and 'asset_id' not in data:
                        data['asset_id'] = data.pop('assetId')
                    if 'sensorId' in data and 'sensor_id' not in data:
                        data['sensor_id'] = data.pop('sensorId')
                    if 'sourceType' in data and 'source_type' not in data:
                        data['source_type'] = data.pop('sourceType')
                    if 'sourceEndpoint' in data and 'source_endpoint' not in data:
                        data['source_endpoint'] = data.pop('sourceEndpoint')
                    
                    # Normalize timestamp if it's a string (fix malformed formats)
                    if 'timestamp' in data and isinstance(data['timestamp'], str):
                        ts = data['timestamp']
                        # Fix: remove 'Z' if it appears after a timezone offset
                        if ts.endswith('+00:00Z') or ts.endswith('-00:00Z'):
                            data['timestamp'] = ts[:-1]  # Remove trailing Z
                        # Replace +00:00 with Z for UTC
                        elif ts.endswith('+00:00'):
                            data['timestamp'] = ts[:-6] + 'Z'
                        elif ts.endswith('-00:00'):
                            data['timestamp'] = ts[:-6] + 'Z'
                    
                    sensor_data = SensorData(**data)
                    
                    # Appeler le handler
                    message_handler(sensor_data)
                    
                    message_count += 1
                    logger.info(f"✓ Message #{message_count} traité: asset={sensor_data.asset_id}, sensor={sensor_data.sensor_id}, value={sensor_data.value}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de désérialisation JSON: {e}")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
                    # Log the raw data for debugging
                    try:
                        raw_data = json.loads(msg.value().decode('utf-8'))
                        logger.debug(f"Données du message: {list(raw_data.keys())}")
                    except:
                        pass
                    
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur dans la boucle de consommation: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Arrête le consumer"""
        if not self.running:
            return
        
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer Kafka arrêté")
    
    def consume_single_message(self, timeout: float = 1.0) -> Optional[SensorData]:
        """
        Consomme un seul message (utile pour les tests)
        
        Args:
            timeout: Timeout en secondes
            
        Returns:
            SensorData ou None si timeout
        """
        if not self.consumer:
            self.consumer = self.create_consumer()
        
        msg = self.consumer.poll(timeout=timeout)
        
        if msg is None:
            return None
        
        if msg.error():
            logger.error(f"Erreur Kafka: {msg.error()}")
            return None
        
        try:
            data = json.loads(msg.value().decode('utf-8'))
            
            # Normalize timestamp if it's a string (fix malformed formats)
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                ts = data['timestamp']
                # Fix: remove 'Z' if it appears after a timezone offset
                if ts.endswith('+00:00Z') or ts.endswith('-00:00Z'):
                    data['timestamp'] = ts[:-1]  # Remove trailing Z
                # Replace +00:00 with Z for UTC
                elif ts.endswith('+00:00'):
                    data['timestamp'] = ts[:-6] + 'Z'
                elif ts.endswith('-00:00'):
                    data['timestamp'] = ts[:-6] + 'Z'
            
            return SensorData(**data)
        except Exception as e:
            logger.error(f"Erreur de désérialisation: {e}")
            return None

