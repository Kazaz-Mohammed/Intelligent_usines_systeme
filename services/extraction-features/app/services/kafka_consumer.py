"""
Service de consommation Kafka pour Extraction Features
"""
import logging
import json
from typing import List, Optional, Callable
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.serialization import SerializationContext, MessageField

from app.config import settings
from app.models.feature_data import PreprocessedDataReference, WindowedDataReference

logger = logging.getLogger(__name__)

# Schema Registry imports are optional (not currently used, but may be needed in future)
try:
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.json_schema import JSONDeserializer
    SCHEMA_REGISTRY_AVAILABLE = True
except ImportError:
    SCHEMA_REGISTRY_AVAILABLE = False
    logger.warning("Schema Registry not available - install dependencies: cachetools, fastavro, jsonschema, referencing, authlib, cryptography")


class KafkaConsumerService:
    """Service pour consommer des données depuis Kafka"""
    
    def __init__(self):
        self.config = {
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'group.id': settings.kafka_consumer_group,
            'auto.offset.reset': settings.kafka_auto_offset_reset,
            'enable.auto.commit': settings.kafka_enable_auto_commit,
            'session.timeout.ms': 30000,
            'max.poll.interval.ms': 300000,
        }
        self.consumer: Optional[Consumer] = None
        self._subscribed_to_preprocessed = False  # Track subscription state
        self._subscribed_to_windowed = False
        self._partitions_assigned = False  # Track partition assignment
        self._create_consumer()
    
    def _create_consumer(self):
        """Crée le consumer Kafka"""
        try:
            self.consumer = Consumer(self.config)
            logger.info(f"Consumer Kafka créé: {settings.kafka_bootstrap_servers}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du consumer Kafka: {e}", exc_info=True)
            raise
    
    def consume_preprocessed_data(
        self,
        callback: Callable[[List[PreprocessedDataReference]], None],
        timeout: float = 1.0,
        max_messages: int = 100
    ):
        """
        Consomme des données prétraitées depuis Kafka
        
        Args:
            callback: Fonction à appeler avec les données consommées
            timeout: Timeout en secondes pour la consommation
            max_messages: Nombre maximum de messages à consommer par batch
        """
        if not self.consumer:
            self._create_consumer()
        
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        # Subscribe only once
        if not self._subscribed_to_preprocessed:
            try:
                # Set up partition assignment callback
                def on_assign(consumer, partitions):
                    logger.info(f"✓ Partitions assignées: {[f'{p.topic}[{p.partition}]' for p in partitions]}")
                    self._partitions_assigned = True
                
                def on_revoke(consumer, partitions):
                    logger.warning(f"Partitions révoquées: {[f'{p.topic}[{p.partition}]' for p in partitions]}")
                    self._partitions_assigned = False
                
                self.consumer.subscribe(
                    [settings.kafka_topic_input_preprocessed],
                    on_assign=on_assign,
                    on_revoke=on_revoke
                )
                self._subscribed_to_preprocessed = True
                logger.info(f"Abonné au topic: {settings.kafka_topic_input_preprocessed}")
                # Wait a bit for partition assignment
                import time
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Erreur lors de l'abonnement: {e}", exc_info=True)
                return
        
        messages = []
        msg_count = 0
        
        # Poll for messages in a loop until we have max_messages or timeout
        start_time = datetime.now()
        poll_count = 0
        while len(messages) < max_messages:
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= timeout:
                if poll_count == 0:
                    logger.debug(f"Aucun message reçu après {timeout}s (poll_count={poll_count})")
                break
            
            remaining_timeout = max(0.1, timeout - elapsed)
            msg = self.consumer.poll(timeout=remaining_timeout)
            poll_count += 1
            
            if msg is None:
                # No message available - this is normal, just continue
                # Only log occasionally to avoid spam
                if poll_count % 10 == 0:
                    logger.debug(f"Poll {poll_count}: aucun message disponible (attente de messages...)")
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"Fin de partition atteinte: {msg.topic()}[{msg.partition()}]")
                    continue
                elif msg.error().code() == KafkaError._RESOLVE:
                    logger.warning(f"Erreur de résolution: {msg.error()}")
                    continue
                else:
                    logger.error(f"Erreur Kafka: {msg.error()}")
                    continue
            
            # Valid message received
            logger.info(f"✓ Message reçu: topic={msg.topic()}, partition={msg.partition()}, offset={msg.offset()}")
            messages.append(msg)
            msg_count += 1
        
        # Process accumulated messages
        if messages:
            logger.info(f"📦 Traitement de {len(messages)} messages prétraités")
            try:
                preprocessed_data = self._deserialize_preprocessed_data(messages)
                if preprocessed_data:
                    logger.info(f"✓ {len(preprocessed_data)} messages désérialisés avec succès, appel du callback...")
                    callback(preprocessed_data)
                    logger.info(f"✓ Callback exécuté avec succès")
                else:
                    logger.warning(f"⚠ Aucune donnée valide après désérialisation de {len(messages)} messages")
            except Exception as e:
                logger.error(f"✗ Erreur lors du traitement des messages: {e}", exc_info=True)
    
    def consume_windowed_data(
        self,
        callback: Callable[[WindowedDataReference], None],
        timeout: float = 1.0
    ):
        """
        Consomme des fenêtres de données depuis Kafka
        
        Args:
            callback: Fonction à appeler avec les fenêtres consommées
            timeout: Timeout en secondes pour la consommation
        """
        if not self.consumer:
            self._create_consumer()
        
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        # Subscribe only once
        if not self._subscribed_to_windowed:
            try:
                self.consumer.subscribe([settings.kafka_topic_input_windowed])
                self._subscribed_to_windowed = True
                logger.info(f"Abonné au topic: {settings.kafka_topic_input_windowed}")
            except Exception as e:
                logger.error(f"Erreur lors de l'abonnement: {e}", exc_info=True)
                return
        
        try:
            
            while True:
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Fin de partition atteinte: {msg.topic()}[{msg.partition()}]")
                        continue
                    else:
                        logger.error(f"Erreur Kafka: {msg.error()}")
                        continue  # Continuer au lieu de lever une exception
                
                # Désérialiser le message
                windowed_data = self._deserialize_windowed_data(msg)
                if windowed_data:
                    callback(windowed_data)
                    
        except KeyboardInterrupt:
            logger.info("Interruption reçue, arrêt de la consommation")
        except Exception as e:
            logger.error(f"Erreur lors de la consommation: {e}", exc_info=True)
            raise
    
    def _deserialize_preprocessed_data(
        self,
        messages: List
    ) -> List[PreprocessedDataReference]:
        """Désérialise les messages en PreprocessedDataReference"""
        preprocessed_data = []
        
        for msg in messages:
            try:
                # Désérialiser JSON
                data = json.loads(msg.value().decode('utf-8'))
                
                # Créer PreprocessedDataReference
                preprocessed_data.append(PreprocessedDataReference(**data))
                
            except Exception as e:
                logger.error(f"Erreur lors de la désérialisation: {e}", exc_info=True)
                continue
        
        return preprocessed_data
    
    def _deserialize_windowed_data(self, msg) -> Optional[WindowedDataReference]:
        """Désérialise un message en WindowedDataReference"""
        try:
            # Désérialiser JSON
            msg_value = msg.value()
            if isinstance(msg_value, bytes):
                msg_value = msg_value.decode('utf-8')
            data = json.loads(msg_value)
            
            # Convertir timestamps si ce sont des chaînes
            if 'start_time' in data and isinstance(data['start_time'], str):
                data['start_time'] = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
            if 'end_time' in data and isinstance(data['end_time'], str):
                data['end_time'] = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
            
            # Convertir sensor_data en PreprocessedDataReference
            if 'sensor_data' in data:
                sensor_data = {}
                for sensor_id, sensor_list in data['sensor_data'].items():
                    processed_list = []
                    for item in sensor_list:
                        # Convertir timestamp si c'est une chaîne
                        if 'timestamp' in item and isinstance(item['timestamp'], str):
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                        processed_list.append(PreprocessedDataReference(**item))
                    sensor_data[sensor_id] = processed_list
                data['sensor_data'] = sensor_data
            
            # Créer WindowedDataReference
            return WindowedDataReference(**data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la désérialisation: {e}", exc_info=True)
            logger.debug(f"Message value: {msg.value() if hasattr(msg, 'value') else 'N/A'}")
            return None
    
    def close(self):
        """Ferme le consumer Kafka"""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer Kafka fermé")

