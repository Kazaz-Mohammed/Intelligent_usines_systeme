"""
Service de production Kafka
"""
import json
import logging
from typing import Optional
from confluent_kafka import Producer, KafkaError
from confluent_kafka.serialization import SerializationContext, MessageField

from app.config import settings
from app.models.sensor_data import PreprocessedData, WindowedData

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """Service pour publier des messages sur Kafka"""
    
    def __init__(self):
        self.producer: Optional[Producer] = None
        
    def create_producer(self) -> Producer:
        """Crée et configure le producer Kafka"""
        config = {
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'acks': 'all',  # Attendre confirmation de tous les replicas
            'retries': 3,
            'max.in.flight.requests.per.connection': 1,
            'enable.idempotence': True,
        }
        
        producer = Producer(config)
        logger.info(f"Kafka producer créé pour topic: {settings.kafka_topic_output}")
        return producer
    
    def get_producer(self) -> Producer:
        """Récupère ou crée le producer"""
        if not self.producer:
            self.producer = self.create_producer()
        return self.producer
    
    def _delivery_callback(self, err, msg):
        """Callback pour la confirmation de livraison"""
        if err:
            logger.error(f"Erreur de livraison Kafka: {err}")
        else:
            logger.debug(f"Message livré: topic={msg.topic()}, partition={msg.partition()}, offset={msg.offset()}")
    
    def publish_preprocessed_data(self, data: PreprocessedData):
        """
        Publie des données prétraitées sur Kafka
        
        Args:
            data: Données prétraitées à publier
        """
        try:
            producer = self.get_producer()
            
            # Sérialiser en JSON
            message_value = json.dumps(data.model_dump(), default=str).encode('utf-8')
            
            # Clé de partitionnement par asset_id
            key = data.asset_id.encode('utf-8')
            
            # Publier le message
            producer.produce(
                topic=settings.kafka_topic_output,
                key=key,
                value=message_value,
                callback=self._delivery_callback
            )
            
            # Flush pour s'assurer que le message est envoyé
            producer.poll(0)
            
            logger.debug(f"Données prétraitées publiées: asset={data.asset_id}, sensor={data.sensor_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication: {e}", exc_info=True)
            raise
    
    def publish_windowed_data(self, data: WindowedData):
        """
        Publie des données fenêtrées sur Kafka
        
        Args:
            data: Données fenêtrées à publier
        """
        try:
            producer = self.get_producer()
            
            # Sérialiser en JSON
            message_value = json.dumps(data.model_dump(), default=str).encode('utf-8')
            
            # Clé de partitionnement par asset_id
            key = data.asset_id.encode('utf-8')
            
            # Publier le message
            producer.produce(
                topic=settings.kafka_topic_output,
                key=key,
                value=message_value,
                callback=self._delivery_callback
            )
            
            # Flush pour s'assurer que le message est envoyé
            producer.poll(0)
            
            logger.debug(f"Données fenêtrées publiées: window_id={data.window_id}, asset={data.asset_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication: {e}", exc_info=True)
            raise
    
    def flush(self, timeout: float = 10.0):
        """
        Force l'envoi de tous les messages en attente
        
        Args:
            timeout: Timeout en secondes
        """
        if self.producer:
            self.producer.flush(timeout=timeout)
            logger.debug("Producer flush effectué")
    
    def close(self):
        """Ferme le producer"""
        if self.producer:
            self.flush()
            self.producer = None
            logger.info("Producer Kafka fermé")

