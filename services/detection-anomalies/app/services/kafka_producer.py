"""
Service de production Kafka pour Detection Anomalies
"""
import logging
import json
from typing import Optional, Dict, Any
from confluent_kafka import Producer

from app.config import settings

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """Service pour publier des anomalies détectées sur Kafka"""
    
    def __init__(self):
        self.config = {
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'acks': 'all',  # Attendre confirmation de tous les replicas
            'retries': 3,
            'max.in.flight.requests.per.connection': 1,  # Garantir l'ordre
            'enable.idempotence': True,  # Idempotence
        }
        self.producer: Optional[Producer] = None
        self._create_producer()
    
    def _create_producer(self):
        """Crée le producer Kafka"""
        try:
            self.producer = Producer(self.config)
            logger.info(f"Producer Kafka créé: {settings.kafka_bootstrap_servers}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du producer Kafka: {e}", exc_info=True)
            raise
    
    def _delivery_callback(self, err, msg):
        """Callback pour la confirmation de livraison"""
        if err:
            logger.error(f"Erreur lors de la livraison du message: {err}")
        else:
            logger.debug(f"Message livré: {msg.topic()}[{msg.partition()}]@{msg.offset()}")
    
    def publish_anomaly(self, anomaly_data: Dict[str, Any]):
        """
        Publie une anomalie détectée sur Kafka
        
        Args:
            anomaly_data: Données de l'anomalie à publier (dict ou Pydantic model)
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            # Sérialiser en JSON
            if hasattr(anomaly_data, 'model_dump_json'):
                # Pydantic model
                data = anomaly_data.model_dump_json()
            elif hasattr(anomaly_data, 'model_dump'):
                # Pydantic model (alternative)
                data = json.dumps(anomaly_data.model_dump())
            else:
                # Dict
                data = json.dumps(anomaly_data)
            
            # Publier sur le topic
            self.producer.produce(
                settings.kafka_topic_output_anomalies,
                value=data.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Flush pour garantir la livraison
            self.producer.poll(0)
            
            # Logger avec asset_id
            asset_id = None
            if hasattr(anomaly_data, 'asset_id'):
                asset_id = anomaly_data.asset_id
            elif isinstance(anomaly_data, dict):
                asset_id = anomaly_data.get('asset_id', 'unknown')
            else:
                asset_id = 'unknown'
            
            logger.debug(f"Anomalie publiée: {asset_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication: {e}", exc_info=True)
            raise
    
    def publish_anomalies_batch(self, anomalies: list):
        """
        Publie un lot d'anomalies détectées sur Kafka
        
        Args:
            anomalies: Liste d'anomalies à publier
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            for anomaly in anomalies:
                # Sérialiser en JSON
                if hasattr(anomaly, 'model_dump_json'):
                    data = anomaly.model_dump_json()
                elif hasattr(anomaly, 'model_dump'):
                    data = json.dumps(anomaly.model_dump())
                else:
                    data = json.dumps(anomaly)
                
                # Publier sur le topic
                self.producer.produce(
                    settings.kafka_topic_output_anomalies,
                    value=data.encode('utf-8'),
                    callback=self._delivery_callback
                )
            
            # Flush pour garantir la livraison
            self.producer.flush()
            
            logger.info(f"Lot de {len(anomalies)} anomalies publié")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication du lot: {e}", exc_info=True)
            raise
    
    def close(self):
        """Ferme le producer Kafka"""
        if self.producer:
            self.producer.flush()
            self.producer = None
            logger.info("Producer Kafka fermé")

