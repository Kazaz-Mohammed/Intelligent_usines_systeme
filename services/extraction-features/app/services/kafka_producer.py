"""
Service de production Kafka pour Extraction Features
"""
import logging
import json
from typing import List, Optional, Dict, Any
from confluent_kafka import Producer
from confluent_kafka.serialization import SerializationContext, MessageField

from app.config import settings
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """Service pour publier des features extraites sur Kafka"""
    
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
    
    def publish_extracted_feature(self, feature: ExtractedFeature):
        """
        Publie une feature extraite sur Kafka
        
        Args:
            feature: Feature extraite à publier
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            # Sérialiser en JSON
            data = feature.model_dump_json()
            
            # Publier sur le topic
            self.producer.produce(
                settings.kafka_topic_output,
                value=data.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Flush pour garantir la livraison
            self.producer.poll(0)
            
            # Get asset_id from feature
            asset_id = feature.asset_id
            logger.debug(f"Feature publiée: {feature.feature_name} pour {asset_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication: {e}", exc_info=True)
            raise
    
    def publish_extracted_features_batch(self, features: List[ExtractedFeature]):
        """
        Publie un lot de features extraites sur Kafka
        
        Args:
            features: Liste de features extraites à publier
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            for feature in features:
                # Sérialiser en JSON
                data = feature.model_dump_json()
                
                # Publier sur le topic
                self.producer.produce(
                    settings.kafka_topic_output,
                    value=data.encode('utf-8'),
                    callback=self._delivery_callback
                )
            
            # Flush pour garantir la livraison
            self.producer.flush()
            
            logger.info(f"Lot de {len(features)} features publié")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication du lot: {e}", exc_info=True)
            raise
    
    def publish_feature_vector(self, feature_vector: ExtractedFeaturesVector):
        """
        Publie un vecteur de features sur Kafka
        
        Args:
            feature_vector: Vecteur de features à publier
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            # Sérialiser en JSON
            data = feature_vector.model_dump_json()
            
            # Publier sur le topic
            self.producer.produce(
                settings.kafka_topic_output,
                value=data.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Flush pour garantir la livraison
            self.producer.poll(0)
            
            logger.debug(f"Vecteur de features publié: {feature_vector.vector_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication: {e}", exc_info=True)
            raise
    
    def publish_feature_vectors_batch(self, feature_vectors: List[ExtractedFeaturesVector]):
        """
        Publie un lot de vecteurs de features sur Kafka
        
        Args:
            feature_vectors: Liste de vecteurs de features à publier
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            for feature_vector in feature_vectors:
                # Sérialiser en JSON
                data = feature_vector.model_dump_json()
                
                # Publier sur le topic
                self.producer.produce(
                    settings.kafka_topic_output,
                    value=data.encode('utf-8'),
                    callback=self._delivery_callback
                )
            
            # Flush pour garantir la livraison
            self.producer.flush()
            
            logger.info(f"Lot de {len(feature_vectors)} vecteurs de features publié")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication du lot: {e}", exc_info=True)
            raise
    
    def publish_feature_message(self, message: Dict[str, Any]):
        """
        Publie un message de features groupées sur Kafka
        (format attendu par detection-anomalies et prediction-rul)
        
        Args:
            message: Dictionnaire avec asset_id, sensor_id, features, timestamp, metadata
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        try:
            # Sérialiser en JSON
            data = json.dumps(message)
            
            # Publier sur le topic avec asset_id comme key pour partitionnement
            asset_id = message.get("asset_id", "unknown")
            self.producer.produce(
                settings.kafka_topic_output,
                key=asset_id.encode('utf-8'),
                value=data.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Flush pour garantir la livraison
            self.producer.poll(0)
            
            logger.debug(f"Message de features publié: {asset_id}/{message.get('sensor_id', 'unknown')} ({len(message.get('features', {}))} features)")
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication du message: {e}", exc_info=True)
            raise
    
    def close(self):
        """Ferme le producer Kafka"""
        if self.producer:
            self.producer.flush()
            self.producer = None
            logger.info("Producer Kafka fermé")

