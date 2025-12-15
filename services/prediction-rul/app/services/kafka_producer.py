"""
Service Kafka Producer pour publier les prédictions RUL
"""
import logging
import json
from typing import List, Optional
from confluent_kafka import Producer, KafkaError

from app.config import settings
from app.models.rul_data import RULPredictionResult

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """Service pour publier des messages vers Kafka"""
    
    def __init__(self):
        """Initialise le producer Kafka"""
        self.producer: Optional[Producer] = None
        self._create_producer()
    
    def _create_producer(self):
        """Crée et configure le producer Kafka"""
        try:
            config = {
                'bootstrap.servers': settings.kafka_bootstrap_servers
            }
            
            self.producer = Producer(config)
            logger.info(f"Producer Kafka initialisé pour: {settings.kafka_bootstrap_servers}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du producer Kafka: {e}", exc_info=True)
            self.producer = None
    
    def _delivery_callback(self, err, msg):
        """Callback pour la confirmation de livraison"""
        if err is not None:
            logger.error(f"Erreur lors de la livraison du message: {err}")
        else:
            logger.debug(f"Message livré: {msg.topic()}[{msg.partition()}]@{msg.offset()}")
    
    def publish_rul_prediction(
        self,
        rul_result: RULPredictionResult,
        topic: Optional[str] = None
    ) -> bool:
        """
        Publie une prédiction RUL vers Kafka
        
        Args:
            rul_result: Résultat de prédiction RUL
            topic: Topic Kafka (utilise config si None)
        
        Returns:
            True si succès, False sinon
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        topic = topic or settings.kafka_topic_output_rul
        
        try:
            # Convertir en JSON
            message_data = rul_result.model_dump_json()
            
            # Publier
            self.producer.produce(
                topic,
                value=message_data.encode('utf-8'),
                key=rul_result.asset_id.encode('utf-8'),
                callback=self._delivery_callback
            )
            
            # Flush pour s'assurer que le message est envoyé
            self.producer.poll(0)
            
            logger.debug(f"Prédiction RUL publiée pour {rul_result.asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication de la prédiction RUL: {e}", exc_info=True)
            return False
    
    def publish_rul_predictions_batch(
        self,
        rul_results: List[RULPredictionResult],
        topic: Optional[str] = None
    ) -> int:
        """
        Publie plusieurs prédictions RUL en batch
        
        Args:
            rul_results: Liste de résultats de prédiction RUL
            topic: Topic Kafka (utilise config si None)
        
        Returns:
            Nombre de messages publiés avec succès
        """
        if not self.producer:
            raise RuntimeError("Producer Kafka non initialisé")
        
        topic = topic or settings.kafka_topic_output_rul
        published_count = 0
        
        try:
            for rul_result in rul_results:
                try:
                    message_data = rul_result.model_dump_json()
                    
                    self.producer.produce(
                        topic,
                        value=message_data.encode('utf-8'),
                        key=rul_result.asset_id.encode('utf-8'),
                        callback=self._delivery_callback
                    )
                    
                    published_count += 1
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la publication d'une prédiction: {e}", exc_info=True)
                    continue
            
            # Flush pour s'assurer que tous les messages sont envoyés
            self.producer.flush()
            
            logger.info(f"{published_count}/{len(rul_results)} prédictions RUL publiées")
            return published_count
            
        except Exception as e:
            logger.error(f"Erreur lors de la publication batch: {e}", exc_info=True)
            return published_count
    
    def close(self):
        """Ferme le producer Kafka"""
        if self.producer:
            try:
                self.producer.flush(timeout=10)
                logger.info("Producer Kafka fermé")
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du producer: {e}", exc_info=True)
            finally:
                self.producer = None

