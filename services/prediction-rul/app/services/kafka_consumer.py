"""
Service Kafka Consumer pour consommer les features extraites
"""
import logging
import json
from typing import Callable, Optional, Dict, Any
from confluent_kafka import Consumer, KafkaError, KafkaException

from app.config import settings

logger = logging.getLogger(__name__)


class KafkaConsumerService:
    """Service pour consommer des messages depuis Kafka"""
    
    def __init__(self):
        """Initialise le consumer Kafka"""
        logger.debug("Initialisation Kafka Consumer...")
        self.consumer: Optional[Consumer] = None
        try:
            self._create_consumer()
            if self.consumer:
                logger.info("✓ Kafka Consumer initialisé")
            else:
                logger.warning("Kafka Consumer non initialisé")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation Kafka Consumer: {e}")
            self.consumer = None
    
    def _create_consumer(self):
        """Crée et configure le consumer Kafka"""
        try:
            config = {
                'bootstrap.servers': settings.kafka_bootstrap_servers,
                'group.id': settings.kafka_consumer_group,
                'auto.offset.reset': settings.kafka_auto_offset_reset,
                'enable.auto.commit': settings.kafka_enable_auto_commit
            }
            
            self.consumer = Consumer(config)
            logger.info(f"Consumer Kafka initialisé pour le groupe: {settings.kafka_consumer_group}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du consumer Kafka: {e}", exc_info=True)
            self.consumer = None
    
    def subscribe(self, topics: list):
        """S'abonne à des topics Kafka"""
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        try:
            self.consumer.subscribe(topics)
            logger.info(f"Abonné aux topics: {topics}")
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux topics: {e}", exc_info=True)
            raise
    
    def consume_features(
        self,
        callback: Callable[[Dict[str, Any]], None],
        max_messages: Optional[int] = None,
        timeout: float = 1.0
    ) -> int:
        """
        Consomme des messages de features et appelle le callback pour chaque message
        
        Args:
            callback: Fonction appelée pour chaque message (reçoit un dict)
            max_messages: Nombre maximum de messages à consommer (None = infini)
            timeout: Timeout en secondes pour poll()
        
        Returns:
            Nombre de messages consommés
        """
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        messages_consumed = 0
        
        try:
            while True:
                if max_messages and messages_consumed >= max_messages:
                    break
                
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Fin de partition atteinte: {msg.topic()}[{msg.partition()}]")
                        continue
                    else:
                        logger.error(f"Erreur Kafka: {msg.error()}")
                        continue
                
                try:
                    # Désérialiser le message JSON
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    # Appeler le callback
                    callback(message_data)
                    
                    messages_consumed += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Message JSON invalide ignoré: {e}")
                    messages_consumed += 1  # Incrémenter pour éviter boucle infinie
                    continue
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
                    messages_consumed += 1
                    continue
        
        except KeyboardInterrupt:
            logger.info("Interruption clavier, arrêt de la consommation")
        except Exception as e:
            logger.error(f"Erreur lors de la consommation: {e}", exc_info=True)
            raise
        
        return messages_consumed
    
    def consume_features_continuous(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout: float = 1.0
    ):
        """
        Consomme des messages en continu (infini)
        
        Args:
            callback: Fonction appelée pour chaque message
            timeout: Timeout en secondes pour poll()
        """
        self.consume_features(callback, max_messages=None, timeout=timeout)
    
    def close(self):
        """Ferme le consumer Kafka"""
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Consumer Kafka fermé")
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du consumer: {e}", exc_info=True)
            finally:
                self.consumer = None

