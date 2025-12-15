"""
Service de consommation Kafka pour Detection Anomalies
"""
import logging
import json
from typing import Optional, Callable, Dict, Any
from confluent_kafka import Consumer, KafkaError, KafkaException

from app.config import settings

logger = logging.getLogger(__name__)


class KafkaConsumerService:
    """Service pour consommer des features extraites depuis Kafka"""
    
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
        self._create_consumer()
    
    def _create_consumer(self):
        """Crée le consumer Kafka"""
        try:
            logger.debug(f"Création du consumer Kafka pour {settings.kafka_bootstrap_servers}...")
            self.consumer = Consumer(self.config)
            logger.info(f"Consumer Kafka créé: {settings.kafka_bootstrap_servers}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du consumer Kafka: {e}", exc_info=True)
            # Don't raise - allow service to start even if Kafka is temporarily unavailable
            logger.warning("Le consumer Kafka n'a pas pu être créé, mais le service continuera")
    
    def consume_features(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout: float = 1.0,
        max_messages: int = 100
    ):
        """
        Consomme des features extraites depuis Kafka
        
        Args:
            callback: Fonction à appeler avec les features consommées
            timeout: Timeout en secondes pour la consommation
            max_messages: Nombre maximum de messages à consommer par batch
        """
        if not self.consumer:
            self._create_consumer()
        
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        try:
            # S'abonner au topic
            self.consumer.subscribe([settings.kafka_topic_input_features])
            logger.info(f"Abonné au topic: {settings.kafka_topic_input_features}")
            
            messages_consumed = 0
            
            while messages_consumed < max_messages:
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    if messages_consumed == 0:
                        continue  # Pas de messages, continuer à attendre
                    else:
                        break  # Batch terminé
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"Fin de partition atteinte: {msg.topic()}[{msg.partition()}]")
                        continue
                    else:
                        logger.error(f"Erreur Kafka: {msg.error()}")
                        continue  # Continuer au lieu de lever une exception
                
                # Désérialiser le message
                try:
                    feature_data = json.loads(msg.value().decode('utf-8'))
                    callback(feature_data)
                    messages_consumed += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de désérialisation JSON: {e}")
                    messages_consumed += 1  # Compter même les messages invalides pour éviter la boucle infinie
                    continue
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
                    messages_consumed += 1  # Compter même les messages en erreur pour éviter la boucle infinie
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Interruption reçue, arrêt de la consommation")
        except Exception as e:
            logger.error(f"Erreur lors de la consommation: {e}", exc_info=True)
            raise
    
    def consume_features_continuous(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout: float = 1.0
    ):
        """
        Consomme des features en continu depuis Kafka
        
        Args:
            callback: Fonction à appeler avec les features consommées
            timeout: Timeout en secondes pour la consommation
        """
        if not self.consumer:
            self._create_consumer()
        
        if not self.consumer:
            raise RuntimeError("Consumer Kafka non initialisé")
        
        try:
            # S'abonner au topic
            self.consumer.subscribe([settings.kafka_topic_input_features])
            logger.info(f"Abonné au topic: {settings.kafka_topic_input_features}")
            
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
                        continue
                
                # Désérialiser le message
                try:
                    logger.info(f"✓ Message reçu: topic={msg.topic()}, partition={msg.partition()}, offset={msg.offset()}")
                    feature_data = json.loads(msg.value().decode('utf-8'))
                    callback(feature_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de désérialisation JSON: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Interruption reçue, arrêt de la consommation")
        except Exception as e:
            logger.error(f"Erreur lors de la consommation: {e}", exc_info=True)
            raise
    
    def close(self):
        """Ferme le consumer Kafka"""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
            logger.info("Consumer Kafka fermé")

