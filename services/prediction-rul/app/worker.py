"""
Worker Kafka pour traitement temps-réel des prédictions RUL
"""
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any

from app.config import settings
from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService
from app.api.rul import get_rul_prediction_service  # Use singleton instance
from app.database.postgresql import PostgreSQLService
from app.models.rul_data import RULPredictionRequest, RULPredictionResult

logger = logging.getLogger(__name__)


class RULPredictionWorker:
    """Worker pour traitement temps-réel des prédictions RUL depuis Kafka"""
    
    def __init__(self):
        """Initialise le worker"""
        import sys
        print("=" * 60, file=sys.stderr)
        print("Initialisation du RUL Prediction Worker...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        logger.info("=" * 60)
        logger.info("Initialisation du RUL Prediction Worker...")
        logger.info("=" * 60)
        
        self.running = False
        
        try:
            print("Step 1: Création du service Kafka Consumer...", file=sys.stderr)
            logger.info("Création du service Kafka Consumer...")
            self.kafka_consumer = KafkaConsumerService()
            print("✓ Kafka Consumer créé", file=sys.stderr)
            logger.info("✓ Kafka Consumer créé")
            
            print("Step 2: Création du service Kafka Producer...", file=sys.stderr)
            logger.info("Création du service Kafka Producer...")
            self.kafka_producer = KafkaProducerService()
            print("✓ Kafka Producer créé", file=sys.stderr)
            logger.info("✓ Kafka Producer créé")
            
            print("Step 3: Récupération du service RUL Prediction (singleton)...", file=sys.stderr)
            logger.info("Récupération du service RUL Prediction (singleton)...")
            logger.info("  (Cela peut prendre du temps si PyTorch/MLflow sont en cours d'initialisation)...")
            # Use the same singleton instance as the API
            self.rul_prediction_service = get_rul_prediction_service()
            print("✓ RUL Prediction Service récupéré (instance partagée avec l'API)", file=sys.stderr)
            logger.info("✓ RUL Prediction Service récupéré (instance partagée avec l'API)")
            
            print("Step 4: Création du service PostgreSQL...", file=sys.stderr)
            logger.info("Création du service PostgreSQL...")
            self.postgresql_service = PostgreSQLService()
            print("✓ PostgreSQL Service créé", file=sys.stderr)
            logger.info("✓ PostgreSQL Service créé")
            
            # S'abonner au topic des features
            print(f"Step 5: Abonnement au topic Kafka: {settings.kafka_topic_input_features}", file=sys.stderr)
            logger.info(f"Abonnement au topic Kafka: {settings.kafka_topic_input_features}")
            if self.kafka_consumer.consumer:
                self.kafka_consumer.subscribe([settings.kafka_topic_input_features])
                print("✓ Abonné au topic Kafka", file=sys.stderr)
                logger.info("✓ Abonné au topic Kafka")
            else:
                print("⚠ Kafka Consumer non disponible, abonnement ignoré", file=sys.stderr)
                logger.warning("Kafka Consumer non disponible, abonnement ignoré")
            
            # Gestion des signaux
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("=" * 60, file=sys.stderr)
            print("✓ RUL Prediction Worker initialisé avec succès", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            logger.info("=" * 60)
            logger.info("✓ RUL Prediction Worker initialisé avec succès")
            logger.info("=" * 60)
        except Exception as e:
            print("=" * 60, file=sys.stderr)
            print(f"✗ Erreur lors de l'initialisation du worker: {e}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            logger.error("=" * 60)
            logger.error(f"✗ Erreur lors de l'initialisation du worker: {e}", exc_info=True)
            logger.error("=" * 60)
            raise
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre"""
        logger.info(f"Signal {signum} reçu, arrêt du worker...")
        self.stop()
    
    def start(self):
        """Démarre le worker en mode streaming"""
        if self.running:
            logger.warning("Worker déjà en cours d'exécution")
            return
        
        self.running = True
        logger.info("Démarrage du RUL Prediction Worker...")
        
        # Vérifier que les modèles sont entraînés dans cette instance
        if not self.rul_prediction_service.is_ready():
            logger.warning("Aucun modèle n'est entraîné dans cette instance du worker.")
            logger.warning("Note: Les modèles entraînés via l'API ne sont pas partagés avec le worker (processus séparés).")
            logger.warning("Solution: Entraînez les modèles via POST /api/v1/rul/train APRÈS avoir démarré le worker,")
            logger.warning("         OU redémarrez le worker après l'entraînement pour qu'il utilise les modèles persistés.")
            logger.warning("Le worker démarrera mais les prédictions échoueront jusqu'à ce que les modèles soient entraînés.")
        
        try:
            self._start_streaming_mode()
        except Exception as e:
            logger.error(f"Erreur dans le worker: {e}", exc_info=True)
            raise
        finally:
            self.stop()
    
    def _start_streaming_mode(self):
        """Mode streaming : consomme et traite les messages en continu"""
        import sys
        print("Mode streaming activé, consommation des features...", file=sys.stderr)
        logger.info("Mode streaming activé, consommation des features...")
        
        if not self.kafka_consumer or not self.kafka_consumer.consumer:
            print("⚠ ERREUR: Kafka Consumer non disponible!", file=sys.stderr)
            logger.error("Kafka Consumer non disponible, impossible de démarrer le mode streaming")
            raise RuntimeError("Kafka Consumer non disponible")
        
        print(f"Consumer prêt, attente de messages sur le topic: {settings.kafka_topic_input_features}", file=sys.stderr)
        logger.info(f"Consumer prêt, attente de messages sur le topic: {settings.kafka_topic_input_features}")
        
        def process_feature_message(message_data: Dict[str, Any]):
            """Traite un message de features"""
            try:
                # Support both formats:
                # 1. New grouped format: {"asset_id": "...", "features": {...}, ...}
                # 2. Old individual ExtractedFeature format: {"name": "...", "value": ..., "metadata": {"asset_id": "...", ...}}
                
                # Try new format first
                asset_id = message_data.get('asset_id')
                sensor_id = message_data.get('sensor_id')
                features = message_data.get('features', {})
                timestamp = message_data.get('timestamp')
                metadata = message_data.get('metadata', {})
                
                # If asset_id not at root, try old format (ExtractedFeature with metadata)
                if not asset_id:
                    metadata_inner = message_data.get('metadata', {})
                    asset_id = metadata_inner.get('asset_id')
                    sensor_id = metadata_inner.get('sensor_id', sensor_id)
                    # For old format, convert single feature to features dict
                    if asset_id and 'name' in message_data and 'value' in message_data:
                        features = {message_data.get('name'): message_data.get('value')}
                        timestamp_str = metadata_inner.get('timestamp')
                        if timestamp_str:
                            try:
                                if isinstance(timestamp_str, str):
                                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                else:
                                    timestamp = timestamp_str
                            except:
                                timestamp = None
                        metadata = metadata_inner
                        logger.debug(f"Message en ancien format détecté, converti: {asset_id}/{sensor_id}")
                
                if not asset_id:
                    logger.warning("Message sans asset_id, ignoré")
                    return
                
                if not features:
                    logger.warning(f"Message sans features pour {asset_id}, ignoré")
                    return
                
                # Convertir timestamp si string
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now(timezone.utc)
                elif timestamp is None:
                    timestamp = datetime.now(timezone.utc)
                
                # Créer la requête de prédiction
                prediction_request = RULPredictionRequest(
                    asset_id=asset_id,
                    sensor_id=sensor_id,
                    features=features,
                    timestamp=timestamp,
                    metadata=metadata
                )
                
                # Prédire la RUL
                try:
                    rul_result = self.rul_prediction_service.predict_rul(
                        prediction_request,
                        use_ensemble=True
                    )
                    
                    # Publier la prédiction sur Kafka
                    self.kafka_producer.publish_rul_prediction(rul_result)
                    
                    # Journaliser dans PostgreSQL
                    try:
                        self.postgresql_service.insert_rul_prediction(rul_result)
                    except Exception as e:
                        logger.warning(f"Impossible de journaliser la prédiction RUL dans PostgreSQL: {e}")
                    
                    logger.info(
                        f"RUL prédite pour {asset_id}: {rul_result.rul_prediction:.2f} "
                        f"(intervalle: [{rul_result.confidence_interval_lower:.2f}, "
                        f"{rul_result.confidence_interval_upper:.2f}])"
                    )
                    
                except RuntimeError as e:
                    logger.warning(f"Modèle non entraîné, prédiction ignorée pour {asset_id}: {e}")
                except Exception as e:
                    logger.error(f"Erreur lors de la prédiction RUL pour {asset_id}: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
        
        # Consommer en continu
        self.kafka_consumer.consume_features_continuous(
            callback=process_feature_message,
            timeout=1.0
        )
    
    def stop(self):
        """Arrête le worker"""
        if not self.running:
            return
        
        logger.info("Arrêt du RUL Prediction Worker...")
        self.running = False
        
        try:
            self.kafka_consumer.close()
            self.kafka_producer.close()
            logger.info("Worker arrêté proprement")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du worker: {e}", exc_info=True)


def main():
    """Point d'entrée principal du worker"""
    import sys
    
    # Configuration du logging - force INFO level for visibility
    logging.basicConfig(
        level=logging.INFO,  # Force INFO level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    
    print("=" * 60)
    print("Starting RUL Prediction Worker...")
    print("=" * 60)
    logger.info("Starting RUL Prediction Worker...")
    
    try:
        print("Creating worker instance...")
        logger.info("Creating worker instance...")
        worker = RULPredictionWorker()
        print("Worker instance created successfully!")
        logger.info("Worker instance created, starting...")
        
        print("Starting worker (will consume from Kafka)...")
        logger.info("Starting worker...")
        worker.start()
    except KeyboardInterrupt:
        print("\nInterruption clavier")
        logger.info("Interruption clavier")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Stopping worker...")
        logger.info("Stopping worker...")
        try:
            worker.stop()
        except:
            pass
        print("Worker stopped")
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()

