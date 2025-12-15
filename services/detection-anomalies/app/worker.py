"""
Worker en arrière-plan pour consommer Kafka et détecter les anomalies
"""
import logging
import signal
import sys
import threading
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from app.config import settings
from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService
from app.api.anomalies import get_anomaly_detection_service
from app.database.postgresql import PostgreSQLService
from app.models.anomaly_data import AnomalyDetectionRequest

logger = logging.getLogger(__name__)


class AnomalyDetectionWorker:
    """Worker pour consommer les features depuis Kafka et détecter les anomalies"""
    
    def __init__(self):
        import sys
        print("=" * 60, file=sys.stderr)
        print("Initialisation du worker de détection d'anomalies...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        logger.info("=" * 60)
        logger.info("Initialisation du worker...")
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
            
            print("Step 3: Création du service Anomaly Detection...", file=sys.stderr)
            logger.info("Création du service Anomaly Detection...")
            logger.info("  (Cela peut prendre du temps si PyTorch/MLflow sont en cours d'initialisation)...")
            # Use singleton to share models between API and worker
            self.anomaly_detection_service = get_anomaly_detection_service()
            print("✓ Anomaly Detection Service créé (singleton)", file=sys.stderr)
            logger.info("✓ Anomaly Detection Service créé (singleton avec modèles auto-chargés)")
            
            print("Step 4: Création du service PostgreSQL...", file=sys.stderr)
            logger.info("Création du service PostgreSQL...")
            self.postgresql_service = PostgreSQLService()
            print("✓ PostgreSQL Service créé", file=sys.stderr)
            logger.info("✓ PostgreSQL Service créé")
            
            # Gestion des signaux
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("=" * 60, file=sys.stderr)
            print("✓ Worker initialisé avec succès", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            logger.info("=" * 60)
            logger.info("✓ Worker initialisé avec succès")
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
        """Handler pour les signaux d'arrêt"""
        logger.info(f"Signal {signum} reçu, arrêt du worker...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Démarre le worker"""
        if self.running:
            logger.warning("Worker déjà en cours d'exécution")
            return
        
        logger.info("Démarrage du worker de détection d'anomalies...")
        self.running = True
        
        try:
            # Vérifier que les modèles sont entraînés
            if not self.anomaly_detection_service.is_ready():
                logger.warning("=" * 60)
                logger.warning("⚠️  Aucun modèle de détection d'anomalies n'est entraîné")
                logger.warning("Le worker démarrera mais ignorera les messages reçus.")
                logger.warning("")
                logger.warning("Pour entraîner les modèles:")
                logger.warning("  1. Via API: POST http://localhost:8084/api/v1/anomalies/train")
                logger.warning("  2. Les modèles seront sauvegardés dans: services/detection-anomalies/models/")
                logger.warning("  3. Le worker chargera automatiquement les modèles au prochain démarrage")
                logger.warning("=" * 60)
            else:
                model_status = self.anomaly_detection_service.get_model_status()
                trained_models = [name for name, info in model_status.items() if info.get('is_trained', False)]
                logger.info(f"✓ Modèles entraînés et prêts: {', '.join(trained_models)}")
            
            # Vérifier que le consumer est disponible
            if not self.kafka_consumer.consumer:
                logger.error("Consumer Kafka non disponible. Impossible de démarrer le worker.")
                raise RuntimeError("Consumer Kafka non initialisé")
            
            # Démarrer la consommation dans un thread séparé
            def consume_thread():
                try:
                    logger.info("Démarrage de la consommation Kafka...")
                    self.kafka_consumer.consume_features_continuous(
                        callback=self._process_feature,
                        timeout=1.0
                    )
                except Exception as e:
                    logger.error(f"Erreur dans le thread de consommation: {e}", exc_info=True)
                    self.running = False
            
            thread = threading.Thread(target=consume_thread, daemon=True)
            thread.start()
            logger.info("Thread de consommation démarré")
            
            # Attendre que le thread se termine
            thread.join()
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du worker: {e}", exc_info=True)
            self.running = False
            raise
    
    def _process_feature(self, feature_data: Dict[str, Any]):
        """
        Traite une feature reçue depuis Kafka
        
        Args:
            feature_data: Données de la feature (dict JSON)
        """
        try:
            # Support both formats:
            # 1. New grouped format: {"asset_id": "...", "features": {...}, ...}
            # 2. Old individual ExtractedFeature format: {"name": "...", "value": ..., "metadata": {"asset_id": "...", ...}}
            
            # Try new format first
            asset_id = feature_data.get("asset_id")
            sensor_id = feature_data.get("sensor_id")
            features = feature_data.get("features", {})
            timestamp = feature_data.get("timestamp")
            
            # If asset_id not at root, try old format (ExtractedFeature with metadata)
            if not asset_id:
                metadata = feature_data.get("metadata", {})
                asset_id = metadata.get("asset_id")
                sensor_id = metadata.get("sensor_id", sensor_id)
                # For old format, convert single feature to features dict
                if asset_id and "name" in feature_data and "value" in feature_data:
                    features = {feature_data.get("name"): feature_data.get("value")}
                    timestamp_str = metadata.get("timestamp")
                    if timestamp_str:
                        try:
                            if isinstance(timestamp_str, str):
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                timestamp = timestamp_str
                        except:
                            timestamp = None
                    logger.debug(f"Message en ancien format détecté, converti: {asset_id}/{sensor_id}")
            
            if not asset_id:
                logger.warning("Feature sans asset_id, ignorée")
                return
            
            if not features:
                logger.warning(f"Feature sans features pour {asset_id}, ignorée")
                return
            
            # Convertir timestamp si nécessaire
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(timezone.utc)
            elif timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Vérifier que les modèles sont prêts
            if not self.anomaly_detection_service.is_ready():
                logger.info(f"✓ Message reçu pour {asset_id}/{sensor_id} mais modèles non entraînés (ignoré)")
                return
            
            logger.info(f"✓ Traitement du message pour {asset_id}/{sensor_id} avec {len(features)} features")
            
            # Créer la requête de détection
            request = AnomalyDetectionRequest(
                asset_id=asset_id,
                sensor_id=sensor_id,
                features=features,
                timestamp=timestamp,
                metadata=feature_data.get("metadata", {})
            )
            
            # Détecter l'anomalie
            result = self.anomaly_detection_service.detect_anomaly(request)
            
            # Publier le résultat sur Kafka
            self.kafka_producer.publish_anomaly(result)
            
            # Journaliser l'anomalie dans la base de données si détectée
            if result.is_anomaly:
                try:
                    self.postgresql_service.insert_anomaly(result)
                except Exception as e:
                    logger.warning(f"Impossible de journaliser l'anomalie dans la base de données: {e}")
                
                logger.warning(
                    f"Anomalie détectée: {asset_id} "
                    f"(score={result.final_score:.3f}, "
                    f"criticality={result.criticality.value})"
                )
            else:
                logger.debug(f"Pas d'anomalie: {asset_id} (score={result.final_score:.3f})")
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la feature: {e}", exc_info=True)
            # Ne pas lever l'exception pour continuer le traitement
    
    def stop(self):
        """Arrête le worker"""
        logger.info("Arrêt du worker...")
        self.running = False
        
        try:
            self.kafka_consumer.close()
            self.kafka_producer.close()
            self.postgresql_service.close()
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture: {e}", exc_info=True)
        
        logger.info("Worker arrêté")


def main():
    """Point d'entrée principal pour le worker"""
    import sys
    
    # Force INFO level for visibility
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True
    )
    
    print("=" * 60)
    print("Starting Anomaly Detection Worker")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("Starting Anomaly Detection Worker")
    logger.info("=" * 60)
    
    try:
        print("Creating worker instance...")
        logger.info("Creating worker instance...")
        worker = AnomalyDetectionWorker()
        print("Worker instance created successfully!")
        logger.info("Worker instance created successfully")
        
        print("Starting worker...")
        logger.info("Starting worker...")
        worker.start()
    except KeyboardInterrupt:
        print("\nInterruption reçue")
        logger.info("Interruption reçue")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        logger.error(f"Fatal error in worker: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise
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

