"""
Worker en arrière-plan pour consommer Kafka et extraire les features
"""
import logging
import signal
import sys
import asyncio
import threading
from typing import Optional, List
from datetime import datetime

from app.config import settings
from app.services.kafka_consumer import KafkaConsumerService
from app.services.feature_extraction_service import FeatureExtractionService
from app.models.feature_data import PreprocessedDataReference, WindowedDataReference

logger = logging.getLogger(__name__)


class FeatureExtractionWorker:
    """Worker en arrière-plan pour l'extraction de features"""
    
    def __init__(self):
        import sys
        print("=" * 60, file=sys.stderr)
        print("Initialisation du Feature Extraction Worker...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        logger.info("=" * 60)
        logger.info("Initialisation du Feature Extraction Worker...")
        logger.info("=" * 60)
        
        self.running = False
        
        try:
            print("Step 1: Création du service Kafka Consumer...", file=sys.stderr)
            logger.info("Création du service Kafka Consumer...")
            self.kafka_consumer = KafkaConsumerService()
            print("✓ Kafka Consumer créé", file=sys.stderr)
            logger.info("✓ Kafka Consumer créé")
            
            print("Step 2: Création du service Feature Extraction...", file=sys.stderr)
            logger.info("Création du service Feature Extraction...")
            self.feature_extraction_service = FeatureExtractionService()
            print("✓ Feature Extraction Service créé", file=sys.stderr)
            logger.info("✓ Feature Extraction Service créé")
            
            self.mode = "streaming"  # "streaming" ou "batch"
            
            # Gestion des signaux
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("=" * 60, file=sys.stderr)
            print("✓ Feature Extraction Worker initialisé avec succès", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            logger.info("=" * 60)
            logger.info("✓ Feature Extraction Worker initialisé avec succès")
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
    
    def start(self, mode: str = "streaming"):
        """
        Démarre le worker
        
        Args:
            mode: Mode de traitement ("streaming" ou "batch")
        """
        if self.running:
            logger.warning("Worker déjà en cours d'exécution")
            return
        
        self.mode = mode
        self.running = True
        
        logger.info(f"Démarrage du worker d'extraction de features en mode: {mode}")
        
        try:
            # Démarrer la consommation depuis les topics
            if mode == "streaming":
                self._start_streaming_mode()
            elif mode == "batch":
                self._start_batch_mode()
            else:
                logger.error(f"Mode inconnu: {mode}")
                return
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du worker: {e}", exc_info=True)
            self.running = False
            raise
    
    def _start_streaming_mode(self):
        """Démarre le mode streaming"""
        logger.info("Démarrage du mode streaming...")
        
        # Consommer depuis le topic preprocessed-data
        def preprocessed_data_handler(preprocessed_data: List[PreprocessedDataReference]):
            """Handler pour les données prétraitées"""
            try:
                # Créer une nouvelle boucle d'événements pour cette tâche
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.feature_extraction_service.process_preprocessed_data(
                            preprocessed_data,
                            mode="streaming"
                        )
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Erreur lors du traitement des données prétraitées: {e}", exc_info=True)
        
        # Subscription will happen automatically in consume_preprocessed_data
        # No need to subscribe here
        
        # Démarrer la consommation dans un thread séparé
        def consume_thread():
            try:
                logger.info(f"Thread de consommation démarré, attente de messages sur: {settings.kafka_topic_input_preprocessed}")
                while self.running:
                    try:
                        self.kafka_consumer.consume_preprocessed_data(
                            preprocessed_data_handler,
                            timeout=1.0,
                            max_messages=100
                        )
                    except KeyboardInterrupt:
                        logger.info("Interruption reçue dans le thread")
                        self.running = False
                        break
                    except Exception as e:
                        logger.error(f"Erreur lors de la consommation: {e}", exc_info=True)
                        # Continue instead of breaking to keep the thread alive
                        import time
                        time.sleep(1)
            except Exception as e:
                logger.error(f"Erreur fatale dans le thread de consommation: {e}", exc_info=True)
                self.running = False
        
        thread = threading.Thread(target=consume_thread, daemon=True)
        thread.start()
        logger.info("Thread de consommation démarré")
        logger.info("Worker en cours d'exécution en arrière-plan.")
        # Don't join the thread - it's a daemon thread and will run in the background
        # This allows the FastAPI startup to complete without blocking
    
    def _start_batch_mode(self):
        """Démarre le mode batch"""
        logger.info("Démarrage du mode batch...")
        
        # Consommer depuis le topic windowed-data
        def windowed_data_handler(windowed_data: WindowedDataReference):
            """Handler pour les fenêtres de données"""
            try:
                # Créer une nouvelle boucle d'événements pour cette tâche
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.feature_extraction_service.process_windowed_data(windowed_data)
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Erreur lors du traitement des fenêtres de données: {e}", exc_info=True)
        
        # Démarrer la consommation dans un thread séparé
        def consume_thread():
            try:
                while self.running:
                    self.kafka_consumer.consume_windowed_data(
                        windowed_data_handler,
                        timeout=1.0
                    )
            except Exception as e:
                logger.error(f"Erreur dans le thread de consommation: {e}", exc_info=True)
                self.running = False
        
        thread = threading.Thread(target=consume_thread, daemon=True)
        thread.start()
        logger.info("Thread de consommation démarré")
    
    def stop(self):
        """Arrête le worker"""
        if not self.running:
            return
        
        self.running = False
        
        # Arrêter la consommation Kafka
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        # Fermer les services
        if self.feature_extraction_service:
            # Fermer les services si nécessaire
            pass
        
        logger.info("Worker d'extraction de features arrêté")

