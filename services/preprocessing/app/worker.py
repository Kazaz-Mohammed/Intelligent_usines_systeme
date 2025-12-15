"""
Worker principal pour le traitement en arrière-plan
"""
import logging
import signal
import sys
from threading import Thread

from app.config import settings
from app.services.preprocessing_service import PreprocessingService
from app.database.timescaledb import TimescaleDBService

logger = logging.getLogger(__name__)


class PreprocessingWorker:
    """Worker qui exécute le pipeline de prétraitement"""
    
    def __init__(self):
        self.preprocessing_service = PreprocessingService()
        self.timescaledb_service = TimescaleDBService()
        self.running = False
        self.processing_thread: Optional[Thread] = None
        
        # Enregistrer les handlers de signaux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler pour les signaux d'arrêt"""
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        self.stop()
        sys.exit(0)
    
    def start(self, mode: str = "streaming"):
        """
        Démarre le worker
        
        Args:
            mode: "streaming" ou "batch"
        """
        if self.running:
            logger.warning("Worker déjà en cours d'exécution")
            return
        
        self.running = True
        logger.info(f"Démarrage du worker en mode: {mode}")
        
        # Démarrer le traitement dans un thread séparé
        self.processing_thread = Thread(
            target=self._run_processing,
            args=(mode,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Worker démarré")
    
    def _run_processing(self, mode: str):
        """Exécute le pipeline de traitement"""
        try:
            # Intégrer TimescaleDB dans le service de prétraitement
            # Pour l'instant, on utilise le service directement
            # TODO: Intégrer le stockage dans le pipeline
            
            self.preprocessing_service.start_processing_loop(mode)
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de traitement: {e}", exc_info=True)
            self.running = False
    
    def stop(self):
        """Arrête le worker"""
        if not self.running:
            return
        
        logger.info("Arrêt du worker...")
        self.running = False
        
        # Arrêter le service de prétraitement
        self.preprocessing_service.stop()
        
        # Fermer TimescaleDB
        self.timescaledb_service.close()
        
        logger.info("Worker arrêté")
    
    def wait(self):
        """Attend que le worker se termine"""
        if self.processing_thread:
            self.processing_thread.join()


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Service de prétraitement")
    parser.add_argument(
        "--mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Mode de traitement (streaming ou batch)"
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer et démarrer le worker
    worker = PreprocessingWorker()
    
    try:
        worker.start(mode=args.mode)
        worker.wait()
    except KeyboardInterrupt:
        logger.info("Interruption clavier, arrêt...")
        worker.stop()


if __name__ == "__main__":
    main()

