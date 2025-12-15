"""
Application FastAPI principale pour le service Extraction Features
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.features import router as features_router
from app.worker import FeatureExtractionWorker
from app.api.features import FeatureExtractionServiceSingleton

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation du worker
worker = FeatureExtractionWorker()

# Initialisation du service d'extraction de features
feature_extraction_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    global feature_extraction_service
    
    # Startup
    logger.info(f"Démarrage du service {settings.service_name}")
    logger.info(f"Configuration Kafka: {settings.kafka_bootstrap_servers}")
    logger.info(f"Topic input (preprocessed): {settings.kafka_topic_input_preprocessed}")
    logger.info(f"Topic input (windowed): {settings.kafka_topic_input_windowed}")
    logger.info(f"Topic output: {settings.kafka_topic_output}")
    logger.info(f"TimescaleDB: {settings.database_host}:{settings.database_port}/{settings.database_name}")
    logger.info(f"Feast: {'activé' if settings.feast_enable else 'désactivé'}")
    logger.info(f"Features temporelles: {'activées' if settings.enable_temporal_features else 'désactivées'}")
    logger.info(f"Features fréquentielles: {'activées' if settings.enable_frequency_features else 'désactivées'}")
    logger.info(f"Features ondelettes: {'activées' if settings.enable_wavelet_features else 'désactivées'}")
    logger.info(f"Standardisation: {'activée' if settings.enable_standardization else 'désactivée'}")
    
    # Initialiser le service d'extraction de features
    from app.services.feature_extraction_service import FeatureExtractionService
    feature_extraction_service = FeatureExtractionService()
    FeatureExtractionServiceSingleton.set_instance(feature_extraction_service)
    
    # Démarrer le worker en arrière-plan
    try:
        worker.start(mode="streaming")  # Mode streaming par défaut
        logger.info("Worker d'extraction de features démarré")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du worker: {e}", exc_info=True)
        # Continuer même si le worker ne démarre pas (pour les tests)
    
    yield
    
    # Shutdown
    logger.info(f"Arrêt du service {settings.service_name}")
    
    # Arrêter le worker
    try:
        worker.stop()
        logger.info("Worker d'extraction de features arrêté")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du worker: {e}", exc_info=True)


app = FastAPI(
    title="Extraction Features Service",
    description="Service d'extraction de caractéristiques temporelles et fréquentielles pour la maintenance prédictive",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(features_router, prefix="/api/v1/features", tags=["Features"])


@app.get("/")
async def root():
    return {"message": "Extraction Features Service is running"}


@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy", "service": settings.service_name, "version": "0.1.0"}

